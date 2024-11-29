

from flax import linen as nn
import jax.numpy as jnp
import jax
import modules
import numpy as np
import commons
import attentions
from nsf import SourceModuleHnNSF

class Generator(nn.Module):
    initial_channel:int
    resblock:int
    resblock_kernel_sizes:int
    resblock_dilation_sizes:int
    upsample_rates:int
    upsample_initial_channel:int
    upsample_kernel_sizes:int
    gin_channels:int
    sr:int
    def setup(self):
        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_upsamples = len(self.upsample_rates)
        self.conv_pre = nn.Conv(self.upsample_initial_channel, kernel_size=[7], strides=[1])
        self.scale_factor = np.prod(self.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=self.sr)
        noise_convs = []
        ups = []
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            ups.append(
                    nn.WeightNorm(nn.ConvTranspose(
                        self.upsample_initial_channel // (2 ** (i + 1)),
                        (k,),
                        (u,),
                        transpose_kernel=True))
                )
            if i + 1 < len(self.upsample_rates):
                stride_f0 = np.prod(self.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                noise_convs.append(
                    nn.Conv(
                        features=self.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=[stride_f0 * 2],
                        strides=[stride_f0]
                    )
                )
            else:
                noise_convs.append(
                    nn.Conv(features=self.upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=[1])
                )

        resblocks = []
        for i in range(len(ups)):
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes):
                resblocks.append(modules.ResBlock1(ch, k, d))

        self.conv_post =  nn.Conv(1, kernel_size=[7], strides=1 , use_bias=False)
        self.cond = nn.Conv(self.upsample_initial_channel, 1)
        self.ups = ups
        self.resblocks = resblocks
        self.noise_convs = noise_convs

    def __call__(self, x, f0,g=None,train=True):
        har_source = self.m_source(f0,self.scale_factor,rng=self.make_rng('rnorms'))
        #har_source = har_source.transpose(0,2,1)
        x = self.conv_pre(x)
        x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x,train=train)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x,train=train)
            x = xs / self.num_kernels
        x = nn.leaky_relu(x)
        x = self.conv_post(x)
        x = nn.tanh(x) 
        return x

class TextEncoder(nn.Module):
    #in_channels:int
    out_channels:int
    hidden_channels:int
    filter_channels:int
    n_heads:int
    n_layers:int
    kernel_size:int
    p_dropout:float
    f0:bool = True
    def setup(self):
        self.emb_phone = nn.Dense(features=self.hidden_channels)
        if self.f0 == True:
            self.emb_pitch = nn.Embed(256, self.hidden_channels)
        self.encoder = attentions.Encoder(
            hidden_channels=self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout)
        self.proj = nn.Conv(features=self.out_channels * 2, kernel_size=[1])
    def __call__(self, 
                 phone : jnp.ndarray,
                 pitch : jnp.ndarray,
                 lengths : jnp.ndarray,
                 train=True):
        x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * jnp.sqrt(self.hidden_channels)
        x = nn.leaky_relu(x,0.1)
        #x = x.transpose(0,2,1)  # [b, h, t]
        x_mask = jnp.expand_dims(commons.sequence_mask(lengths, x.shape[1]), 2)
        x = self.encoder(x * x_mask, x_mask,train=train)
        stats = self.proj(x) * x_mask
        m, logs = jnp.split(stats,2, axis=2)
        return m, logs, x_mask

class ResidualCouplingBlock(nn.Module):
    channels:int
    hidden_channels:int
    kernel_size:int
    dilation_rate:int
    n_layers:int
    n_flows:int=4
    gin_channels:int=0
    def setup(
        self
    ):
        flows = []
        for i in range(self.n_flows):
            flows.append(
                modules.ResidualCouplingLayer(
                    self.channels,
                    self.hidden_channels,
                    self.kernel_size,
                    self.dilation_rate,
                    self.n_layers,
                    gin_channels=self.gin_channels,
                    mean_only=True
                )
            )
            flows.append(modules.Flip())
        self.flows=flows

    def __call__(self, x, x_mask, g=None, reverse=False,train=True):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse,train=train)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse,train=train)
        return x


class PosteriorEncoder(nn.Module):
    #in_channels:int
    out_channels:int
    hidden_channels:int
    kernel_size:int
    dilation_rate:int
    n_layers:int
    gin_channels:int=0
    def setup(
        self
    ):
        self.pre = nn.Conv(features=self.hidden_channels, kernel_size=[1])
        self.enc = modules.WN(
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            gin_channels=self.gin_channels
        )
        self.proj = nn.Conv(features=self.out_channels * 2,kernel_size=[1])

    def __call__(self, x, x_lengths,g=None,train=True):
        x_mask = jnp.expand_dims(commons.sequence_mask(x_lengths, x.shape[1]), 2)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g,train=train)
        stats = self.proj(x) * x_mask
        m, logs = jnp.split(stats,2, axis=2)
        z = (m + jax.random.normal(self.make_rng('rnorms'),m.shape) * jnp.exp(logs)) * x_mask
        return z, m, logs, x_mask
    
class SynthesizerTrn(nn.Module):
    spec_channels:int
    segment_size:int
    inter_channels:int
    hidden_channels:int
    filter_channels:int
    n_heads:int
    n_layers:int
    kernel_size:int
    p_dropout:float
    resblock:int
    resblock_kernel_sizes:tuple[int]
    resblock_dilation_sizes:tuple[int]
    upsample_rates:tuple[int]
    upsample_initial_channel:int
    upsample_kernel_sizes:tuple[int]
    spk_embed_dim:int
    gin_channels:int
    sr:int
    def setup(self):
        self.emb_g = nn.Embed(self.spk_embed_dim,self.gin_channels)
        self.enc_p = TextEncoder(
            #256,
            self.inter_channels,
            self.hidden_channels,
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            float(self.p_dropout),
        )
        self.enc_q = PosteriorEncoder(
            #self.spec_channels,
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            16,
            gin_channels=self.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            3,
            gin_channels=self.gin_channels
        )
        self.dec = Generator(
            self.inter_channels,
            self.resblock,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
            gin_channels=self.gin_channels,
            sr=self.sr,
            )

    def __call__(self,
        phone: jnp.ndarray,
        phone_lengths: jnp.ndarray,
        pitch: jnp.ndarray,
        pitchf: jnp.ndarray,
        y: jnp.ndarray,
        y_lengths: jnp.ndarray,
        ds: jnp.ndarray,
        train=True):
        
        g = self.emb_g(ds)[:,jnp.newaxis]
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths,train=train)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g,train=train)
        z_p = self.flow(z, y_mask, g=g,reverse=False,train=train)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size,rng=self.make_rng('rnorms')
        )
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)

        o = self.dec(z_slice, pitchf,g=g,train=train)
        
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(
        self,
        phone: jnp.ndarray,
        phone_lengths: jnp.ndarray,
        pitch: jnp.ndarray,
        nsff0: jnp.ndarray,
        sid: jnp.ndarray
    ):
        g = self.emb_g(sid)[:,jnp.newaxis]
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths,train=False)
        z_p = (m_p + jnp.exp(logs_p) * jax.random.normal(self.make_rng('rnorms'),m_p.shape) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True,train=False)
        o = self.dec(z * x_mask,nsff0,g=g,train=False)
        return o
