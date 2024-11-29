
import numpy as np
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn

class SineGen(nn.Module):
    
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)

    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)

    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """
    samp_rate:int
    harmonic_num:int=0
    sine_amp:float=0.1
    noise_std:float=0.003
    voiced_threshold:int=0
    flag_for_pulse:bool=False
    def setup(self):
        self.dim = self.harmonic_num + 1
        self.sampling_rate = self.samp_rate

    def _f02uv(self, f0):
        # generate uv signal
        uv = jnp.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def __call__(self, f0,upp,rng):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        f0 = f0[:, jnp.newaxis].transpose(0,2,1)
        f0_buf = jnp.zeros((f0.shape[0], f0.shape[1], self.dim))
        # fundamental component
        f0_buf = f0_buf.at[:, :, 0].set(f0[:, :, 0])
        for idx in range(self.harmonic_num):
            f0_buf = f0_buf.at[:, :, idx + 1].set(f0_buf[:, :, 0] * (idx + 2))  
            # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
        rad_values = (
            f0_buf / self.sampling_rate
        ) % 1  ###%1意味着n_har的乘积无法后处理优化
        rand_ini = jax.random.uniform(rng,(f0_buf.shape[0], f0_buf.shape[2]))
        rand_ini = rand_ini.at[:, 0].set(0)
        rad_values = rad_values.at[:, 0, :].set(rad_values[:, 0, :] + rand_ini)
        tmp_over_one = jnp.cumsum(
            rad_values, 1
        )  # % 1  #####%1意味着后面的cumsum无法再优化
        tmp_over_one *= upp
        tmp_over_one = jax.image.resize(tmp_over_one, shape=(tmp_over_one.shape[0], tmp_over_one.shape[1]* upp, tmp_over_one.shape[2] ), method='linear')
        rad_values = jax.image.resize(rad_values, shape=(rad_values.shape[0], rad_values.shape[1]* upp, rad_values.shape[2] ), method='nearest')
        tmp_over_one %= 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = jnp.zeros_like(rad_values)
        cumsum_shift = cumsum_shift.at[:, 1:, :].set(tmp_over_one_idx * -1.0)
        sine_waves = jnp.sin(
            jnp.cumsum(rad_values + cumsum_shift, axis=1) * 2 * jnp.pi
        )
        sine_waves = sine_waves * self.sine_amp
        uv = self._f02uv(f0)
        uv = jax.image.resize(uv, shape=(uv.shape[0], uv.shape[1]* upp, uv.shape[2] ), method='nearest')
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * jax.random.normal(rng,sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise




class SourceModuleHnNSF(nn.Module):
    sampling_rate:int=40000
    harmonic_num:int = 0
    sine_amp:float=0.1
    add_noise_std:float=0.003
    voiced_threshod:int=0
    def setup(self):
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            self.sampling_rate, self.harmonic_num, self.sine_amp, self.add_noise_std, self.voiced_threshod
        )
        # to merge source harmonics into a single excitation
        self.l_linear = nn.Dense(1)

    def __call__(self, x,upp,rng):
        """
        Sine_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        """
        # source for harmonic branch
        sine_wavs,uv,_ = jax.lax.stop_gradient(self.l_sin_gen(x,upp,rng))
        sine_merge = nn.tanh(self.l_linear(sine_wavs))
        return sine_merge
