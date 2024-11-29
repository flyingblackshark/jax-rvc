from models import SynthesizerTrn
import jax.numpy as jnp 
import jax
import flax
from librosa.filters import mel as librosa_mel_fn
import audax
from jax_fcpe.utils import load_model
import librosa
import numpy as np
from transformers import FlaxAutoModel
import soundfile as sf
model = SynthesizerTrn(spec_channels=1025,
                      segment_size=32,
                      inter_channels=192,
                      hidden_channels=192,
                      filter_channels=768,
                      n_heads=2,
                      n_layers=6,
                      kernel_size=3,
                      p_dropout=0,
                      resblock=1,
                      resblock_kernel_sizes=[3,7,11],
                      resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                      upsample_rates=[10, 10, 2, 2],
                      upsample_initial_channel=512,
                      upsample_kernel_sizes=[16,16,4,4],
                      spk_embed_dim=109,
                      gin_channels=256,
                      sr=40000
                      )
rng = jax.random.key(0)
params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
# example_inputs = {
#     "phone":jnp.ones((1,400,256)),
#     "pitch":jnp.ones((1,400),dtype=jnp.int32),
#     "pitchf":jnp.ones((1,400)),
#     "y":jnp.ones((1,400,1025)),
#     "phone_lengths":jnp.ones((1),dtype=jnp.int32),
#     "y_lengths":jnp.ones((1),dtype=jnp.int32),
#     "ds":jnp.ones((1),dtype=jnp.int32),
# }

# params = model.init(init_rngs, **example_inputs,train=False)
# flatten_params = flax.traverse_util.flatten_dict(params,sep='.')
from convert import load_params
converted_params = load_params("Alto.pth")


def get_f0(wav):
    WIN_SIZE = 1024
    HOP_SIZE = 160
    N_FFT = 1024
    NUM_MELS = 128
    f0_min = 80.
    f0_max = 880.
    mel_basis = librosa_mel_fn(sr=16000, n_fft=N_FFT, n_mels=NUM_MELS, fmin=0, fmax=8000)
    mel_basis = jnp.asarray(mel_basis,dtype=jnp.float32)

    model,params = load_model()
    wav = jnp.asarray(wav)
    window = jnp.hanning(WIN_SIZE)
    pad_size = (WIN_SIZE-HOP_SIZE)//2
    wav = jnp.pad(wav, ((0,0),(pad_size, pad_size)),mode="reflect")
    spec = audax.core.stft.stft(wav,N_FFT,HOP_SIZE,WIN_SIZE,window,onesided=True,center=False)
    spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
    spec = spec.transpose(0,2,1)
    mel = jnp.matmul(mel_basis, spec)
    mel = jnp.log(jnp.clip(mel, min=1e-5) * 1)
    mel = mel.transpose(0,2,1)

    def model_predict(mel):
        f0 = model.apply(params,mel,threshold=0.03,method=model.infer)
        uv = (f0 < f0_min).astype(jnp.float32)
        f0 = f0 * (1 - uv)
        return f0
    return model_predict(mel).squeeze(-1)
def f0_to_coarse(f0):
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)
    return f0_coarse

wav, sr = librosa.load("test.wav", sr=16000,mono=True)
#wav = np.pad(wav, (16000, 16000), mode="reflect")

pit = get_f0(np.expand_dims(wav,0)).squeeze(0)
#pit = np.ones((1000)) * 180
hubert_model = FlaxAutoModel.from_pretrained("./hubert",trust_remote_code=True)
ppg = hubert_model(np.expand_dims(wav,0)).last_hidden_state

pit = np.expand_dims(pit,0)
ppg = np.repeat(ppg,repeats=2,axis=1)
pit = pit[:,:ppg.shape[1]]
# example_inputs = {
#     "phone":jnp.ones((1,400,256)),
#     "pitch":jnp.ones((1,400),dtype=jnp.int32),
#     "nsff0":jnp.ones((1,400)),
#     "phone_lengths":jnp.ones((1),dtype=jnp.int32),
#     "sid":jnp.ones((1),dtype=jnp.int32)
# }

example_inputs = {
    "phone":ppg,
    "pitch":f0_to_coarse(pit),
    "nsff0":pit,
    "phone_lengths": jnp.array((ppg.shape[1],)),
    "sid":jnp.zeros((1),dtype=jnp.int32)
}

test_output = model.apply({"params":converted_params},**example_inputs,method=model.infer,rngs=init_rngs)
sf.write("output.wav", jnp.squeeze(test_output,axis=(0,2)), samplerate=40000)
