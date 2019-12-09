import numpy as np
import scipy.io.wavfile as wav

from Parameters import AUDIO_SAMPLING_RATE, SAMPLE_DURATION, OUTPUT_DIM
from tqdm import tqdm


def synthese_torch(outputs, f0):
    num_harmonics = OUTPUT_DIM - 1
    n_sample = AUDIO_SAMPLING_RATE * SAMPLE_DURATION
    lo = outputs[:, 0]
    alpha = outputs
    oldx = np.linspace(0, 1, f0.shape[0])
    newx = np.linspace(0, 1, n_sample)
    f0 = np.interp(newx, oldx, f0[:,0]) / AUDIO_SAMPLING_RATE
    lo = np.interp(newx, oldx, lo.numpy())
    alpha = np.stack([np.interp(newx, oldx, alpha[:, i]) for i in range(1, num_harmonics+1)])
    phi = np.zeros(f0.shape)
    aa = np.zeros((num_harmonics, f0.shape[-1]))
    k = np.arange(1, num_harmonics + 1).reshape(1, -1)
    for i in tqdm(np.arange(1, phi.shape[-1])):
        phi[i] = 2 * np.pi * f0[i] + phi[i - 1]
        aa[:, i] = (f0[i] * k) < .5
    y = lo * np.sum(aa * alpha * np.sin(k.reshape(-1, 1) * phi.reshape(1, -1)), 0)
    wav.write("Test_antoine.wav", AUDIO_SAMPLING_RATE, y)
    return y


def synthese(outputs, f0):
    num_harmonics = OUTPUT_DIM - 1
    n_sample = AUDIO_SAMPLING_RATE * SAMPLE_DURATION
    lo = outputs[:, 0]
    alpha = outputs
    oldx = np.linspace(0, 1, f0.shape[0])
    newx = np.linspace(0, 1, n_sample)
    f0 = np.interp(newx, oldx, f0[:,0]) / AUDIO_SAMPLING_RATE
    lo = np.interp(newx, oldx, lo.detach().numpy())
    alpha = np.stack([np.interp(newx, oldx, alpha[:, i].detach().numpy()) for i in range(1, num_harmonics+1)])
    phi = np.zeros(f0.shape)
    aa = np.zeros((num_harmonics, f0.shape[-1]))
    k = np.arange(1, num_harmonics + 1).reshape(1, -1)
    for i in tqdm(np.arange(1, phi.shape[-1])):
        phi[i] = 2 * np.pi * f0[i] + phi[i - 1]
        aa[:, i] = (f0[i] * k) < .5
    y = lo * np.sum(aa * alpha * np.sin(k.reshape(-1, 1) * phi.reshape(1, -1)), 0)

    return y


def synthese_write(outputs, f0):
    num_harmonics = OUTPUT_DIM - 1
    n_sample = AUDIO_SAMPLING_RATE * SAMPLE_DURATION
    stride = AUDIO_SAMPLING_RATE // 100
    x = np.linspace(0, SAMPLE_DURATION, n_sample // stride)
    xp = np.linspace(0, SAMPLE_DURATION, n_sample // stride)
    fp = f0
    f0 = np.interp(x, xp, fp)
    lo = np.ones(n_sample//stride)
    alpha = outputs
    oldx = np.linspace(0, 1, len(f0))
    newx = np.linspace(0, 1, n_sample)
    f0 = np.interp(newx, oldx, f0) / AUDIO_SAMPLING_RATE
    lo = np.interp(newx, oldx, lo)
    alpha = np.stack([np.interp(newx, oldx, alpha[:, i]) for i in range(1, num_harmonics+1)])
    phi = np.zeros(f0.shape)
    aa = np.zeros((num_harmonics, f0.shape[-1]))
    k = np.arange(1, num_harmonics + 1).reshape(1, -1)
    for i in tqdm(np.arange(1, phi.shape[-1])):
        phi[i] = 2 * np.pi * f0[i] + phi[i - 1]
        aa[:, i] = (f0[i] * k) < .5
    y = lo * np.sum(aa * alpha * np.sin(k.reshape(-1, 1) * phi.reshape(1, -1)), 0)
    wav.write("Test_antoine.wav", AUDIO_SAMPLING_RATE, y)