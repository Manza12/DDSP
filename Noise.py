import numpy as np
import torch
import torch.nn.functional as func
import scipy.io.wavfile as wav
import os

from Parameters import AUDIO_SAMPLE_RATE


def create_white_noise(samples, write=False, nom="white_noise"):
    noise_time = torch.tensor(np.float32(np.random.uniform(-1, 1, samples)))
    if write:
        wav.write(os.path.join("Outputs", nom + ".wav"), AUDIO_SAMPLE_RATE, noise_time.numpy())

    return noise_time


def filter_noise(noise_time, filter_freq, write=False, nom="filtered_noise"):
    filter_freq_complex = torch.stack((filter_freq, torch.zeros(filter_freq.shape[0])), dim=1)
    filter_time = torch.irfft(filter_freq_complex, 1, onesided=True)
    hann_window = torch.hann_window(filter_time.shape[0])
    filter_time = filter_time * hann_window
    filter_time = torch.roll(filter_time, filter_time.shape[0] // 2 + 1)
    pad = (noise_time.shape[0]-filter_time.shape[0]) // 2
    filter_time = func.pad(filter_time, [pad, pad + 1])
    noise_freq = torch.rfft(noise_time, 1)
    filter_freq = torch.rfft(filter_time, 1)
    filtered_noise_freq = complex_mult_torch(noise_freq, filter_freq)
    filtered_noise_time = torch.irfft(filtered_noise_freq, 1, signal_sizes=noise_time.shape)

    if write:
        wav.write(os.path.join("Outputs", nom + ".wav"), AUDIO_SAMPLE_RATE, filtered_noise_time.numpy())

    return filtered_noise_time


def complex_mult_torch(z, w):
    assert z.shape[-1] == 2 and w.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (z[..., 0] * w[..., 0] - z[..., 1] * w[..., 1],
         z[..., 0] * w[..., 1] + z[..., 1] * w[..., 0]),
        dim=-1)


if __name__ == "__main__":
    noise_sound = create_white_noise(16000, write=True)
    noise = create_white_noise(160)

    filter_transfer_ampl = torch.zeros(65)
    mu = 30
    filter_transfer_ampl[mu] = 1
    filter_transfer_ampl[mu - 1] = 0.5
    filter_transfer_ampl[mu + 1] = 0.5


    filtered_noise = filter_noise(noise, filter_transfer_ampl)

    sound = np.zeros(16000)
    for i in range(100):
        sound[160 * i: 160 * (i+1)] = filtered_noise.numpy()

    wav.write(os.path.join("Outputs", "filtered_noise" + ".wav"), AUDIO_SAMPLE_RATE, sound)
