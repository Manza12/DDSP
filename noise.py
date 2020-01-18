import numpy as np
import torch
import torch.nn.functional as func
import scipy.io.wavfile as wav
import os

from parameters import *


def filter_noise(noise_time, filter_freq, write=False, nom="filtered_noise",
                 device=DEVICE):
    # Décomposition
    if NOISE_AMPLITUDE:
        filter_lo = filter_freq[:, :, 0]
        filter_freq = filter_freq[:, :, 1:]
        hh_sum = torch.sum(filter_freq, dim=2)
        hh_sum[hh_sum == 0.] = 1.
        hh_norm = filter_freq / hh_sum.unsqueeze(-1)
        filter_freq = hh_norm * filter_lo.unsqueeze(-1)
    else:
        filter_freq = filter_freq / NUMBER_NOISE_BANDS

    # Noise part
    new_shape = (noise_time.shape[0] // FRAME_LENGTH, FRAME_LENGTH)
    noise_time_splited = noise_time.reshape(new_shape)
    noise_time_splited = noise_time_splited.unsqueeze(0)
    noise_freq = torch.rfft(noise_time_splited, 1)

    # Filter part
    filter_freq_complex = torch.stack((filter_freq,
                                       torch.zeros(filter_freq.shape,
                                                   device=device)), dim=-1)
    filter_time = torch.irfft(filter_freq_complex, 1, onesided=True)
    filter_time = filter_time.roll(filter_time.shape[-1] // 2)

    if HAMMING_NOISE:
        hann_window = torch.hann_window(filter_time.shape[-1], device=device)
        hann_window = torch.unsqueeze(hann_window, 0)
        hann_window = torch.unsqueeze(hann_window, 0)
        filter_time = filter_time * hann_window

    pad = (noise_time_splited.shape[-1] - filter_time.shape[-1]) // 2
    filter_time = func.pad(filter_time, [pad, pad + 1])
    filter_freq = torch.rfft(filter_time, 1)

    # Filtered noise
    filtered_noise_freq = complex_mult_torch(noise_freq, filter_freq)
    filtered_noise_time = torch.irfft(filtered_noise_freq, 1)
    filtered_noise_time = filtered_noise_time[:, :, 0:FRAME_LENGTH]

    noise = filtered_noise_time.reshape([filtered_noise_time.shape[0], -1])
    noise = noise[:, :-FRAME_LENGTH]

    if write:
        wav.write(os.path.join("Outputs", "w_noise" + ".wav"),
                  AUDIO_SAMPLE_RATE, noise_time.cpu().detach().numpy())
        wav.write(os.path.join("Outputs", nom + ".wav"), AUDIO_SAMPLE_RATE,
                  noise[0, :].cpu().detach().numpy())

    torch.cuda.empty_cache()

    return noise


def create_white_noise(samples, write=False, nom="white_noise", device="cpu"):
    noise_time = torch.tensor(np.float32(np.random.uniform(-1, 1, samples)),
                              device=device)
    if write:
        wav.write(os.path.join("Outputs", nom + ".wav"), AUDIO_SAMPLE_RATE,
                  noise_time.numpy())

    return noise_time


def complex_mult_torch(z, w):
    assert z.shape[-1] == 2 and w.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (z[..., 0] * w[..., 0] - z[..., 1] * w[..., 1],
         z[..., 0] * w[..., 1] + z[..., 1] * w[..., 0]),
        dim=-1)


if __name__ == "__main__":
    noise_sound = create_white_noise(16000, write=True)
    duration = 1  # in seconds
    length = duration * AUDIO_SAMPLE_RATE
    NOISE = create_white_noise(length)

    filter_transfer_ampl = torch.zeros(65)
    filter_transfer_ampl[20:30] = torch.linspace(1, 0, 10)
    filter_transfer_ampl[10:20] = torch.linspace(0, 1, 10)
    filter_transfer_ampl = \
        filter_transfer_ampl.expand(1, length // FRAME_LENGTH, 65)

    filtered_noise = filter_noise(NOISE, filter_transfer_ampl, device="cpu")

    sound = filtered_noise[0].numpy()

    wav.write(os.path.join("Outputs", "filtered_noise" + ".wav"),
              AUDIO_SAMPLE_RATE, sound)