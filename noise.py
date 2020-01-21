import numpy as np
import torch.nn.functional as func
import scipy.io.wavfile as wav

from parameters import *


def synthetize_noise(hh, device):
    # Drop last frame
    hh = hh[:, :-1, :]

    if NOISE_AMPLITUDE:
        h0 = hh[:, :, 0].unsqueeze(-1)
        hh = hh[:, :, 1:]
    else:
        h0 = torch.ones(hh.shape[0], hh.shape[1], 1, device=device)

    assert len(h0.size()) == 3
    assert len(hh.size()) == 3
    assert h0.size()[0] == hh.size()[0]  # batch dim
    assert h0.size()[1] == hh.size()[1]  # frame dim

    nb_batchs, nb_frames, _ = hh.size()

    # Transform filter coeffs back to time
    z = torch.zeros(hh.shape, device=device)
    hh = torch.stack((hh, z), dim=-1)
    ir = torch.irfft(hh, 1, onesided=True)

    # To linear phase
    ir = torch.roll(ir, ir.size()[-1] // 2, -1)

    # Apply hann window
    if HAMMING_NOISE:
        win = torch.hann_window(ir.shape[-1], device=device)
        win = win.reshape(1, 1, -1)
        ir = ir * win

    # Pad filter in time so it has the same length as a frame
    padding = (FRAME_LENGTH - ir.size()[-1]) + FRAME_LENGTH - 1
    padding_left = padding // 2
    padding_right = padding - padding_left
    ir = func.pad(ir, [padding_left, padding_right])

    # Init noise in [-1, +1] * NOISE_LEVEL
    noise = torch.rand(nb_frames, FRAME_LENGTH, device=device) * 2 - 1
    noise *= NOISE_LEVEL

    noise = torch.conv1d(ir, noise.unsqueeze(1), groups=nb_frames)
    noise = noise * h0

    noise = noise.reshape(nb_batchs, -1)

    return noise


def filter_noise(noise_time, hh, write=False, nom="filtered_noise",
                 device=DEVICE):
    # Décomposition
    if NOISE_AMPLITUDE:
        filter_lo = hh[:, :, 0]
        hh = hh[:, :, 1:]
        hh_sum = torch.sum(hh, dim=2)
        hh_sum[hh_sum == 0.] = 1.
        hh_norm = hh / hh_sum.unsqueeze(-1)
        hh = hh_norm * filter_lo.unsqueeze(-1)

    # Noise part
    new_shape = (noise_time.shape[0] // FRAME_LENGTH, FRAME_LENGTH)
    noise_time_splited = noise_time.reshape(new_shape)
    noise_time_splited = noise_time_splited.unsqueeze(0)
    noise_freq = torch.rfft(noise_time_splited, 1)

    # Filter part
    hh_imaginary = torch.zeros(hh.shape, device=device)
    hh_complex = torch.stack((hh, hh_imaginary), dim=-1)

    # Hamming smoothing
    filter_time = torch.irfft(hh_complex, 1, onesided=True)
    if HAMMING_NOISE:
        hann_window = torch.hann_window(filter_time.shape[-1], device=device)
        hann_norm = sum(hann_window)
        hann_window = hann_window / hann_norm
        hann_window = hann_window.unsqueeze(0).unsqueeze(0)
        filter_time = filter_time * hann_window
        filter_time = torch.roll(filter_time, filter_time.shape[0] // 2 + 1,
                                 dims=-1)

    pad = (noise_time_splited.shape[-1] - filter_time.shape[-1]) // 2
    filter_time = func.pad(filter_time, [pad, pad + 1])
    hh_complex = torch.rfft(filter_time, 1)

    # Filtered noise
    filtered_noise_freq = complex_mult_torch(noise_freq, hh_complex)
    filtered_noise_time = torch.irfft(filtered_noise_freq, 1)
    filtered_noise_time = filtered_noise_time[:, :, 0:FRAME_LENGTH]

    noises_list = torch.split(filtered_noise_time, 1, dim=1)
    noise = torch.cat(noises_list[0:-1], dim=-1)
    noise = torch.squeeze(noise, dim=1)

    if write:
        wav.write(os.path.join("Outputs", "Original_Noise" + ".wav"),
                  AUDIO_SAMPLE_RATE, noise_time.cpu().detach().numpy())
        wav.write(os.path.join("Outputs", nom + ".wav"),
                  AUDIO_SAMPLE_RATE, noise[0, :].cpu().detach().numpy())

    torch.cuda.empty_cache()

    return noise


def create_white_noise(samples, write=False, nom="white_noise", device="cpu"):
    noise_time = torch.tensor(np.float32(np.random.uniform(-1, 1, samples)),
                              device=device)
    noise_time = noise_time * NOISE_LEVEL

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
