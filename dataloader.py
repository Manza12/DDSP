import scipy
import numpy as np
import pandas as pd
import scipy.signal as sg

from scipy.io.wavfile import read
from torch.utils.data import Dataset as ParentDataset
from parameters import *


class Dataset(ParentDataset):
    """ F0 and Loudness dataset."""

    def __init__(self):
        if COMPUTE_CACHE or not os.path.isdir(FRAGMENT_CACHE_PATH):
            compute_cache()

        self.nb_frags = len(os.listdir(FRAGMENT_CACHE_PATH))

    def __len__(self):
        return self.nb_frags

    def __getitem__(self, idx):
        frag_path = os.path.join(FRAGMENT_CACHE_PATH,
                                 FRAGMENT_CACHE_PATTERN.format(idx))
        frag = torch.load(frag_path)
        return frag


def compute_cache():
    try:
        os.mkdir(FRAGMENT_CACHE_PATH)
    except OSError:
        pass

    f0_filenames = sorted(os.listdir(F0_PATH))
    audio_filenames = sorted(os.listdir(AUDIO_PATH))

    item_i = 0
    for audio_filename, f0_filename in zip(audio_filenames, f0_filenames):
        f0_full = read_f0(f0_filename)
        lo_full = read_lo(audio_filename)

        lo_full = smooth_scale_loudness(lo_full)

        waveform_full = read_waveform(audio_filename)

        for frag_i in range(FRAGMENTS_PER_FILE):
            inputs, waveforms = compute_fragment_cache(f0_full, lo_full,
                                                       waveform_full, frag_i)
            frag_path = os.path.join(FRAGMENT_CACHE_PATH,
                                     FRAGMENT_CACHE_PATTERN.format(item_i))
            torch.save((inputs, waveforms), frag_path)
            item_i += 1


def compute_fragment_cache(f0_full, lo_full, waveform_full, frag_i):
    f0_lo_stride = FRAME_SAMPLE_RATE * FRAGMENT_DURATION
    waveform_stride = FRAGMENT_DURATION * AUDIO_SAMPLE_RATE

    f0_lo_start_i = frag_i * f0_lo_stride
    f0_lo_end_i = (frag_i + 1) * f0_lo_stride

    f0 = f0_full[f0_lo_start_i:f0_lo_end_i]
    f0 = f0.reshape((f0.shape[0], 1))
    f0 = torch.tensor(f0)

    lo = lo_full[f0_lo_start_i:f0_lo_end_i]
    lo = lo.reshape((lo.shape[0], 1))
    inputs = {"f0": f0, "lo": lo}

    waveform_start_i = frag_i * waveform_stride
    waveform_end_i = (frag_i + 1) * waveform_stride

    waveform_end_i -= FRAME_LENGTH
    waveform = waveform_full[waveform_start_i:waveform_end_i]
    waveform = torch.tensor(waveform)

    return inputs, waveform


def read_f0(file_name):
    file_path = os.path.join(F0_PATH, file_name)
    raw_data = pd.read_csv(file_path, header=0)
    raw_array = raw_data.to_numpy()
    f0 = raw_array[:-1, 1]
    f0 = f0.astype(np.float32)

    return f0


def read_lo(file_name):
    file_path = os.path.join(AUDIO_PATH, file_name)
    [fs, waveform] = scipy.io.wavfile.read(file_path)

    assert fs == AUDIO_SAMPLE_RATE

    waveform = int_2_float(waveform)

    waveform = torch.tensor(waveform)

    stft = torch.stft(waveform, FRAME_LENGTH * 4, hop_length=FRAME_LENGTH,
                      center=True, pad_mode='reflect',
                      normalized=STFT_NORMALIZED, onesided=True)
    stft = torch.sum(stft ** 2, dim=-1)
    lo = torch.sum(stft, dim=0)
    lo = torch.log(lo + np.finfo(np.float32).eps)

    return lo


def smooth_scale_loudness(loudness, filter_loudness=SMOOTH_LOUDNESS):
    audio_filenames = sorted(os.listdir(AUDIO_PATH))
    mean_lo = get_mean_lo(audio_filenames)
    lo = loudness - mean_lo

    if filter_loudness:
        n = 20
        cutoff_fq = 0.3 * 2
        [b, a] = sg.butter(n, cutoff_fq)
        lo_fliped = sg.filtfilt(b, a, lo)
        lo = np.flip(np.flip(lo_fliped, axis=0).copy()).copy()
        lo = torch.from_numpy(lo).float()

    return lo


def get_mean_lo(audio_filenames):
    lo_all = [read_lo(f) for f in audio_filenames]
    lo_all = np.concatenate(lo_all, axis=0)
    return np.mean(lo_all)


def read_waveform(file_name):
    file_path = os.path.join(AUDIO_PATH, file_name)
    [fs, waveform] = scipy.io.wavfile.read(file_path)

    assert fs == AUDIO_SAMPLE_RATE

    waveform = int_2_float(waveform)

    return waveform


def int_2_float(int_like):
    dtype = int_like.dtype
    if np.issubdtype(dtype, np.integer):
        float_like = int_like.astype(np.float32) / np.iinfo(dtype).max
        return float_like
    else:
        return int_like


if __name__ == "__main__":
    compute_cache()
