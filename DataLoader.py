import torch
import os

from scipy.io.wavfile import read
from torch.utils.data import Dataset as ParentDataset

import librosa as li
import numpy as np
import pandas as pd

from Parameters import AUDIO_PATH, RAW_PATH, FRAME_SAMPLE_RATE, FRAGMENTS_PER_FILE, AUDIO_SAMPLE_RATE, STFT_SIZE

# Possible modes : DEBUG, INFO, RUN
PRINT_LEVEL = "INFO"


class Dataset(ParentDataset):
    """ F0 and Loudness dataset."""

    def __init__(self):
        self.audio_files = os.listdir(AUDIO_PATH)
        self.raw_files = os.listdir(RAW_PATH)

    def __len__(self):
        len_audio = len(self.audio_files)
        len_raw = len(self.raw_files)

        if len_audio == len_raw:
            return len_audio * FRAGMENTS_PER_FILE
        else:
            raise Exception("Length of data set does not fit.")

    def __getitem__(self, idx):
        file_idx = int(idx / 30)
        fragment_idx = idx % 30

        f0_file_name = self.raw_files[file_idx]
        f0_full = f0_2_tensor(f0_file_name)
        f0 = f0_full[int(fragment_idx * 60 * FRAME_SAMPLE_RATE / FRAGMENTS_PER_FILE):
                                   int((fragment_idx+1) * 60 * FRAME_SAMPLE_RATE / FRAGMENTS_PER_FILE)]
        f0 = f0.reshape((f0.shape[0], 1))

        audio_file_name = self.audio_files[file_idx]

        [lo_full, waveform_full] = audio_2_loudness_tensor(audio_file_name)
        lo = lo_full[0, int(fragment_idx * 60 * FRAME_SAMPLE_RATE / FRAGMENTS_PER_FILE):
                                   int((fragment_idx + 1) * 60 * FRAME_SAMPLE_RATE / FRAGMENTS_PER_FILE)]
        waveform = waveform_full[int(fragment_idx * 60 * AUDIO_SAMPLE_RATE / FRAGMENTS_PER_FILE):
                                   int((fragment_idx + 1) * 60 * AUDIO_SAMPLE_RATE / FRAGMENTS_PER_FILE)]

        lo = lo.reshape((lo.shape[0], 1))

        fragment = {'f0': f0, 'lo': lo}

        return fragment, waveform


def f0_2_tensor(file_name):
    file_path = os.path.join(RAW_PATH, file_name)
    raw_data = pd.read_csv(file_path, header=0)
    raw_array = raw_data.to_numpy()
    frecuency_data = raw_array[:-1, 1]
    frecuency_tensor = torch.from_numpy(frecuency_data)

    return frecuency_tensor


def audio_2_loudness_tensor(file_name):
    file_path = os.path.join(AUDIO_PATH, file_name)
    [fs, data] = read(file_path)
    data = data.astype(np.float)
    frame_length = int(fs / FRAME_SAMPLE_RATE)
    loudness_array = li.feature.rms(data, hop_length=frame_length, frame_length=frame_length)
    loudness_tensor = torch.from_numpy(loudness_array)

    return loudness_tensor, data
