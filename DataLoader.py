import torch
import os

from scipy.io.wavfile import read
from torch.utils.data import Dataset as ParentDataset

import numpy as np
import pandas as pd

from Parameters import *

#### Debug settings ####
PRINT_LEVEL = "INFO"  # Possible modes : DEBUG, INFO, RUN


class Dataset(ParentDataset):
    """ F0 and Loudness dataset."""

    def __init__(self):
        self.audio_files = sorted(os.listdir(AUDIO_PATH))
        self.raw_files = sorted(os.listdir(RAW_PATH))

    def __len__(self):
        len_audio = len(self.audio_files)
        len_raw = len(self.raw_files)

        if len_audio == len_raw:
            return len_audio * FRAGMENTS_PER_FILE
        else:
            raise Exception("Length of data set does not fit.")

    def __getitem__(self, idx):
        if PRINT_LEVEL == "DEBUG":
            print("Getting item", idx)

        nb_frags = AUDIOFILE_DURATION // FRAGMENT_DURATION
        file_idx = int(idx / nb_frags)
        fragment_idx = idx % nb_frags

        f0_file_name = self.raw_files[file_idx]
        if PRINT_LEVEL == "DEBUG":
            print("F0 file name :", f0_file_name)

        f0_full = torch.tensor(read_f0(f0_file_name)).float()
        f0 = f0_full[int(fragment_idx * 60 * FRAME_SAMPLE_RATE / FRAGMENTS_PER_FILE):
                                   int((fragment_idx+1) * 60 * FRAME_SAMPLE_RATE / FRAGMENTS_PER_FILE)]
        f0 = f0.unsqueeze(-1)

        audio_file_name = self.audio_files[file_idx]
        if PRINT_LEVEL == "DEBUG":
            print("Audio file name :", audio_file_name)

        lo_full = torch.tensor(read_lo(audio_file_name)).float()
        lo = lo_full[int(fragment_idx * 60 * FRAME_SAMPLE_RATE / FRAGMENTS_PER_FILE):
                                   int((fragment_idx + 1) * 60 * FRAME_SAMPLE_RATE / FRAGMENTS_PER_FILE)].float()
        lo = lo.unsqueeze(-1)

        waveform_full = torch.tensor(read_waveform(audio_file_name))
        waveform = waveform_full[int(fragment_idx * 60 * AUDIO_SAMPLE_RATE / FRAGMENTS_PER_FILE):
                                   int((fragment_idx + 1) * 60 * AUDIO_SAMPLE_RATE / FRAGMENTS_PER_FILE)]

        fragment = {'f0': f0, 'lo': lo}

        return fragment, waveform



def read_f0(file_name):
    file_path = os.path.join(RAW_PATH, file_name)
    raw_data = pd.read_csv(file_path, header=0)
    raw_array = raw_data.to_numpy()
    f0 = raw_array[:-1, 1]
    f0 = f0.astype(np.float32)

    return f0


import scipy
import librosa as rosa

def read_lo(file_name):
    file_path = os.path.join(AUDIO_PATH, file_name)
    [fs, waveform] = scipy.io.wavfile.read(file_path)

    assert fs == AUDIO_SAMPLE_RATE

    # int to float
    dtype = waveform.dtype
    if np.issubdtype(dtype, np.integer):
        waveform = waveform.astype(np.float32) / np.iinfo(dtype).max

    lo = rosa.feature.rms(waveform, hop_length=FRAME_LENGTH, frame_length=FRAME_LENGTH)
    lo = lo.flatten()
    lo = lo.astype(np.float32)

    lo = np.log(lo + np.finfo(np.float32).eps)

    return lo

def read_waveform(file_name):
    file_path = os.path.join(AUDIO_PATH, file_name)
    [fs, waveform] = scipy.io.wavfile.read(file_path)

    assert fs == AUDIO_SAMPLE_RATE

    # int to float
    dtype = waveform.dtype
    if np.issubdtype(dtype, np.integer):
        waveform = waveform.astype(np.float32) / np.iinfo(dtype).max

    return waveform
