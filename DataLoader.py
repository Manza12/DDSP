import numpy as np
import torch

import os
import csv

from scipy.io.wavfile import read
import librosa as li

from Parameters import AUDIO_PATH, RAW_PATH, RAW_DATA_FRECUENCY

# Possible modes : DEBUG, INFO, RUN
PRINT_LEVEL = "INFO"


def charge_data():
    if PRINT_LEVEL == "INFO" or PRINT_LEVEL == "DEBUG":
        print("Charging audio ...")

    audio_tensors = list()
    for file in os.listdir(AUDIO_PATH):
        audio_tensors.append(audio_2_loudness_tensor(file))

    if PRINT_LEVEL == "INFO" or PRINT_LEVEL == "DEBUG":
        print("Audio charged.")

    if PRINT_LEVEL == "INFO" or PRINT_LEVEL == "DEBUG":
        print("Charging raw ...")

    raw_tensors = list()
    for file in os.listdir(RAW_PATH):
        raw_tensors.append(raw_2_tensor(file))

    if PRINT_LEVEL == "INFO" or PRINT_LEVEL == "DEBUG":
        print("Raw charged.")

    return audio_tensors, raw_tensors


def raw_2_tensor(file_name):
    if file_name[-4:-3] == ".":
        if file_name[-4:] == ".csv":
            file_path = os.path.join(RAW_PATH, file_name)
        else:
            print("This file is not a .csv")
    else:
        file_path = os.path.join(RAW_PATH, file_name + ".csv")

    if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO":
        print("Charging", file_name, "...")

    try:
        full_data = list()
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    if PRINT_LEVEL == "DEBUG":
                        print("Tags :", row)
                        print("Data dimensions :", len(row))

                    line_count += 1
                else:
                    data = np.array(row).astype(np.float)
                    full_data.append(data)

            if PRINT_LEVEL == "DEBUG":
                print("Parsing to tensors ...")

            full_data_numpy = np.array(full_data)
            shape = np.shape(full_data_numpy)

            time_freq_data = full_data_numpy[np.ix_(np.arange(shape[0] - 1), [0, 1])]
            time_confidence_data = full_data_numpy[np.ix_(np.arange(shape[0] - 1), [0, 2])]

            time_freq_tensor = torch.from_numpy(time_freq_data)
            time_confidence_tensor = torch.from_numpy(time_confidence_data)

            if PRINT_LEVEL == "DEBUG":
                print("Time-Frecuency tensor : ")
                print(time_freq_tensor)

                print("Time-Confidence tensor : ")
                print(time_confidence_tensor)

            if PRINT_LEVEL == "DEBUG":
                print(file_name, "charged.")

            return time_freq_tensor  # time_confidence_tensor

    except UnboundLocalError:
        print("The file", file_name, "is not a .csv")
    except FileNotFoundError:
        print("There is no file", file_name, "in /Raw folder.")


def audio_2_loudness_tensor(file_name):
    if file_name[-4:-3] == ".":
        if file_name[-4:] == ".wav":
            file_path = os.path.join(AUDIO_PATH, file_name)
        else:
            print("This file is not a .wav")
    else:
        file_path = os.path.join(AUDIO_PATH, file_name + ".csv")

    if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO":
        print("Charging ", file_name, "...")

    try:
        [fs, data] = read(file_path)
        data = data.astype(np.float)

        nb_samples = len(data)
        frame_length = int(fs / RAW_DATA_FRECUENCY)
        nb_frames = int(nb_samples / frame_length)

        time = np.multiply(np.arange(nb_frames), 1 / RAW_DATA_FRECUENCY)
        loudness = li.feature.rms(data, hop_length=frame_length, frame_length=frame_length)[0, 0:nb_frames]

        time_loudness_data = np.stack([time, loudness], axis=0)

        time_loudness_tensor = torch.from_numpy(time_loudness_data)

        if PRINT_LEVEL == "DEBUG":
            print("Audio charged.")

        return time_loudness_tensor

    except UnboundLocalError:
        print("The file", file_name, "is not a .wav")
    except FileNotFoundError:
        print("There is no file", file_name, "in /Raw folder.")
