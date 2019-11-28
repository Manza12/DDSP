import numpy as np
import torch

import os
import csv

from scipy.io.wavfile import read
import librosa.feature

# Possible modes : DEBUG, INFO, RUN
PRINT_LEVEL = "INFO"


def raw_2_tensor(file_name):
    if PRINT_LEVEL == "INFO" or PRINT_LEVEL == "DEBUG":
        print("Charging data ...")

    if file_name[-4:-3] == ".":
        if file_name[-4:] == ".csv":
            file_path = os.path.join("Raw", file_name)
        else:
            print("This file is not a .csv")
    else:
        file_path = os.path.join("Raw", file_name + ".csv")

    if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO":
        print("Charging .csv ...")

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

            if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO":
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

            if PRINT_LEVEL == "INFO" or PRINT_LEVEL == "DEBUG":
                print("Données chargées.")

            return time_freq_tensor, time_confidence_tensor

    except UnboundLocalError:
        print("The file", file_name, "is not a .csv")
    except FileNotFoundError:
        print("There is no file", file_name, "in /Raw folder.")


def audio_2_loudness_tensor(file_name):
    if PRINT_LEVEL == "INFO" or PRINT_LEVEL == "DEBUG":
        print("Calculating loudness ...")

    if file_name[-4:-3] == ".":
        if file_name[-4:] == ".wav":
            file_path = os.path.join("Audio", file_name)
        else:
            print("This file is not a .wav")
    else:
        file_path = os.path.join("Audio", file_name + ".csv")

    if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO":
        print("Charging .wav ...")

    try:
        [fs, data] = read(file_path)
        data = data.astype(np.float)

        nb_samples = len(data)
        frame_step = int(fs / 100)
        nb_frames = int(nb_samples / frame_step)

        time_loudness_data = np.zeros((nb_frames, 2))

        for fr in range(nb_frames):
            sample = fr*frame_step
            data_frame = data[sample:sample+frame_step]

            rms = librosa.feature.rms(data_frame)

            time_loudness_data[fr, 0] = sample / fs
            time_loudness_data[fr, 1] = rms

        time_loudness_tensor = torch.from_numpy(time_loudness_data)

        return time_loudness_tensor

    except UnboundLocalError:
        print("The file", file_name, "is not a .wav")
    except FileNotFoundError:
        print("There is no file", file_name, "in /Raw folder.")
