import numpy as np
import torch

import os
import csv

# Possible modes : DEBUG, INFO, RUN
PRINT_LEVEL = "INFO"


def raw_2_tensor(file_name):
    if PRINT_LEVEL == "INFO" or PRINT_LEVEL == "DEBUG":
        print("Charging data ...")

    file_path = os.path.join("Raw", file_name + ".csv")

    if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO":
        print("Charging .csv ...")

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
        time_loudness_data = full_data_numpy[np.ix_(np.arange(shape[0] - 1), [0, 2])]

        time_freq_tensor = torch.from_numpy(time_freq_data)
        time_loudness_tensor = torch.from_numpy(time_loudness_data)

        if PRINT_LEVEL == "DEBUG":
            print("Time-Frecuency tensor : ")
            print(time_freq_tensor)

            print("Time-Loudness tensor : ")
            print(time_loudness_tensor)

        if PRINT_LEVEL == "INFO" or PRINT_LEVEL == "DEBUG":
            print("Données chargées.")
