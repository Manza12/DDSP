import torch
import torch.nn as nn
import torch.nn.functional as func

from Parameters import SAMPLE_DURATION, RAW_DATA_FRECUENCY, LINEAR_OUT_DIM, OUTPUT_DIM


SAMPLE_SIZE = SAMPLE_DURATION * RAW_DATA_FRECUENCY

class DDSPNet(nn.Module):
    def __init__(self):
        super(DDSPNet, self).__init__()

        """ Frecuency MLP """
        self.frecuency_linear_1 = nn.Linear(SAMPLE_SIZE, LINEAR_OUT_DIM)
        self.frecuency_linear_2 = nn.Linear(LINEAR_OUT_DIM, LINEAR_OUT_DIM)
        self.frecuency_linear_3 = nn.Linear(LINEAR_OUT_DIM, LINEAR_OUT_DIM)

        self.frecuency_batch_norm_1 = nn.BatchNorm1d(LINEAR_OUT_DIM)
        self.frecuency_batch_norm_2 = nn.BatchNorm1d(LINEAR_OUT_DIM)
        self.frecuency_batch_norm_3 = nn.BatchNorm1d(LINEAR_OUT_DIM)

        """ Loudness MLP """
        self.loudness_linear_1 = nn.Linear(SAMPLE_SIZE, LINEAR_OUT_DIM)
        self.loudness_linear_2 = nn.Linear(LINEAR_OUT_DIM, LINEAR_OUT_DIM)
        self.loudness_linear_3 = nn.Linear(LINEAR_OUT_DIM, LINEAR_OUT_DIM)

        self.loudness_batch_norm_1 = nn.BatchNorm1d(LINEAR_OUT_DIM)
        self.loudness_batch_norm_2 = nn.BatchNorm1d(LINEAR_OUT_DIM)
        self.loudness_batch_norm_3 = nn.BatchNorm1d(LINEAR_OUT_DIM)

        """ Concatenate part """
        self.gru = nn.GRU(LINEAR_OUT_DIM, LINEAR_OUT_DIM)

        self.linear_1 = nn.Linear(LINEAR_OUT_DIM, OUTPUT_DIM)
        self.linear_2 = nn.Linear(OUTPUT_DIM, OUTPUT_DIM)
        self.linear_3 = nn.Linear(OUTPUT_DIM, OUTPUT_DIM)

        self.batch_norm_1 = nn.BatchNorm1d(OUTPUT_DIM)
        self.batch_norm_2 = nn.BatchNorm1d(OUTPUT_DIM)
        self.batch_norm_3 = nn.BatchNorm1d(OUTPUT_DIM)

    def forward(self, sample):
        frecuency = sample["frecuency"]
        loudness = sample["loudness"]

        """ Frecuency MLP """
        frecuency = func.relu(self.frecuency_batch_norm_1(self.frecuency_linear_1(frecuency)))
        frecuency = func.relu(self.frecuency_batch_norm_2(self.frecuency_linear_2(frecuency)))
        frecuency = func.relu(self.frecuency_batch_norm_3(self.frecuency_linear_3(frecuency)))

        """ Loudness MLP """
        loudness = func.relu(self.loudness_batch_norm_1(self.loudness_linear_1(loudness)))
        loudness = func.relu(self.loudness_batch_norm_2(self.loudness_linear_2(loudness)))
        loudness = func.relu(self.loudness_batch_norm_3(self.loudness_linear_3(loudness)))

        """ Concatenate part """
        output = torch.stack((frecuency, loudness), dim=0)
        [output, h_n] = self.gru(output)

        output = self.linear_1(output)
        output = self.batch_norm_1(output)
        output = func.relu(output)

        output = func.relu(self.batch_norm_1(self.linear_1(output)))
        output = func.relu(self.batch_norm_2(self.linear_2(output)))
        output = func.relu(self.batch_norm_3(self.linear_3(output)))

        return output
