import torch
import torch.nn as nn
import torch.nn.functional as func

from Parameters import SAMPLE_DURATION, RAW_DATA_FRECUENCY, LINEAR_OUT_DIM, OUTPUT_DIM, HIDDEN_DIM


SAMPLE_SIZE = SAMPLE_DURATION * RAW_DATA_FRECUENCY

class DDSPNet(nn.Module):
    def __init__(self):
        super(DDSPNet, self).__init__()

        """ Frecuency MLP """
        self.frecuency_linear_1 = nn.Linear(1, LINEAR_OUT_DIM)
        self.frecuency_linear_2 = nn.Linear(LINEAR_OUT_DIM, LINEAR_OUT_DIM)
        self.frecuency_linear_3 = nn.Linear(LINEAR_OUT_DIM, LINEAR_OUT_DIM)

        self.frecuency_layer_norm_1 = nn.LayerNorm(LINEAR_OUT_DIM)
        self.frecuency_layer_norm_2 = nn.LayerNorm(LINEAR_OUT_DIM)
        self.frecuency_layer_norm_3 = nn.LayerNorm(LINEAR_OUT_DIM)

        """ Loudness MLP """
        self.loudness_linear_1 = nn.Linear(1, LINEAR_OUT_DIM)
        self.loudness_linear_2 = nn.Linear(LINEAR_OUT_DIM, LINEAR_OUT_DIM)
        self.loudness_linear_3 = nn.Linear(LINEAR_OUT_DIM, LINEAR_OUT_DIM)

        self.loudness_layer_norm_1 = nn.LayerNorm(LINEAR_OUT_DIM)
        self.loudness_layer_norm_2 = nn.LayerNorm(LINEAR_OUT_DIM)
        self.loudness_layer_norm_3 = nn.LayerNorm(LINEAR_OUT_DIM)

        """ Concatenated part """
        self.gru = nn.GRU(2*LINEAR_OUT_DIM, HIDDEN_DIM)

        self.linear_1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.linear_2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.linear_3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        self.layer_norm_1 = nn.LayerNorm(HIDDEN_DIM)
        self.layer_norm_2 = nn.LayerNorm(HIDDEN_DIM)
        self.layer_norm_3 = nn.LayerNorm(HIDDEN_DIM)

        self.linear = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, sample):
        frecuency = sample["frecuency"]
        loudness = sample["loudness"]

        """ Frecuency MLP """
        frecuency = func.relu(self.frecuency_layer_norm_1(self.frecuency_linear_1(frecuency)))
        frecuency = func.relu(self.frecuency_layer_norm_2(self.frecuency_linear_2(frecuency)))
        frecuency = func.relu(self.frecuency_layer_norm_3(self.frecuency_linear_3(frecuency)))

        """ Loudness MLP """
        loudness = func.relu(self.loudness_layer_norm_1(self.loudness_linear_1(loudness)))
        loudness = func.relu(self.loudness_layer_norm_2(self.loudness_linear_2(loudness)))
        loudness = func.relu(self.loudness_layer_norm_3(self.loudness_linear_3(loudness)))

        """ Concatenate part """
        output = torch.cat((frecuency, loudness), dim=2)
        output = self.gru(output)[0]

        output = func.relu(self.layer_norm_1(self.linear_1(output)))
        output = func.relu(self.layer_norm_2(self.linear_2(output)))
        output = func.relu(self.layer_norm_3(self.linear_3(output)))

        output = self.linear(output)

        return output
