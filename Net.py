import torch
import torch.nn as nn
import torch.nn.functional as func

from Parameters import LINEAR_OUT_DIM, OUTPUT_DIM, HIDDEN_DIM, USE_SIGMOID, SIGMOID_EXP


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        """ 1st step """
        self.dense1 = nn.Linear(input_size, output_size)
        self.norm1 = nn.LayerNorm(output_size)

        """ 2nd step """
        self.dense2 = nn.Linear(output_size, output_size)
        self.norm2 = nn.LayerNorm(output_size)

        """ 3rd step """
        self.dense3 = nn.Linear(output_size, output_size)
        self.norm3 = nn.LayerNorm(output_size)

    def forward(self, x):
        y = func.relu(self.norm1(self.dense1(x)))
        y = func.relu(self.norm2(self.dense2(y)))
        y = func.relu(self.norm3(self.dense3(y)))

        return y


class DDSPNet(nn.Module):
    def __init__(self):
        super(DDSPNet, self).__init__()

        self.mlp_f0 = MLP(1, LINEAR_OUT_DIM)
        self.mlp_lo = MLP(1, LINEAR_OUT_DIM)
        self.gru = nn.GRU(2*LINEAR_OUT_DIM, HIDDEN_DIM, batch_first=True)
        self.mlp = MLP(HIDDEN_DIM, HIDDEN_DIM)
        self.dense = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        x_f0 = x["f0"]
        x_lo = x["lo"]

        """ MLP's """
        y_f0 = self.mlp_f0(x_f0)
        y_lo = self.mlp_lo(x_lo)

        """ Concatenate part """
        y = torch.cat((y_f0, y_lo), dim=2)
        y = self.gru(y)[0]
        y = self.mlp(y)
        y = self.dense(y)

        if USE_SIGMOID:
            y = 2 * func.sigmoid(y) ** SIGMOID_EXP + 1e-7

        return y
