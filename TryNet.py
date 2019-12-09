import torch

from NetGon import DDSPNet
from DataLoader import Dataset
from Synthese import synthese_harmonique
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.DoubleTensor)

net = DDSPNet()

dataset = Dataset()
data_loader = DataLoader(dataset, batch_size=6, shuffle=True)

for i, data in enumerate(data_loader, 0):
    print(i + 1)

    outputs = net(data)
    f0 = data["frecuency"]

    son = synthese_harmonique(outputs, f0)

    # stft_1 = torch.stft(son)
