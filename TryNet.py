import torch

from NetGon import DDSPNet
from DataLoader import Dataset
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.DoubleTensor)

net = DDSPNet()

dataset = Dataset()
data_loader = DataLoader(dataset, batch_size=6, shuffle=True)

for i, data in enumerate(data_loader, 0):
    # get the inputs; data is a list of [inputs, labels]

    outputs = net(data)

    print(i+1)
