import torch

from NetGon import DDSPNet
from DataLoader import Dataset
from Synthese import synthese
from torch.utils.data import DataLoader
from Parameters import STFT_SIZE


torch.set_default_tensor_type(torch.DoubleTensor)

net = DDSPNet()

dataset = Dataset()
data_loader = DataLoader(dataset, batch_size=6, shuffle=True)

for i, data in enumerate(data_loader, 0):
    print(i + 1)

    outputs = net(data)
    f0 = data["frecuency"]

    for k in range(outputs.shape[0]):
        outputs_sample = outputs[k,:,:]
        f0_sample = f0[k,:,:]
        son = synthese(outputs_sample, f0_sample)
        son_tensor = torch.tensor(son)
        stft = torch.stft(son_tensor, STFT_SIZE, window=torch.hann_window(STFT_SIZE), onesided=True)
        squared_module = stft[:,:,0]**2 + stft[:,:,1]**2
