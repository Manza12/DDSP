import torch

from NetGon import DDSPNet
from DataLoader import Dataset
from Synthese import synthese
from torch.utils.data import DataLoader
from Parameters import STFT_SIZE, PATH_TO_MODEL


torch.set_default_tensor_type(torch.DoubleTensor)

net = DDSPNet()
loss_function = torch.nn.MSELoss()

dataset = Dataset()
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for i, data in enumerate(data_loader, 0):
        print(i + 1)

        outputs = net(data)
        f0 = data["frecuency"]
        squared_module_original = data["stft"]

        loss = 0

        for k in range(outputs.shape[0]):
            outputs_sample = outputs[k,:,:]
            f0_sample = f0[k,:,:]
            son = synthese(outputs_sample, f0_sample)
            son_tensor = torch.tensor(son)
            stft = torch.stft(son_tensor, STFT_SIZE, window=torch.hann_window(STFT_SIZE), onesided=True)
            squared_module = stft[:,:,0]**2 + stft[:,:,1]**2
            squared_module.requires_grad_(True)

            loss += loss_function(squared_module, squared_module_original[k,:,:])

        print("Loss :", loss.item())

        loss.backward()

torch.save(net, PATH_TO_MODEL)
