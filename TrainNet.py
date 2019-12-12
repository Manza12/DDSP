import torch

from NetGon import DDSPNet
from DataLoader import Dataset
from Synthese import synthetize
from torch.utils.data import DataLoader
from torch import optim
from Parameters import STFT_SIZE, PATH_TO_MODEL, NUMBER_EPOCHS, FRAME_LENGTH, AUDIO_SAMPLE_RATE


torch.set_default_tensor_type(torch.DoubleTensor)

net = DDSPNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss()

dataset = Dataset()
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(NUMBER_EPOCHS):
    print("Epoch", epoch+1)
    for i, data in enumerate(data_loader, 0):
        print("Data", i + 1)

        fragment, waveform = data

        optimizer.zero_grad()

        y = net(fragment)

        f0s = fragment["f0"][:, :, 0]
        a0s = y[:, :, 0]
        aa = y[:, :, 1:]

        sons = synthetize(a0s, f0s, aa, FRAME_LENGTH, AUDIO_SAMPLE_RATE)

        stft = torch.stft(sons, STFT_SIZE, window=torch.hann_window(STFT_SIZE), onesided=True)
        squared_module = stft[:, :, :, 0] ** 2 + stft[:, :, :, 1] ** 2
        stft_original = torch.stft(waveform[:, 0:sons.shape[1]], STFT_SIZE, window=torch.hann_window(STFT_SIZE), onesided=True)
        squared_module_original = stft_original[:,:,:,0]**2 + stft_original[:,:,:,0]**2

        loss = loss_function(squared_module, squared_module_original)

        print("Loss :", loss.item())

        loss.backward()

        # for k in range(y.shape[0]):
        #     outputs_sample = y[k,:,:]
        #     f0_sample = f0[k,:,:]
        #     son = synthetize() synthese(outputs_sample, f0_sample)
        #     son_tensor = torch.tensor(son)
        #     stft = torch.stft(son_tensor, STFT_SIZE, window=torch.hann_window(STFT_SIZE), onesided=True)
        #     squared_module = stft[:,:,0]**2 + stft[:,:,1]**2
        #
        #     # squared_module.requires_grad_(True)
        #     print(squared_module.grad_fn)
        #
        #     loss += loss_function(squared_module, squared_module_original[k,:,:])



torch.save(net, PATH_TO_MODEL)
