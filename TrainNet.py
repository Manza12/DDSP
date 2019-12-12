import torch
import time

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

        # Time #
        time_pre_net = time.time()
        ########

        y = net(fragment)

        # Time #
        time_post_net = time.time()
        print("Time through net", time_post_net - time_pre_net)
        ########

        f0s = fragment["f0"][:, :, 0]
        a0s = y[:, :, 0]
        aa = y[:, :, 1:]

        # Time #
        time_pre_synth = time.time()
        ########

        sons = synthetize(a0s, f0s, aa, FRAME_LENGTH, AUDIO_SAMPLE_RATE)

        # Time #
        time_post_synth = time.time()
        print("Time to synthetize", time_post_synth - time_pre_synth)
        ########

        stft = torch.stft(sons, STFT_SIZE, window=torch.hann_window(STFT_SIZE), onesided=True)
        squared_module = stft[:, :, :, 0] ** 2 + stft[:, :, :, 1] ** 2
        stft_original = torch.stft(waveform[:, 0:sons.shape[1]], STFT_SIZE, window=torch.hann_window(STFT_SIZE), onesided=True)
        squared_module_original = stft_original[:,:,:,0]**2 + stft_original[:,:,:,0]**2

        # Time #
        time_post_stft = time.time()
        print("Time to perform stft", time_post_stft - time_post_synth)
        ########

        loss = loss_function(squared_module, squared_module_original)

        print("Loss :", loss.item())

        loss.backward()

        # Time #
        time_end = time.time()
        print("Time to backpropagate", time_end - time_post_stft)
        ########

torch.save(net, PATH_TO_MODEL)
