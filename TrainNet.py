import torch
import time

from Net import DDSPNet
from DataLoader import Dataset
from Synthese import synthetize
from torch.utils.data import DataLoader
from torch import optim
from Parameters import STFT_SIZE, PATH_TO_MODEL, NUMBER_EPOCHS, FRAME_LENGTH, AUDIO_SAMPLE_RATE, \
    DEVICE, SHUFFLE_DATALOADER, BATCH_SIZE, LEARNING_RATE


#### Debug settings ####
PRINT_LEVEL = "TRAIN"  # Possible modes : DEBUG, INFO, RUN, TRAIN

#### Pytorch settings ####
torch.set_default_tensor_type(torch.DoubleTensor)

#### Net ####
net = DDSPNet()
net = net.to(DEVICE)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.MSELoss()

if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO":
    print("Working device :", DEVICE)

#### Data ####
dataset = Dataset()
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATALOADER)

#### Train ####
for epoch in range(NUMBER_EPOCHS):
    if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO" or PRINT_LEVEL == "RUN" or PRINT_LEVEL == "TRAIN":
        print("#### Epoch", epoch+1, "####")

    # Time #
    if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO" or PRINT_LEVEL == "RUN":
        time_epoch_start = time.time()
    else:
        time_epoch_start = None
    ########

    for i, data in enumerate(data_loader, 0):
        if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO" or PRINT_LEVEL == "RUN":
            print("## Data", i + 1, "##")

        # Time #
        if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO":
            time_start = time.time()
        else:
            time_start = None
            ########

        fragment, waveform = data

        fragment["f0"] = fragment["f0"].to(DEVICE)
        fragment["lo"] = fragment["lo"].to(DEVICE)

        optimizer.zero_grad()

        # Time #
        if PRINT_LEVEL == "DEBUG":
            time_pre_net = time.time()
        else:
            time_pre_net = None
        ########

        y = net(fragment)

        # Time #
        if PRINT_LEVEL == "DEBUG":
            time_post_net = time.time()
            print("Time through net :", round(time_post_net - time_pre_net, 3), "s")
        ########

        f0s = fragment["f0"][:, :, 0]
        a0s = y[:, :, 0]
        aa = y[:, :, 1:]

        # Time #
        if PRINT_LEVEL == "DEBUG":
            time_pre_synth = time.time()
        else:
            time_pre_synth = None
        ########

        sons = synthetize(a0s, f0s, aa, FRAME_LENGTH, AUDIO_SAMPLE_RATE)

        # Time #
        if PRINT_LEVEL == "DEBUG":
            time_post_synth = time.time()
            print("Time to synthetize :", round(time_post_synth - time_pre_synth, 3), "s")
        else:
            time_post_synth = None
        ########

        """ STFT's """
        waveform = waveform.to(DEVICE)
        window = torch.hann_window(STFT_SIZE, device=DEVICE)
        stft = torch.stft(sons, STFT_SIZE, window=window, onesided=True)
        squared_module = stft[:, :, :, 0] ** 2 + stft[:, :, :, 1] ** 2
        stft_original = torch.stft(waveform[:, 0:sons.shape[1]], STFT_SIZE, window=window, onesided=True)
        squared_module_original = stft_original[:,:,:,0]**2 + stft_original[:,:,:,0]**2

        # Time #
        if PRINT_LEVEL == "DEBUG":
            time_post_stft = time.time()
            print("Time to perform stft :", round(time_post_stft - time_post_synth, 3), "s")
        else:
            time_post_stft = None
        ########

        loss = loss_function(squared_module, squared_module_original)
        loss.backward()

        # Time #
        if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO":
            time_end = time.time()
        else:
            time_end = None
        if PRINT_LEVEL == "DEBUG":
            print("Time to backpropagate :", round(time_end - time_post_stft, 3), "s")

        ########

        if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO" or PRINT_LEVEL == "RUN":
            print("Loss :", loss.item())

        # Time #
        if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO":
            print("Total time :", round(time_end - time_start, 3), "s")
        ########

    # Time #
    if PRINT_LEVEL == "DEBUG" or PRINT_LEVEL == "INFO" or PRINT_LEVEL == "RUN":
        time_epoch_end = time.time()
        print("Time of the epoch :", round(time_epoch_end - time_epoch_start, 3), "s")
    ########

#### Save ####
torch.save(net, PATH_TO_MODEL)
