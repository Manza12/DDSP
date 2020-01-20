import time

from net import DDSPNet
from dataloader import Dataset
from synthesis import synthetize
from timing import print_time, print_info
from loss import compute_stft, spectral_loss
from torch.utils.data import DataLoader
from torch import optim
from reverb import add_reverb
from scipy.io.wavfile import read as wave_read
from parameters import *


def train(net, dataloader, number_epochs, debug_level):
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, SCHEDULER_RATE)

    time_start = time.time()

    rate, impulse_response = wave_read('ir.wav')
    impulse_response = impulse_response.astype(float)/max(abs(impulse_response))
    impulse_response = torch.from_numpy(impulse_response)
    impulse_response = impulse_response.type(torch.float).to(DEVICE)

    for epoch in range(number_epochs):
        print_info("#### Epoch " + str(epoch+1) + "/" + str(number_epochs)
                   + " ####", debug_level, "TRAIN")
        time_epoch_start = time.time()
        nb_batchs = len(dataloader)
        epoch_loss = 0

        for i, data in enumerate(dataloader, 0):
            print_info("## Data " + str(i + 1) + "/" + str(nb_batchs)
                       + " ##", debug_level, "RUN")
            time_data_start = time.time()
            optimizer.zero_grad()

            fragments, waveforms = data

            time_device_start = time.time()

            fragments["f0"] = fragments["f0"].to(DEVICE)
            fragments["lo"] = fragments["lo"].to(DEVICE)

            print_time("Time to device :", debug_level, "DEBUG",
                       time_device_start, 6)

            time_pre_net = time.time()

            y_additive, y_noise = net(fragments)

            time_pre_synth = print_time("Time through net :", debug_level,
                                        "INFO", time_pre_net, 3)

            f0s = fragments["f0"][:, :, 0]

            a0s = y_additive[:, :, 0]
            aa = y_additive[:, :, 1:NUMBER_HARMONICS + 1]

            hs = y_noise[:, :, 0:NUMBER_NOISE_BANDS + 1]

            if NOISE_ON:
                additive, bruit = synthetize(a0s, f0s, aa, hs, FRAME_LENGTH,
                                             AUDIO_SAMPLE_RATE, DEVICE)
                sons = additive + bruit
            else:
                additive, bruit = synthetize(a0s, f0s, aa, hs, FRAME_LENGTH,
                                             AUDIO_SAMPLE_RATE, DEVICE)
                sons = additive

            if REVERB:
                sons = add_reverb(sons, impulse_response)

            time_post_synth = print_time("Time to synthetize :", debug_level,
                                         "INFO", time_pre_synth, 3)

            """ STFT's """
            waveforms = waveforms.to(DEVICE)

            squared_modules_synth = compute_stft(sons, FFT_SIZES)
            squared_module_truth = compute_stft(waveforms[:, 0:sons.shape[1]],
                                                FFT_SIZES)

            time_post_stft = print_time("Time to perform stfts :", debug_level,
                                        "INFO", time_post_synth, 3)

            """ Loss & Backpropagation """

            loss = spectral_loss(squared_modules_synth, squared_module_truth,
                                 FFT_SIZES)
            loss.backward()
            optimizer.step()

            print_time("Time to backpropagate :", debug_level, "INFO",
                       time_post_stft, 3)

            print_time("Total time :", debug_level, "INFO", time_data_start, 3)

            print_info("\n", debug_level, "DEBUG")
            print_info("Batch Loss : " + str(round(loss.item(), 3)),
                       debug_level, "RUN")
            print_info("\n", debug_level, "DEBUG")

            epoch_loss += loss

        torch.save(net.state_dict(), PATH_TO_CHECKPOINT)
        scheduler.step()

        torch.cuda.empty_cache()

        print_info("\n\n", debug_level, "RUN")
        print_time("Time of the epoch :", debug_level, "TRAIN",
                   time_epoch_start, 3)
        print_info("Epoch Loss : " +
                   str(round(epoch_loss.item() / nb_batchs, 3)),
                   debug_level, "TRAIN")
        print_info("\n\n\n------------\n\n\n", debug_level, "RUN")

    ''' Save '''
    torch.save(net.state_dict(), PATH_TO_MODEL)

    print_time("Training time :", debug_level, "TRAIN", time_start, 3)

    return


if __name__ == "__main__":
    ''' Debug settings '''
    PRINT_LEVEL = "TRAIN"  # Possible modes : DEBUG, INFO, RUN, TRAIN
    print_info("Starting training with debug level : " + PRINT_LEVEL,
               PRINT_LEVEL, "TRAIN")

    ''' Pytorch settings '''
    torch.set_default_tensor_type(torch.FloatTensor)
    print_info("Working device : " + str(DEVICE), PRINT_LEVEL, "INFO")

    ''' Net '''
    Net = DDSPNet().float()
    Net = Net.to(DEVICE)

    ''' Data '''
    Dataset = Dataset()
    Dataloader = DataLoader(Dataset, batch_size=BATCH_SIZE,
                            shuffle=SHUFFLE_DATALOADER)

    ''' Train '''
    train(Net, Dataloader, NUMBER_EPOCHS, PRINT_LEVEL)
