import torch

from Parameters import PATH_TO_MODEL, AUDIO_SAMPLING_RATE
from DataLoader import Dataset
from torch.utils.data import DataLoader
from Synthese import synthese
from DataLoader import raw_2_tensor
import scipy.io.wavfile as wav
import os


saved_net = torch.load(PATH_TO_MODEL)
audio_name = "First_result.wav"
path_audio = os.path.join("Outputs", audio_name)

dataset = Dataset()
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# raw_file = dataset.raw_files[0]
# f0 = raw_2_tensor(raw_file)

for i, data in enumerate(data_loader, 0):
    f0 = data["frecuency"]
    outputs = saved_net(data)
    outputs_sample = outputs[0, :, :]
    f0_sample = f0[0, :, :]
    son = synthese(outputs_sample, f0_sample)
    wav.write(path_audio, AUDIO_SAMPLING_RATE, son)
    break

