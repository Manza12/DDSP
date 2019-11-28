from DataLoader import raw_2_tensor, audio_2_loudness_tensor

time_loudness_tensor = audio_2_loudness_tensor("sample_0.wav")
[time_frecuency_tensor, time_confidence_tensor] = raw_2_tensor("sample_0.f0.csv")

# Pour vérifier si CUDA marche
# print("CUDA available :", torch.cuda.is_available())

# PARTIE MATPLOTIB
#
# import torch
# import matplotlib.pyplot as plt
#
# time_frecuency_array = time_frecuency_tensor.numpy()
#
# plt.plot(time_frecuency_array[:, 0], time_frecuency_array[:, 1])
# plt.show()

