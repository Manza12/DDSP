from DataLoader import raw_2_tensor, audio_2_loudness_tensor, charge_data


#### Pour vérifier que les données se chargent ####
[audio_tensors, raw_tensors] = charge_data()

#### Pour vérifier si CUDA marche ####
# print("CUDA available :", torch.cuda.is_available())

#### PARTIE MATPLOTIB ####

# import matplotlib.pyplot as plt
#
# time_loudness_tensor = audio_2_loudness_tensor("sample_0.wav")
# time_frecuency_tensor = raw_2_tensor("sample_0.f0.csv")
#
# time_loudness_array = time_loudness_tensor.numpy()
# time_frecuency_array = time_frecuency_tensor.numpy()
#
# time = time_loudness_array[0, :]
# loudness = time_loudness_array[1, :]
# frecuency = time_frecuency_array[:, 1]
#
# plt.plot(time, frecuency)
# plt.show()
