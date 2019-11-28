from DataLoader import raw_2_tensor, audio_2_loudness_tensor


# print("CUDA available :", torch.cuda.is_available())

time_loudness_tensor = audio_2_loudness_tensor("sample_0.wav")
[time_frecuency_tensor, time_confidence_tensor] = raw_2_tensor("sample_0.f0.csv")
