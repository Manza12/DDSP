import torch
from DataLoader import raw_2_tensor

print("CUDA available :", torch.cuda.is_available())

print(raw_2_tensor("example.csv"))


