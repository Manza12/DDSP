import torch
from DataLoader import raw_2_tensor

print(torch.cuda.is_available())

raw_2_tensor("example.csv")


