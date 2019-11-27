import torch
from DataLoader import raw_2_tensor

raw_2_tensor("example")

print(torch.cuda.is_available())
