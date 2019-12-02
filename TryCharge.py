from torch.utils.data import DataLoader
from DataLoader import Dataset

dataset = Dataset()

samples_list = list()

length_dataset = len(dataset)
for i in range(length_dataset):
    print("Charging sample ", i+1, "/",length_dataset, "...")
    sample = dataset[i]
    samples_list.append(sample)

print("Samples charged.")

data_load = DataLoader(dataset, batch_size=16, shuffle=True)
