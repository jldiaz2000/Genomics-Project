import pickle
import torch
import linecache
import re
import os

PATH = '/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data'
folder_to_keys = pickle.load(open(f'{PATH}/folder_to_keys.pickle', 'rb'))


class DNADataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, folder, set_type):
        'Initialization'
        self.folder = folder
        self.set_type = set_type  # Either 'train' or 'test'
        self.list_IDs = list(folder_to_keys[folder])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        line = linecache.getline(
            f'{PATH}/raw-numerical/{self.folder}/{self.set_type}.txt', index+1)
        result = re.search('(.*) (\w+) (\d+)', line)
        X = result[1]
        y = result[2]

        return X, y


folders = list(folder_to_keys.keys())

folders_using = folders[0:50]
folders_using = folders
print(folders_using)

# Generators
set_lst = []
for folder in folders_using:
    set_lst.append(DNADataset(folder, 'train'))

training_sets = torch.utils.data.ConcatDataset(set_lst)
training_generator = torch.utils.data.DataLoader(training_sets)

print('starting training')
for epoch in range(1):
    # Training
    for local_batch, local_labels in training_generator:
        pass
print('finished train')
