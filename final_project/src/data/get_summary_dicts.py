import os
import pickle
from pathlib import Path
import linecache
import re
import itertools


# pickles dictionaries:
#   - folder_to_keys: maps data set folder to a set of sequence keys
#   - keys_to_folder: maps sequence keys to a data set folder
#   - keys_to_label: maps a sequence key to a label

if __name__ == '__main__':
    # PATH = '/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data/interm'
    # PATH = '/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data/raw'

    PATH = Path(os.getcwd())
    PATH = f'{str(PATH.parent.parent)}/data'

    # assert(len(set(os.listdir(PATH))) == len(os.listdir(PATH)))
    # assert('.DS_Store' in os.listdir(PATH))
    data_sets = set(os.listdir(f'{PATH}/raw')) - set(['.DS_Store'])
    # assert(len(data_sets) + 1 == len(os.listdir(PATH)))

    folder_to_keys = {folder: {'train': set(), 'test': set()}
                      for folder in data_sets}

    i = 1
    for data_set in data_sets:
        print(i)
        i += 1
        train_lines = open(f'{PATH}/raw/{data_set}/train.txt').readlines()
        for line in train_lines:
            result = re.search('(.*) (\w+) (\d+)', line)
            folder_to_keys[data_set]['train'].add(result[1][1:])

        test_lines = open(f'{PATH}/raw/{data_set}/test.txt').readlines()
        for line in test_lines:
            result = re.search('(.*) (\w+) (\d+)', line)
            folder_to_keys[data_set]['test'].add(result[1][1:])

    pickle.dump(folder_to_keys, open(f'{PATH}/folder_to_keys.pickle', 'wb'))
