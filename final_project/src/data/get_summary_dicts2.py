import os
import pickle
from pathlib import Path
import linecache
import re
import itertools


# pickles dictionaries:
#   - folder_to_keys: maps data set folder to a set of sequence keys
#   - folder_key_to_label: maps a folder + keys to a label

if __name__ == '__main__':

    PATH = Path(os.getcwd())
    PATH = f'{str(PATH.parent.parent)}/data'

    # assert(len(set(os.listdir(PATH))) == len(os.listdir(PATH)))
    # assert('.DS_Store' in os.listdir(PATH))
    data_sets = set(os.listdir(f'{PATH}/raw')) - set(['.DS_Store'])
    # assert(len(data_sets) + 1 == len(os.listdir(PATH)))

    folder_to_keys = {folder: {'train': set(), 'test': set()}
                      for folder in data_sets}

    folder_key_to_label = {folder: {'train': dict(), 'test': dict()}
                           for folder in data_sets}

    folders_to_use = ['wgEncodeAwgTfbsHaibA549Atf3V0422111Etoh02UniPk', 'wgEncodeAwgTfbsBroadGm12878CtcfUniPk',
                      'wgEncodeAwgTfbsBroadHepg2CtcfUniPk', 'wgEncodeAwgTfbsBroadH1hescChd1a301218aUniPk', 'wgEncodeAwgTfbsBroadHelas3CtcfUniPk']

    i = 1
    for data_set in data_sets:
        print(i)
        i += 1

        if data_set in folders_to_use:
            train_lines = open(f'{PATH}/raw/{data_set}/train.txt').readlines()
            for line in train_lines:
                result = re.search('(.*) (\w+) (\d+)', line)
                folder_to_keys[data_set]['train'].add(result[1][1:])
                folder_key_to_label[data_set]['train'][result[1]
                                                       [1:]] = result[3]

            test_lines = open(f'{PATH}/raw/{data_set}/test.txt').readlines()
            for line in test_lines:
                result = re.search('(.*) (\w+) (\d+)', line)
                folder_to_keys[data_set]['test'].add(result[1][1:])
                folder_key_to_label[data_set]['test'][result[1]
                                                      [1:]] = result[3]
        else:
            pass

    pickle.dump(folder_to_keys, open(
        f'{PATH}/small_folder_to_keys.pickle', 'wb'))
    pickle.dump(folder_key_to_label, open(
        f'{PATH}/small_folder_key_to_label.pickle', 'wb'))
