import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import linecache
import re
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
import pandas as pd


PATH = Path(os.getcwd())
PATH = PATH.parent.parent
# folder_to_keys = pickle.load(open(f'{PATH}/data/folder_to_keys.pickle', 'rb'))
# folder_key_to_label = pickle.load(
#     open(f'{PATH}/data/folder_key_to_label.pickle', 'rb'))


def get_k_mer_embedding(k, path):
    # path is path ending in /src/features/name-of-txt
    k_mer_embedding = dict()
    with open(path) as f:
        for line in f:
            result = re.search('(\w*)(( (-?\d*\.?\d*))*)', line)
            curr_key = result[1]
            if len(result[1]) == k:
                vector = result[2].split(' ')
                vector = list(map(lambda x: float(x), vector[1:]))
                k_mer_embedding[curr_key] = np.array(vector)

    idx_to_kmer = list(k_mer_embedding.keys())
    kmer_to_idx = dict(zip(idx_to_kmer, np.arange(4**k)))
    embedding = np.array(list(k_mer_embedding.values()))

    assert (4**k == len(kmer_to_idx))
    assert (embedding.shape[0] == 4**k)
    return kmer_to_idx, embedding, idx_to_kmer


def seq_to_tensor(sequence, kmer_to_idx, k):
    lst = []
    kmers = np.lib.stride_tricks.sliding_window_view(sequence, k)
    for i in range(len(kmers)):
        kmer_lst = kmers[i]
        kmer = f'{kmer_lst[0]}{kmer_lst[1]}{kmer_lst[2]}'
        kmer_idx = kmer_to_idx[kmer]
        lst.append(kmer_idx)

    return np.array(lst)


def get_tensor(input_data, kmer_to_idx, k):
    matrix = np.array(input_data).T
    #     assert matrix.shape == (_,101), f'matrix.shape {matrix.shape} != (_,101)'
    return torch.tensor(np.apply_along_axis(seq_to_tensor, 1, matrix, kmer_to_idx=kmer_to_idx, k=k))


def main():
    path = f'{PATH}/src/features/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.txt'
    k = 3
    kmer_to_idx, embedding, idx_to_kmer = get_k_mer_embedding(k, path)

    check = set(os.listdir(f'{PATH}/data/raw')) - set(['.DS_Store'])

    df = pd.read_csv(f'{PATH}/data/summary.csv')
    df = df[df['quality'] == 'good']
    df = df[df['cell'].isin(
        {'A549', 'GM12878', 'HepG2', 'H1-hESC', 'HeLa-S3'})]

    data_sets = list(df['ID'])

    assert set(data_sets).issubset(check)

    if not os.path.exists(f'{PATH}/data/preprocessed'):
        os.mkdir(f'{PATH}/data/preprocessed')

    i = 1
    denom = len(data_sets)
    for data_set in data_sets:
        print(f'{i}/{denom}')
        i += 1
        if not os.path.exists(f'{PATH}/data/preprocessed/{data_set}'):
            os.mkdir(f'{PATH}/data/preprocessed/{data_set}')
            os.mkdir(f'{PATH}/data/preprocessed/{data_set}/train')
            os.mkdir(f'{PATH}/data/preprocessed/{data_set}/test')

        try:

            train_lines = open(
                f'{PATH}/data/raw/{data_set}/train.txt').readlines()
            for line in train_lines:
                result = re.search('(.*) (\w+) (\d+)', line)
                key = result[1][1:]
                X = list(result[2])
                # print(X)
                # print(len(X))
                tensor = seq_to_tensor(X, kmer_to_idx, k)
                # print(tensor)
                # print(tensor.shape)
                # assert False
                pickle.dump(tensor, open(
                    f'{PATH}/data/preprocessed/{data_set}/train/{key}.pt', 'wb'))

            test_lines = open(
                f'{PATH}/data/raw/{data_set}/test.txt').readlines()
            for line in test_lines:
                result = re.search('(.*) (\w+) (\d+)', line)
                key = result[1][1:]
                X = list(result[2])
                tensor = seq_to_tensor(X, kmer_to_idx, k)
                pickle.dump(tensor, open(
                    f'{PATH}/data/preprocessed/{data_set}/test/{key}.pt', 'wb'))

        except KeyError as e:
            print(e)
            os.system(f'rm -r {PATH}/data/preprocessed/{data_set}')
            print(f'Removing directory {PATH}/data/preprocessed/{data_set}')
            print(
                f'\'{data_set}\' should be removed from the set of candidate datasets')


if __name__ == '__main__':
    main()
