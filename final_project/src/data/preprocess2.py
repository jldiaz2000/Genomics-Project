import re
import os
import pickle


# creates raw-numerical data

def seq_to_num(seq):
    if 'N' in seq:
        print(seq)

    base_to_num = {'N': '0', 'A': '1', 'T': '2', 'C': '3', 'G': '4'}
    new_seq = ''.join(map(lambda x: base_to_num[x], seq))
    return new_seq


if __name__ == '__main__':

    # regex = re.compile("f\(\s*([^,]+)\s*,\s*([^,]+)\s*\)")
    regex = re.compile('/(.*) (\w+) (\d+)/')

    PATH = '/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data/raw'
    path2 = f'/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data/raw-numerical'
    if not os.path.exists(path2):
        os.mkdir(path2)
    i = 1

    assert(len(set(os.listdir(PATH))) == len(os.listdir(PATH)))
    assert('.DS_Store' in os.listdir(PATH))
    data_sets = set(os.listdir(PATH)) - set(['.DS_Store'])
    assert(len(data_sets) + 1 == len(os.listdir(PATH)))

    for data_set in data_sets:
        print(i)
        i += 1

        write_path = f'{path2}/{data_set}'
        if not os.path.exists(write_path):
            os.mkdir(write_path)

        with open(f'{PATH}/{data_set}/train.txt', 'r') as f1:
            with open(f'{write_path}/train.txt', 'w') as f2:
                for line in f1.readlines():
                    result = re.search('(.*) (\w+) (\d+)', line)
                    new_seq = seq_to_num(result[2])
                    new_line = f'{(result[1])[1:]} {new_seq} {result[3]}\n'
                    f2.write(new_line)

        with open(f'{PATH}/{data_set}/test.txt', 'r') as f1:
            with open(f'{write_path}/test.txt', 'w') as f2:
                for line in f1.readlines():
                    result = re.search('(.*) (\w+) (\d+)', line)
                    new_seq = seq_to_num(result[2])
                    new_line = f'{(result[1])[1:]} {new_seq} {result[3]}\n'
                    f2.write(new_line)
