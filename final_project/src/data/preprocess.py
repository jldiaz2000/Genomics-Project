import re
import os
import pickle


# creates interim dictionaries


if __name__ == '__main__':

    # regex = re.compile("f\(\s*([^,]+)\s*,\s*([^,]+)\s*\)")
    regex = re.compile('/(.*) (\w+) (\d+)/')

    PATH = '/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data/raw'
    i = 1

    assert(len(set(os.listdir(PATH))) == len(os.listdir(PATH)))
    assert('.DS_Store' in os.listdir(PATH))
    data_sets = set(os.listdir(PATH)) - set(['.DS_Store'])
    assert(len(data_sets) + 1 == len(os.listdir(PATH)))

    for data_set in data_sets:
        print(i)
        i += 1
        # subject_lst.append(subject)

        curr_train_data_dict = dict()
        curr_train_labels_dict = dict()

        with open(f'{PATH}/{data_set}/train.txt') as f:
            for line in f.readlines():
                result = re.search('(.*) (\w+) (\d+)', line)
                key = (result[1])[1:]
                curr_train_data_dict[key] = result[2]
                curr_train_labels_dict[key] = result[3]

        path2 = f'/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data/interm/{data_set}'
        if not os.path.exists(path2):
            os.mkdir(path2)

        pickle.dump(curr_train_data_dict, open(
            f'{path2}/train_data.pickle', 'wb'))
        pickle.dump(curr_train_labels_dict, open(
            f'{path2}/train_labels.pickle', 'wb'))

        curr_test_data_dict = dict()
        curr_test_labels_dict = dict()

        with open(f'{PATH}/{data_set}/test.txt') as f:
            for line in f.readlines():
                result = re.search('(.*) (\w+) (\d+)', line)
                key = (result[1])[1:]
                curr_test_data_dict[key] = result[2]
                curr_test_labels_dict[key] = result[3]

        pickle.dump(curr_test_data_dict, open(
            f'{path2}/test_data.pickle', 'wb'))
        pickle.dump(curr_test_labels_dict, open(
            f'{path2}/test_labels.pickle', 'wb'))
