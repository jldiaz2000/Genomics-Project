import os
import pickle

if __name__ == '__main__':
    # file to set of IDs
    #

    PATH = '/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data/interm'

    # assert(len(set(os.listdir(PATH))) == len(os.listdir(PATH)))
    # assert('.DS_Store' in os.listdir(PATH))
    data_sets = set(os.listdir(PATH)) - set(['.DS_Store'])
    # assert(len(data_sets) + 1 == len(os.listdir(PATH)))

    folder_to_keys = {folder: {'train': set(), 'test': set()}
                      for folder in data_sets}
    keys_to_folder = dict()
    keys_to_label = dict()

    i = 1
    for data_set in data_sets:
        print(i)
        i += 1
        curr_train_data_dict = pickle.load(
            open(f'{PATH}/{data_set}/train_data.pickle', 'rb'))
        curr_train_keys = set(curr_train_data_dict.keys())
        folder_to_keys[data_set]['train'] = curr_train_keys
        for key in curr_train_keys:
            keys_to_folder[key] = data_set

        curr_train_labels_dict = pickle.load(
            open(f'{PATH}/{data_set}/train_labels.pickle', 'rb'))
        for key in curr_train_keys:
            keys_to_label[key] = curr_train_labels_dict[key]

        curr_test_data_dict = pickle.load(
            open(f'{PATH}/{data_set}/test_data.pickle', 'rb'))
        curr_test_keys = set(curr_test_data_dict.keys())
        folder_to_keys[data_set]['test'] = curr_test_keys
        for key in curr_test_keys:
            keys_to_folder[key] = data_set

        curr_test_labels_dict = pickle.load(
            open(f'{PATH}/{data_set}/test_labels.pickle', 'rb'))
        for key in curr_test_keys:
            keys_to_label[key] = curr_test_labels_dict[key]

    path2 = f'/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data/'
    pickle.dump(folder_to_keys, open(f'{path2}/folder_to_keys.pickle', 'wb'))
    pickle.dump(keys_to_folder, open(f'{path2}/keys_to_folder.pickle', 'wb'))
    pickle.dump(keys_to_label, open(f'{path2}/keys_to_label.pickle', 'wb'))
