import os
import pickle

# creates final form of data to be called by dataset class

if __name__ == '__main__':

    PATH = '/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data'

    if not os.path.exists(f'{PATH}/final'):
        os.mkdir(f'{PATH}/final')

    data_sets = set(os.listdir(f'{PATH}/interm')) - set(['.DS_Store'])

    i = 1
    for data_set in data_sets:
        print(i)
        i += 1
        curr_train_data_dict = pickle.load(
            open(f'{PATH}/interm/{data_set}/train_data.pickle', 'rb'))

        if not os.path.exists(f'{PATH}/final/{data_set}'):
            os.mkdir(f'{PATH}/final/{data_set}')

        for key, data in curr_train_data_dict.items():
            pickle.dump(data, open(
                f'{PATH}/final/{data_set}/{key}.pickle', 'wb'))

        curr_test_data_dict = pickle.load(
            open(f'{PATH}/interm/{data_set}/test_data.pickle', 'rb'))

        for key, data in curr_test_data_dict.items():
            pickle.dump(data, open(
                f'{PATH}/final/{data_set}/{key}.pickle', 'wb'))
