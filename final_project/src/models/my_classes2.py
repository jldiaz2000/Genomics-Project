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
from sklearn.metrics import roc_auc_score


print("Fetching Data")
PATH = Path(os.getcwd())
PATH = PATH.parent.parent
folder_to_keys = pickle.load(open(f'{PATH}/data/folder_to_keys.pickle', 'rb'))
folder_key_to_label = pickle.load(
    open(f'{PATH}/data/folder_key_to_label.pickle', 'rb'))

forward_time = 0
backward_time = 0
load_time = 0


class DNADataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, folder, set_type):
        'Initialization'
        self.folder = folder
        self.set_type = set_type  # Either 'train' or 'test'
        # self.list_IDs = list(folder_to_keys[folder][set_type])
        self.list_IDs, self.key_to_tensor = self.get_tensor_dict()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        key = self.list_IDs[index]
        # X = pickle.load(
        #     open(f'{PATH}/data/preprocessed/{self.folder}/{self.set_type}/{key}.pt', 'rb'))
        # y = folder_key_to_label[self.folder][self.set_type][key]
        X = self.key_to_tensor[key]
        y = folder_key_to_label[self.folder][self.set_type][key[:-3]]
        return X, y

    def get_tensor_dict(self):
        curr_path = f'{PATH}/data/preprocessed/{self.folder}/{self.set_type}'
        scans = set(os.listdir(curr_path)) - set(['.DS_Store'])
        list_IDs = []
        key_to_tensor = dict()
        for scan in scans:
            X = pickle.load(
                open(f'{PATH}/data/preprocessed/{self.folder}/{self.set_type}/{scan}', 'rb'))
            key_to_tensor[scan] = X

            list_IDs.append(scan)

        return list_IDs, key_to_tensor

    @staticmethod
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

    @staticmethod
    def seq_to_tensor(sequence, kmer_to_idx, k):
        lst = []
        kmers = np.lib.stride_tricks.sliding_window_view(sequence, k)
        for i in range(len(kmers)):
            kmer_lst = kmers[i]
            kmer = f'{kmer_lst[0]}{kmer_lst[1]}{kmer_lst[2]}'
            kmer_idx = kmer_to_idx[kmer]
            lst.append(kmer_idx)

        return np.array(lst)


def PaperEmbedding(k):
    assert (k > 2 and k < 9)
    filepath = Path(os.getcwd())
    filepath = str(filepath.parent) + \
        '/features/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.txt'
    kmer_to_idx, embedding, idx_to_kmer = DNADataset.get_k_mer_embedding(
        k, filepath)
    dna2vec = torch.FloatTensor(embedding)

    # One Hot Embedding
    one_hot = torch.diag(torch.ones(4**k))

    # Concatination of the two
    embedding = nn.Embedding.from_pretrained(
        torch.cat((dna2vec, one_hot), axis=1))

    return embedding, kmer_to_idx


class Convolution(nn.Module):
    def __init__(self, k):
        super(Convolution, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=(5, 5), padding=2)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=(5, 5), padding=2)

    def forward(self, input_matrix):
        input_reshaped = input_matrix[:, None, :, :]
        X = self.relu(self.conv1(input_reshaped))
        X = self.relu(self.conv2(X))
        # Paper does not explicitly softmax at the end
        # X = self.softmax(X)
        return X


class LstmAttn(nn.Module):
    def __init__(self, k, input_dim, output_dim):
        super(LstmAttn, self).__init__()
        self.k = k
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, output_dim,
                            batch_first=True, bidirectional=True)

        self.w = nn.Linear(2*output_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_matrix):
        lstm_input = torch.squeeze(input_matrix, 1)
        all_hiddens, (last_hidden, last_cell_state) = self.lstm(lstm_input)
        H = all_hiddens
        M = self.tanh(H)
        wt_M = self.w(M)
        alpha = self.softmax(wt_M)
        H = torch.permute(H, (0, 2, 1))
        r = torch.bmm(H, alpha)
        h_star = self.tanh(r)
        h_star = torch.squeeze(h_star, 2)
        return h_star


class BinaryClassificationHead(nn.Module):
    def __init__(self, k, input_dim, hidden_dim):
        super(BinaryClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 2

        self.hl_1 = nn.Linear(input_dim, hidden_dim)
        self.hl_2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=0.2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_matrix):
        dropout = self.dropout(input_matrix)
        layer1 = self.hl_1(dropout)
        layer2 = self.hl_2(layer1)
        pred = self.sigmoid(layer2)
        pred = torch.squeeze(pred, 1)
        return pred


class PaperModel(nn.Module):
    def __init__(self, k, output_dim, hidden_dim):
        super(PaperModel, self).__init__()
        self.k = k
        self.embedding_dim = 100 + 4**k

        # TODO: check only getting DNA2vec embeddings once
        self.embedding, self.kmer_to_idx = PaperEmbedding(k)
        self.convolution = Convolution(k)
        self.lstm_attn = LstmAttn(k, self.embedding_dim, output_dim)
        self.binary_classification_head = BinaryClassificationHead(
            k, 2*output_dim, hidden_dim)

    def forward(self, input_tensor):
        embedded_input = self.embedding(input_tensor)
        X = self.convolution(embedded_input)
        X = self.lstm_attn(X)
        X = self.binary_classification_head(X)
        return X


class Trainer():
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ############# new ##############
        loss_fn: torch.nn,
        ################################
        gpu_id: int,
        save_interval: int,
        metric_interval: int,
        train_data: DataLoader,
        validation_data: DataLoader = None,
        test_data: DataLoader = None
    ) -> None:
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        ############# new ##############
        self.loss_fn = loss_fn
        ################################
        self.gpu_id = gpu_id
        self.save_interval = save_interval
        self.metric_interval = metric_interval
        self.validation_data = validation_data
        self.test_data = test_data

    def _run_batch(self, batch_tensor, batch_labels):
        self.optimizer.zero_grad()

        predicted_output = self.model(batch_tensor)

        loss = self.loss_fn(predicted_output, batch_labels)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        print(f'\t[GPU {self.gpu_id}] Epoch {epoch}')
        for batch_data, batch_labels in self.train_data:
            batch_tensor = batch_data.to(self.gpu_id)
            batch_labels = torch.tensor(
                (np.array(batch_labels)).astype(np.float32))
            batch_labels = batch_labels.to(self.gpu_id)
            self._run_batch(batch_tensor, batch_labels)

    def _save_checkpoint(self, epoch: int):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, 'checkpoint_model.pt')
        print(f'\tModel Saved at Epoch {epoch}')

    def train(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            self._run_epoch(epoch)
            if self.save_interval > 0 and epoch % self.save_interval == 0:
                self._save_checkpoint(epoch)
            elif epoch == num_epochs:
                self._save_checkpoint(epoch)

            if self.metric_interval > 0 and epoch % self.metric_interval == 0:
                print("\tTrain:")
                self.evaluate(self.train_data)
                if self.validation_data != None:
                    print("\tTest:")
                    self.evaluate(self.validation_data)

    def evaluate(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            cumulative_loss = 0
            num_correct = 0
            total = 0
            num_batches = len(dataloader)
            all_preds = []
            all_labels = []

            for batch_data, batch_labels in dataloader:
                batch_tensor = batch_data.to(self.gpu_id)
                batch_labels = torch.tensor(
                    (np.array(batch_labels)).astype(np.float32))
                batch_labels = batch_labels.to(self.gpu_id)
                predicted_output = self.model(batch_tensor)

                all_labels += batch_labels.cpu().tolist()
                all_preds += predicted_output.cpu().tolist()

                cumulative_loss += self.loss_fn(predicted_output, batch_labels)
                # assuming decision boundary to be 0.5
                total += batch_labels.size(0)
                num_correct += (torch.round(predicted_output)
                                == batch_labels).sum().item()

            loss = cumulative_loss/num_batches
            accuracy = num_correct/total

            curr_roc_score = roc_auc_score(all_labels, all_preds)
            print(f'\t\tLoss: {loss} = {cumulative_loss}/{num_batches}')
            print(f'\t\tAccuracy: {accuracy} = {num_correct}/{total}')
            print(f'\t\tROC AUC: {curr_roc_score} ')
            print()

        self.model.train()


def get_num_train_test(folders_using: list):
    train_count = 0
    test_count = 0
    for folder in folders_using:
        train_count += len(folder_to_keys[folder]['train'])
        test_count += len(folder_to_keys[folder]['test'])

    print(f"Number of train:  {train_count}")
    print(f"Number of test:  {test_count}")


def get_folders_random():
    df = pd.read_csv(f'{PATH}/data/summary.csv')
    df = df[df['quality'] == 'good']

    removable = ['wgEncodeAwgTfbsUtaHelas3CtcfUniPk',
                 'wgEncodeAwgTfbsSydhHepg2Rad21IggrabUniPk',
                 'wgEncodeAwgTfbsSydhHepg2Mafkab50322IggrabUniPk',
                 'wgEncodeAwgTfbsSydhHelas3Pol2UniPk',
                 'wgEncodeAwgTfbsSydhH1hescSuz12UcdUniPk',
                 'wgEncodeAwgTfbsSydhGm12878Pol2IggmusUniPk',
                 'wgEncodeAwgTfbsSydhGm12878Ebf1sc137065UniPk',
                 'wgEncodeAwgTfbsHaibHepg2Sin3ak20Pcr1xUniPk',
                 'wgEncodeAwgTfbsHaibA549Pol2Pcr2xEtoh02UniPk']

    df = df[~df['ID'].isin(removable)]

    df = df[df['cell'].isin(
        {'A549', 'GM12878', 'HepG2', 'H1-hESC', 'HeLa-S3'})].sample(50)
    return list(df['ID'])


def get_folders_sequential():
    df = pd.read_csv(f'{PATH}/data/summary.csv')
    df = df[df['quality'] == 'good']

    removable = ['wgEncodeAwgTfbsUtaHelas3CtcfUniPk',
                 'wgEncodeAwgTfbsSydhHepg2Rad21IggrabUniPk',
                 'wgEncodeAwgTfbsSydhHepg2Mafkab50322IggrabUniPk',
                 'wgEncodeAwgTfbsSydhHelas3Pol2UniPk',
                 'wgEncodeAwgTfbsSydhH1hescSuz12UcdUniPk',
                 'wgEncodeAwgTfbsSydhGm12878Pol2IggmusUniPk',
                 'wgEncodeAwgTfbsSydhGm12878Ebf1sc137065UniPk',
                 'wgEncodeAwgTfbsHaibHepg2Sin3ak20Pcr1xUniPk',
                 'wgEncodeAwgTfbsHaibA549Pol2Pcr2xEtoh02UniPk']

    df = df[~df['ID'].isin(removable)]

    A549_df = df[df['cell'] == 'A549'].head(10)
    GM12878_df = df[df['cell'] == 'GM12878'].head(10)
    HepG2_df = df[df['cell'] == 'HepG2'].head(10)
    H1_hESC_df = df[df['cell'] == 'H1-hESC'].head(10)
    HeLa_S3_df = df[df['cell'] == 'HeLa-S3'].head(10)
    df = pd.concat([A549_df, GM12878_df, HepG2_df, H1_hESC_df,
                   HeLa_S3_df], axis=0, ignore_index=True)

    assert len(list(df['ID'])) == 50

    return list(df['ID'])


def make_train_test(B, random: bool):

    if random:
        folders = get_folders_random()
    else:
        folders = get_folders_sequential()

    train_set_lst = []
    for folder in folders:
        train_set_lst.append(DNADataset(folder, 'train'))

    training_sets = torch.utils.data.ConcatDataset(train_set_lst)
    training_generator = torch.utils.data.DataLoader(training_sets,
                                                     batch_size=B,
                                                     shuffle=True)

    test_set_lst = []
    for folder in folders:
        test_set_lst.append(DNADataset(folder, 'test'))

    test_sets = torch.utils.data.ConcatDataset(test_set_lst)
    test_generator = torch.utils.data.DataLoader(test_sets,
                                                 batch_size=B,
                                                 shuffle=True)

    pickle.dump(training_generator, open(
        f'{PATH}/data/training_generator.pt', 'wb'))
    pickle.dump(test_generator, open(f'{PATH}/data/test_generator.pt', 'wb'))


def load_train_test():

    training_generator = pickle.load(
        open(f'{PATH}/data/training_generator.pt', 'rb'))
    test_generator = pickle.load(open(f'{PATH}/data/test_generator.pt', 'rb'))

    return training_generator, test_generator


def main(device):

    # folders_using = ['wgEncodeAwgTfbsBroadGm12878CtcfUniPk']

    s1 = datetime.now()
    ################## Training Generator ##################
    ################### Test Generator ##################
    ## make_train_test(B=256, random=False)

    # assert False

    training_generator, test_generator = load_train_test()
    ######################################################
    f1 = datetime.now()

    paper_model = PaperModel(k=3, output_dim=16, hidden_dim=16)
    adam_optimizer = torch.optim.Adam(paper_model.parameters(), lr=0.001)
    ce_loss = torch.nn.BCELoss(reduction='mean')
    save_interval = 10
    metric_interval = 10

    trainer = Trainer(paper_model, adam_optimizer, ce_loss, device,
                      save_interval, metric_interval, training_generator, test_generator)

    # assert False
    s2 = datetime.now()
    print('Starting Training')
    num_epochs = 100
    trainer.train(num_epochs)
    print('Finished Training')
    f2 = datetime.now()
    print(f'Time to Generate Dataset {num_epochs} epochs: {f1-s1} (HH:MM:SS)')
    print(f'Time to Train {num_epochs} epochs: {f2-s2} (HH:MM:SS)')

    global load_time
    print(f'load_time: {load_time}')


if __name__ == "__main__":
    import sys
    device = 0
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    main(device)
