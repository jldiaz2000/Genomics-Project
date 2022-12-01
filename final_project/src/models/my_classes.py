import pickle
import numpy as np
import torch
from torch import nn
from  torch.utils.data import Dataset, DataLoader
import linecache
import re
import os
from pathlib import Path
import time
from datetime import datetime, timedelta

print("Fetching Data")
PATH = Path(os.getcwd())
PATH = PATH.parent.parent
folder_to_keys = pickle.load(open(f'{PATH}/data/folder_to_keys.pickle', 'rb'))

class DNADataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, folder, set_type):
        'Initialization'
        self.folder = folder
        self.set_type = set_type  # Either 'train' or 'test'
        self.list_IDs = list(folder_to_keys[folder][set_type])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        line = linecache.getline(
            f'{str(PATH)}/data/raw/{self.folder}/{self.set_type}.txt', index+1)
        result = re.search('(.*) (\w+) (\d+)', line)
        key = result[1]
        X = list(result[2])
        y = result[3]
        return X, y

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
            
        assert(4**k == len(kmer_to_idx))
        assert(embedding.shape[0] == 4**k)
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

    @staticmethod
    # Converts input data to a tensor with B tensors containing a DNA sequences 
    # k-mer indices on the embedding matrix 
    # 
    # input data type: torch.tensor(torch.tensor([torch.float()])) shape (101,B)
    def get_tensor(input_data, kmer_to_idx, k):
        matrix = np.array(input_data).T
    #     assert matrix.shape == (_,101), f'matrix.shape {matrix.shape} != (_,101)'
        return torch.tensor(np.apply_along_axis(DNADataset.seq_to_tensor, 1, matrix, kmer_to_idx = kmer_to_idx, k=k))

def PaperEmbedding(k):
    assert(k > 2 and k < 9)
    filepath = Path(os.getcwd())
    filepath = str(filepath.parent) + '/features/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.txt'
    kmer_to_idx, embedding, idx_to_kmer = DNADataset.get_k_mer_embedding(k, filepath)
    dna2vec = torch.FloatTensor(embedding)

    # One Hot Embedding
    one_hot = torch.diag(torch.ones(4**k))
    
    #Concatination of the two
    embedding = nn.Embedding.from_pretrained(torch.cat( (dna2vec,one_hot) , axis = 1))
    
    return embedding, kmer_to_idx

class Convolution(nn.Module):
    def __init__(self, k):
        super(Convolution, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,5), padding=2)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,5), padding=2)
        
    def forward(self, input_matrix):
        input_reshaped = input_matrix[:,None,:,:]
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
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first = True, bidirectional = True)

        self.w = nn.Linear(2*output_dim, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, input_matrix):
        lstm_input = torch.squeeze(input_matrix, 1)
        all_hiddens, (last_hidden, last_cell_state) =  self.lstm(lstm_input)
        H = all_hiddens
        M = self.tanh(H)
        wt_M = self.w(M)
        alpha = self.softmax(wt_M)
        H = torch.permute(H, (0,2,1))        
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
        
        ## TODO: check only getting DNA2vec embeddings once
        self.embedding, self.kmer_to_idx = PaperEmbedding(k)
        self.convolution = Convolution(k)
        self.lstm_attn = LstmAttn( k, self.embedding_dim, output_dim)
        self.binary_classification_head = BinaryClassificationHead( k, 2*output_dim, hidden_dim)
        
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
        self.validation_data = validation_data
        self.test_data = test_data
    
    def _run_batch(self, batch_tensor, batch_labels):
        self.optimizer.zero_grad()
        predicted_output = self.model(batch_tensor) 
        loss = self.loss_fn(predicted_output, batch_labels)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        ## TODO: Switch prints with tqdm prints 
        print(f'\t[GPU {self.gpu_id}] Epoch {epoch}')
        for batch_data, batch_labels in self.train_data:
            ## TODO: make getting reformatted kmer tensor more efficent
            batch_tensor = DNADataset.get_tensor(batch_data, self.model.kmer_to_idx, self.model.k)
            batch_tensor = batch_tensor.to(self.gpu_id)
            batch_labels = torch.tensor((np.array(batch_labels)).astype(np.float32))
            batch_labels = batch_labels.to(self.gpu_id)
            self._run_batch(batch_tensor, batch_labels)

    def _save_checkpoint(self, epoch: int):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, 'checkpoint_model.pt')
        print(f'\tModel Saved at Epoch {epoch}')

    def train(self, num_epochs: int):
        for epoch in range(1,num_epochs + 1):
            self._run_epoch(epoch)
            if self.save_interval > 0 and epoch % self.self.save_interval:
                self._save_checkpoint(epoch)
            elif epoch == num_epochs:
                self._save_checkpoint(epoch)
        
        if self.validation_data != None:
            self.evaluate(self.train_data)
            self.evaluate(self.validation_data)
        
    def evaluate(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            cumulative_loss = 0
            num_correct = 0
            total = 0
            num_batches = len(dataloader)

            for batch_data, batch_labels in dataloader: 
                batch_tensor = DNADataset.get_tensor(batch_data, self.model.kmer_to_idx, self.model.k)
                batch_tensor = batch_tensor.to(self.gpu_id)
                batch_labels = torch.tensor((np.array(batch_labels)).astype(np.float32))
                batch_labels = batch_labels.to(self.gpu_id)
                predicted_output = self.model(batch_tensor) 

                cumulative_loss += self.loss_fn(predicted_output, batch_labels)
                # assuming decision boundary to be 0.5
                total += batch_labels.size(0)
                num_correct += (torch.round(predicted_output) == batch_labels).sum().item()

            loss = cumulative_loss/num_batches
            accuracy = num_correct/total
            print(f'\tLoss: {loss} = {cumulative_loss}/{num_batches}')
            print(f'\tAccuracy: {accuracy} = {num_correct}/{total}')
            print()

        self.model.train()
                

    def test(self):
        ## TODO: Create function almost identical to validate above
        pass

    def get_AUC_ROC(self):
        ## TODO: Create function to generate AUC_ROC curve
        pass


def main(device):

    folders = list(folder_to_keys.keys())
    folders_using = folders[0:1]
    print(folders_using)

    ################## Training Generator ##################
    train_set_lst = []
    for folder in folders_using:
        train_set_lst.append(DNADataset(folder, 'train'))

    training_sets = torch.utils.data.ConcatDataset(train_set_lst)
    training_generator = torch.utils.data.DataLoader(training_sets, batch_size = 64)
    #######################################################

    ################### Test Generator ##################
    test_set_lst = []
    for folder in folders_using:
        test_set_lst.append(DNADataset(folder, 'test'))

    test_sets = torch.utils.data.ConcatDataset(test_set_lst)
    test_generator = torch.utils.data.DataLoader(test_sets, batch_size = 64)
    ######################################################

    paper_model = PaperModel(k = 3, output_dim = 16, hidden_dim = 16)
    adam_optimizer = torch.optim.Adam(paper_model.parameters(), lr= 0.001)
    ce_loss = torch.nn.BCELoss( reduction = 'mean' )
    save_interval = 0

    trainer = Trainer(paper_model, adam_optimizer, ce_loss, device, save_interval, training_generator, test_generator)

    # assert False
    s = datetime.now()
    print('Starting Training')
    num_epochs = 1
    trainer.train(num_epochs)
    print('Finished Training')
    f = datetime.now()
    print(f'Time to run {num_epochs} epochs: {f-s} (HH:MM:SS)')

if __name__ == "__main__":
    import sys
    device = 0
    main(device)
