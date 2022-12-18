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

from my_classes2 import PaperModel, Trainer, DNADataset

PATH = Path(os.getcwd())
PATH = PATH.parent.parent


def main(device):

    training_generator = None

    # Loading in small test set
    test_generator = pickle.load(
        open(f'{PATH}/data/small_test_generator.pt', 'rb'))

    paper_model = PaperModel(k=3, output_dim=16, hidden_dim=16)

    # Loading in Trained model
    paper_model.load_state_dict(torch.load(
        f'{PATH}/src/models/checkpoint_model.pt'))

    adam_optimizer = torch.optim.Adam(paper_model.parameters(), lr=0.001)
    ce_loss = torch.nn.BCELoss(reduction='mean')
    save_interval = 10
    metric_interval = 10

    trainer = Trainer(paper_model, adam_optimizer, ce_loss, device,
                      save_interval, metric_interval, training_generator, test_generator)

    trainer.evaluate(test_generator)


if __name__ == "__main__":
    device = 0
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    main(device)
