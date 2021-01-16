"""
Inference Engine utilities

A collection of classes, methods, functions and other data structures that implement helper functions
for the inference engine. These utilities facilitate both training and inference modes.
"""

__author__ = "tyronevb"
__date__ = "2021"

import torch
from torch.utils.data import Dataset


class LogSeqDataset(Dataset):
    """
    Load log file dataset consisting of input squence and label key.
    """
    def __init__(self, logs, labels):
        self.logs = logs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.logs[idx], dtype=torch.float).unsqueeze(dim=-1), \
               self.labels[idx]
    # need to unsequeeze to get the input size to match what the lstm model expects
# end

# todo --> possibly more relavant on the framework level?
def train_test_split():
    pass
