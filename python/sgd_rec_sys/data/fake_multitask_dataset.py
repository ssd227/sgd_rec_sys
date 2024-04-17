import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['MultiTaskDataset',]

class MultiTaskDataset(Dataset):
    def __init__(self, data_info, device):
        self.data_meta = data_info['meta']
        self.data_cache = data_info['data']
        self.device = device

    def __len__(self):
        return self.data_meta['n']

    def __getitem__(self, idx):
        fea_emb = torch.tensor(self.data_cache['fea_emb'][idx], dtype=torch.float32).to(self.device)
        targets = torch.tensor(self.data_cache['targets'][idx], dtype=torch.float32).to(self.device)
        return fea_emb, targets