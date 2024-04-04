import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['DssmDataset',]

class DssmDataset(Dataset):
    def __init__(self, data_info, device):
        self.data_meta = data_info['meta']
        self.data_cache = data_info['data']
        self.device = device

    def __len__(self):
        return self.data_meta['n']

    def __getitem__(self, idx):
        user_fea_emb = torch.tensor(self.data_cache['user_fea_emb'][idx], dtype=torch.float32).to(self.device)
        item_fea_emb = torch.tensor(self.data_cache['item_fea_emb'][idx], dtype=torch.float32).to(self.device)
        return user_fea_emb, item_fea_emb