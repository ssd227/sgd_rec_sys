import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['LastnDataset',]

class LastnDataset(Dataset):
    def __init__(self, data_info, device):
        self.data_meta = data_info['meta']
        self.data_cache = data_info['data']
        self.device = device

    def __len__(self):
        return self.data_meta['n']

    def __getitem__(self, idx):
        user_behavior = torch.tensor(self.data_cache['user_behavior'][idx], dtype=torch.float32).to(self.device)
        candidate_ad = torch.tensor(self.data_cache['candidate_ad'][idx], dtype=torch.float32).to(self.device)
        return candidate_ad, user_behavior