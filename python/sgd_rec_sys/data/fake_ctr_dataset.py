import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['CtrDataset', 'ctr_collate_fn', 'CtrDataset011', 'ctr_collate_fn_011']

class CtrDataset(Dataset):
    def __init__(self, data_info, transform=None, target_transform=None):
        self.data_meta = data_info['meta']
        self.data_cache = data_info['data']
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data_meta['n']

    def __getitem__(self, idx):
        label = torch.tensor(self.data_cache['label'][idx], dtype=torch.float32)
        dense_x = torch.tensor(self.data_cache['dense'][idx], dtype=torch.float32)
        one_hot_x = torch.tensor(self.data_cache['one-hot'][idx], dtype=torch.long)
        multi_hot_x = self.data_cache['multi-hot'][idx]

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return label, dense_x, one_hot_x, multi_hot_x, 


def ctr_collate_fn(device):
    '''
    # TODO 这种实现真是丑到爆炸，修改
    # 定义一个 collate_fn 函数来处理不同长度的样本
    # 高阶函数，传入device参数，方便batch数据处理
    '''
    
    def collate_fn(batch):
        batch_tuple = list(zip(*batch)) # 行列互换，方便取出第三列数据

        # 处理第4列数据
        tcid = 3 # target col id
        block = batch_tuple[tcid]
        M = len(block)
        N = len(block[0])
        # calc col max length
        col_max = [0]* len(block[0])
        for j in range(N):
            max_len = 0
            for i in range(M):
                max_len = max(len(block[i][j]), max_len)
            col_max[j] = max_len
        # padding
        new_block = [[0]*N for _ in range(M)]
        for i in range(M):
            for j in range(N):
                item = block[i][j]
                new_block[i][j] = item + [0] * (col_max[j]-len(item))
        
        # 按照fea拆分
        np_block = np.array(new_block)
        np_block_list = [np.array(item.tolist()).squeeze() for item in np.hsplit(np_block, np_block.shape[1])]

        
        label = torch.stack(batch_tuple[0], dim=0).to(device)
        dense_x = torch.stack(batch_tuple[1], dim=0).to(device)
        one_hot_x = torch.stack(batch_tuple[2], dim=0).to(device)
        multi_hot_x = [torch.tensor(fea, dtype=torch.long).to(device) for fea in np_block_list] # [fea1, fea2, fea3]
        return label, dense_x, one_hot_x, multi_hot_x
    
    return collate_fn


#################################  无连续特征数据预处理  ###########################################
class CtrDataset011(Dataset):
    def __init__(self, data_info, transform=None, target_transform=None):
        self.data_meta = data_info['meta']
        self.data_cache = data_info['data']
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data_meta['n']

    def __getitem__(self, idx):
        label = torch.tensor(self.data_cache['label'][idx], dtype=torch.float32)
        one_hot_x = torch.tensor(self.data_cache['one-hot'][idx], dtype=torch.long)
        multi_hot_x = self.data_cache['multi-hot'][idx]

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return label, one_hot_x, multi_hot_x


def ctr_collate_fn_011(device):
    '''
    # TODO 这种实现真是丑到爆炸，修改
    # 定义一个 collate_fn 函数来处理不同长度的样本
    # 高阶函数，传入device参数，方便batch数据处理
    '''
    def collate_fn(batch):
        batch_tuple = list(zip(*batch)) # 行列互换，方便取出第三列数据

        # 处理第3列数据
        tcid = 2 # target col id
        block = batch_tuple[tcid]
        M = len(block)
        N = len(block[0])
        # calc col max length
        col_max = [0]* len(block[0])
        for j in range(N):
            max_len = 0
            for i in range(M):
                max_len = max(len(block[i][j]), max_len)
            col_max[j] = max_len
        # padding
        new_block = [[0]*N for _ in range(M)]
        for i in range(M):
            for j in range(N):
                item = block[i][j]
                new_block[i][j] = item + [0] * (col_max[j]-len(item))
        # 按照fea拆分
        np_block = np.array(new_block)
        np_block_list = [np.array(item.tolist()).squeeze() for item in np.hsplit(np_block, np_block.shape[1])]

        label = torch.stack(batch_tuple[0], dim=0).to(device)
        one_hot_x = torch.stack(batch_tuple[1], dim=0).to(device)
        multi_hot_x = [torch.tensor(fea, dtype=torch.long).to(device) for fea in np_block_list] # [fea1, fea2, fea3]
        return label, one_hot_x, multi_hot_x
    
    return collate_fn