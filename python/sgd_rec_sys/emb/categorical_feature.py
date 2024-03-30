''' 
特征的类型
    单独互斥类别
        vid uid
    类别向量
        topic vec, 输出emb为融合向量(融合方式avg、sum等)
'''

import torch
from torch import nn
from typing import Union, List


__all__ = ['EmbeddingLayer',]


class EmbeddingLayer(nn.Module):
    '''
    one-hot emb
    multi-hot emb
    '''
    def __init__(self,
                 one_hot_fea_list,
                 one_hot_fea_emb_dim:Union[int, List[int]],
                 multi_hot_fea_list,
                 multi_hot_fea_emb_dim:Union[int, List[int]],
                 cat_emb=False) -> None:
        super().__init__()
        
        # 方便固定编码维度K的参数处理
        if isinstance(one_hot_fea_emb_dim, int):
            one_hot_fea_emb_dim = [one_hot_fea_emb_dim]*len(one_hot_fea_list)
        if isinstance(multi_hot_fea_emb_dim, int):
            multi_hot_fea_emb_dim = [multi_hot_fea_emb_dim]*len(multi_hot_fea_list)

        if one_hot_fea_list:
            self.use_one_hot = True
            self.one_hot_emb_layer = OneHotEmbeddingLayer(one_hot_fea_list,
                                                          one_hot_fea_emb_dim,
                                                          cat_emb=cat_emb)
        
        if multi_hot_fea_list:
            self.use_multi_hot = True
            self.multi_hot_emb_layer = MultiHotEmbeddingLayer(multi_hot_fea_list,
                                                              multi_hot_fea_emb_dim,
                                                              cat_emb=cat_emb)
    
    def forward(self, one_hot_x=None, multi_hot_x=None):
        embs = []
        if self.use_one_hot and (one_hot_x is not None):
            one_hot_emb = self.one_hot_emb_layer(one_hot_x)
            embs.append(one_hot_emb)
        
        if self.use_multi_hot and (multi_hot_x is not None):
            multi_hot_emb = self.multi_hot_emb_layer(multi_hot_x)
            embs.append(multi_hot_emb)
        
        assert len(embs) >= 1
        
        if len(embs)==2:
            return torch.cat(embs, dim=1)
        else:
            return embs[0]

class OneHotEmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings_list, embedding_dim_list, cat_emb=False) -> None:
        super().__init__()
        self.cat_emb = cat_emb
        self.emb_dic = nn.ModuleList()
        for num_embeddings, embedding_dim in zip(num_embeddings_list, embedding_dim_list):
            self.emb_dic.append(
                nn.Embedding(num_embeddings+1,
                             embedding_dim)
                ) # 0 is default(not use)

    def forward(self, x):
        embs = []
        for i in range(x.shape[1]):
            embs.append(self.emb_dic[i](x[:, i])) # 单取一列
        if self.cat_emb:
            return torch.cat(embs, dim=1)
        else:
            return torch.stack(embs, dim=1) # stack增加新维度，cat不新增
        
class MultiHotEmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings_list, embedding_dim_list, mode='mean', cat_emb=False) -> None:
        super().__init__()
        self.cat_emb = cat_emb
        self.emb_dic = nn.ModuleList()
        for num_embeddings, embedding_dim in zip(num_embeddings_list, embedding_dim_list):
            self.emb_dic.append(
                nn.EmbeddingBag(num_embeddings+1, 
                                embedding_dim,
                                mode=mode,
                                padding_idx=0)
                ) # 0 is default(not use)

    def forward(self, x):
        embs = []
        for i in range(len(x)):
            embs.append(self.emb_dic[i](x[i]))
        if self.cat_emb:
            return torch.cat(embs, dim=1)
        else:
            return torch.stack(embs, dim=1) # stack增加新维度，cat不新增
