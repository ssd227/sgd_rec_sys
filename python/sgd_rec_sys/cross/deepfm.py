"""
个人理解版本，不参考原论文实现

架构
    deepNet + FMNet

功能
    - 可以处理 dense fea, one_hot_fea, multi_hot_fea
    - 由于FM的限制, 离散型特征需要统一编码维度  emb_dim = K
    - FM部分只负责处理离散型特征的二阶特征交叉
    
实现细节（！！与原论文有出入！！）
    - fmnet 去掉FM中的一阶项，更类似于facebook-DLRM
    - deepnet 增加连续特征, 离散特征原一阶项并入deep net处理
    - deepnet、FM内积输出, 相加后(weight-1连接)过sigmod计算最终的ctr预估值
        - 内积的缺点：特征向量被压扁成单个值
"""

import torch
from torch import nn
from ..emb import EmbeddingLayer

__all__ = ['DeepFM']

class DeepFM(nn.Module):
    def __init__(self,
                 all_fea_dim,
                 fix_emb_dim, # fibinet所有特征维度固定，统一为设为K
                 hidden_dims,
                 one_hot_fea_list,
                 multi_hot_fea_list,
                 ) -> None:
        super().__init__()

        
        self.emb_layer = EmbeddingLayer(one_hot_fea_list,
                                        fix_emb_dim,
                                        multi_hot_fea_list,
                                        fix_emb_dim,
                                        cat_emb=False,) # emb维度[B,F,K]
        
        self.fm_net = FMNet()
        self.deep_net = DeepNet(all_fea_dim, hidden_dims, activation_fun=nn.ReLU())
        
        last_layer_dim = hidden_dims[-1]
        self.linear = nn.Linear(last_layer_dim, 1) # ctr预估，只有一个输出概率
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        dense_x, one_hot_x, multi_hot_x = X
        emb_x = self.emb_layer(one_hot_x, multi_hot_x) # [B, F, K]
        B, F, K = emb_x.shape
        
        # deep out
        x0 = torch.cat((dense_x, emb_x.reshape(B, F*K)), dim=1)
        h_deep = self.deep_net(x0)
        out_deep = self.linear(h_deep)
        
        # fm cross out
        out_fm = torch.sum(self.fm_net(emb_x), dim=1, keepdim=True)
        
        assert out_deep.shape == out_fm.shape
        out = self.sigmoid(out_deep+out_fm)
        return out


def DeepNet(in_dim, hidden_dims, activation_fun=nn.ReLU()):
    layers = []
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
        layers.append(activation_fun)
        in_dim = h_dim
    return nn.Sequential(*layers)


class FMNet(nn.Module):
    '''只实现内积二阶交互'''
    def __init__(self) -> None:
        super().__init__()
        # 二阶内积不需要参数
    def forward(self, x):
        B, F, K = x.shape
        # 计算W·X每一行filed向量的内积交互, [B,F,K] op [B,F,K]=>[B,F,F]
        out = x @ x.reshape(B, K, F) # 内积后得到:[B,F,F]
        # 生成下三角部分的逻辑掩码
        mask = torch.tril(torch.ones(F, F, dtype=torch.bool)).reshape(-1)
        return out.reshape(B, F*F)[:,mask] # field cross out [B, (1+f)*f/2]
        # 注意 out保留了下三角矩阵，fm假设不包括 field_i自身的内积