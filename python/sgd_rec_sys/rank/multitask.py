"""
多目标模型

实现过程：
    * 不考虑embedding layer的模型训练
    * FakeMultiTaskDataFactory直接合成concatenation后的数据
    * 一共有四个输出头（点击率，点赞率，收藏率，转发率）
    
    * BaseNN 多目标基座可以替换成任意fea-cross模型
        * 这里默认使用MLP
    * 输出头的NN结构同理

"""

import torch
import torch.nn as nn

__all__ = ['OutputHead', 'BaseNN', 'MultiTaskNet', 'CrossEntropyLoss']

class OutputHead(nn.Module):
    def __init__(self, in_dim, hidden_dims,) -> None:
        super(OutputHead, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pass


def OutputHead(in_dim, hidden_dims, activation_fun=nn.ReLU()):
    layers = []
    for h_dim in hidden_dims: # 不包含最后一层的结构
        layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
        layers.append(activation_fun)
        in_dim = h_dim
    
    # last layer
    layers.append(nn.Linear(in_features=in_dim, out_features=1, bias=True))
    layers.append(nn.Softmax()) # 输出概率p [0, 1]
    
    return nn.Sequential(*layers)

def BaseNN(in_dim, hidden_dims, activation_fun=nn.ReLU()):   
    layers = []
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
        layers.append(activation_fun)
        in_dim = h_dim
    return nn.Sequential(*layers)


class MultiTaskNet(nn.Module):
    def __init__(self, basenet, heads) -> None:
        super(MultiTaskNet, self).__init__()
        self.base = basenet
        self.heads = heads

    def forward(self, x):
        h = self.base(x)
        result = [head(h) for head in self.heads]
        return torch.cat(result, dim=1) # out [B,N_heads]
            

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        '''
        # 为了避免无穷大，数据溢出的问题，nn.BCELoss做了截断处理
            "Our solution is that BCELoss clamps its log function outputs 
            to be greater than or equal to -100. This way, we can always 
            have a finite loss value and a linear backward method."

            from https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        '''

        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, ps, ys):
        # 输入维度 [B, head_num]
        assert ps.shape == ys.shape
        N = ps.shape[0]
        
        loss_mat = self.bce_loss(ps,ys) # 元素级二元交叉熵 [B, head_num]
        return torch.sum(loss_mat)/N    # 符合王树森讲义上的定义(每个样本上，multi task head上的二元交叉熵loss取sum，batch内loss取mean)

        # 按照下面公式实现，数值上可能会爆掉  
        # N_loss = -(ys * torch.log2(ps) + (1-ys) * torch.log2(1-ps))  
