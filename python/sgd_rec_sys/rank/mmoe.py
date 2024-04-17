"""
MMOE

实现细节：
    * 不考虑embedding layer的模型训练
    * FakeMultiTaskDataFactory直接合成concatenation后的数据
    * 共四个输出头-TaskHead（点击率，点赞率，收藏率，转发率）
    * 每个输出头对应一个TaskGate，用来merge专家输出。
    
    * Expert 多目标专家模型可以替换成任意fea-cross模型，
        专家数量是超参数，这里默认使用MLP。
    
    * TaskGate中，Softmax输出的𝑛 个数值被mask 的概率都是10%。
        每个“专家”被随机丢弃的概率都是10%。

    TODO
        * gate net如果简化为MLP, 可以通过垂直叠加矩阵，结果垂直分割后就是各个gate的输出值（p1,p2,p3...）
        * expert net 同理也可以进行相同的并行计算
            但是通过torch.einsum好像计算效率更加生猛（得研究下这个函数的原理和正确性）
        * head net 对应 N_task个 input单独操作， 如何提高并行度呢
        
"""

import torch
import torch.nn as nn

__all__ = ['MMOE', 'Expert', 'TaskHead', 'TaskGate']


def TaskGate(in_dim, hidden_dims, expert_num, activation_fun=nn.ReLU(), dropout_p=0.1):
    layers = []
    for h_dim in hidden_dims: # 不包含最后一层的结构
        layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
        layers.append(activation_fun)
        in_dim = h_dim

    # last layer
    layers.append(nn.Linear(in_features=in_dim, out_features=expert_num, bias=True))
    layers.append(nn.Softmax()) # 输出概率p [0, 1]
    layers.append(nn.Dropout(p=dropout_p)) # 按照10%的概率丢弃专家，防止极化问题
    
    return nn.Sequential(*layers)

def TaskHead(in_dim, hidden_dims, activation_fun=nn.ReLU()):
    layers = []
    for h_dim in hidden_dims: # 不包含最后一层的结构
        layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
        layers.append(activation_fun)
        in_dim = h_dim

    # last layer
    layers.append(nn.Linear(in_features=in_dim, out_features=1, bias=True))
    layers.append(nn.Softmax()) # 输出概率p [0, 1]
    
    return nn.Sequential(*layers)

def Expert(in_dim, hidden_dims, activation_fun=nn.ReLU()):   
    layers = []
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
        layers.append(activation_fun)
        in_dim = h_dim
    return nn.Sequential(*layers)


class MMOE(nn.Module):
    def __init__(self, expert_nets, heads_nets, gate_nets) -> None:
        super(MMOE, self).__init__()
        self.experts = expert_nets # 专家网络可以自定义
        self.heads = heads_nets
        self.gates = gate_nets

    def forward(self, x):
        # TODO 优化计算效率，迭代不太行
        h_expert_list = torch.stack([expert(x) for expert in self.experts], dim=1) # [B, N_expert, K]
        h_gate_list = torch.stack([gate(x) for gate in self.gates], dim=1) # [B, Task_num, N_expert]

        head_input = torch.einsum('bek, bte -> btk', h_expert_list, h_gate_list) # [B, Task_num, K]

        result = [self.heads[i](head_input[:,i,:]) for i in range(len(self.heads))]
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
