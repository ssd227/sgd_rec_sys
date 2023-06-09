"""
矩阵补充/矩阵分解算法（过时的召回算法）

缺点：
    1|仅⽤ID embedding，没利⽤物品、⽤户属性
    2|负样本：曝光之后，没有点击、交互。（错误的做法）
    3|做训练的⽅法不好(回归不如分类，内积不如余弦相似度)

TODO:
    # 1|简单的矩阵分解算法，后续可以升级加入偏差系数
        # bi 物品偏差
        # bu 用户偏差
        # u  常量 用户的平均打分

    # 2|稀疏矩阵分解实现，统一接口。
"""

import torch
from torch import nn

class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding()
        self.item_emb = nn.Embedding()

        self.user_matrix = nn.Parameter(torch.randn(4, 2, requires_grad=True) * 1e-1)
        self.item_matrix = nn.Parameter(torch.randn(2, 4, requires_grad=True) * 1e-1)

    def forward(self, x):
        score_matrix = self.user_matrix.matmul(self.item_matrix)
        xs, ys = list(zip(*x))

        return score_matrix[xs, ys]


def load_data():
    data = [(1, 2, 4.5), (1, 3, 2.0),
            (2, 1, 4.0), (2, 3, 3.5),
            (3, 2, 5.0), (3, 4, 2.0),
            (4, 2, 3.5), (4, 3, 4.0), (4, 4, 1.0)]
    train_fea = [(x[0] - 1, x[1] - 1) for x in data]
    target = [x[2] for x in data]

    return train_fea, target


def train(model, loss_fn, optimizer):
    features, targets = load_data()
    y = model.forward(features)
    out = loss_fn(y, torch.Tensor(targets))

    # Backpropagation
    optimizer.zero_grad()
    out.backward()
    # print(core.user_matrix.grad)
    optimizer.step()
    return out


def cos_sim(ves):
    a = torch.matmul(ves, torch.transpose(ves, 0, 1))
    ves_norm = ves.norm(dim=1, keepdim=True)
    # print(ves_norm.shape)
    norm_div = torch.matmul(ves_norm, torch.transpose(ves_norm, 0, 1)) + 1e-6
    return a / norm_div
