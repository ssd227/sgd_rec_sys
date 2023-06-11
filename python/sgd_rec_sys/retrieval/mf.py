"""
矩阵补充/矩阵分解算法（过时的召回算法）

原理：
    1、计算物品emb 和用户emb
    2、使用dot计算 user_emb, item_emb见的内积
    3、内积结果用来拟合user给item的打分值

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
    def __init__(self, userN, itemN, dim =32):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(userN, dim)
        self.item_emb = nn.Embedding(itemN, dim)

    def forward(self, uids, iids):
        uemb = self.user_emb(uids)
        iemb = self.item_emb(iids)
        dot_score = torch.sum(uemb * iemb, dim=1)
        return dot_score