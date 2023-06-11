"""
DSSM Deep Structured Semantic Models 双塔模型

原理：
    * 物品塔、用户塔分别做表征(特征预处理、语义提取)
    * 双塔的输出用
    

双塔模型的训练
• Pointwise:独⽴看待每个正样本、负样本，做简单的⼆元分类。
    • 把召回看做⼆元分类任务。
    • 对于正样本,⿎励cos(a,b+) 接近+1。
    • 对于负样本,⿎励cos(a,b-)接近-1。
    • 控制正负样本数量为1: 2或者1: 3。

• Pairwise:每次取⼀个正样本、⼀个负样本[1]。
    基本想法:⿎励cos(a,b+)⼤于cos(a,b-)
    Triplet hinge loss:
        • 如果cos(a, b+)⼤于cos(a,b-)+ 𝑚，则没有损失。
        • 否则,损失等于cos(a,b-) + 𝑚 - cos(a,b+) 。

• Listwise:每次取⼀个正样本、多个负样本[2]。

"""

import torch
import torch.nn as nn

# 注意
# 特征预处理（只区分为两类）
    # dense fea（组成一组向量，然后使用低阶的全连接层）
    # emb fea(类别id号)
    

class ItemNet():


class UserNet():


class SubNet(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    
    

class DSSM(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    
    

    def forward():
        # 目前只实现单机单卡训练
    
    
    
    def 

