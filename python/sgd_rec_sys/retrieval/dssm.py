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

__all__ = ['DSSM', 'DefaultItemTower', 'DefaultUserTower', 
           'TripletHingeLoss', 'TripletLogisticLoss',
           'CrossEntropyLoss']

# 注意
# 特征预处理（只区分为两类）
    # dense fea（组成一组向量，然后使用低阶的全连接层）
    # emb fea(类别id号)

class DefaultItemTower(nn.Module):
    '''
        物品塔模型确保可以处理
            item_emb [B,N,K_item] 正样本N=0，其他样本N=1,2,3...
        保证用户塔和物品塔输出emb维度一致
    '''
    def __init__(self, in_dim, hidden_dims, activation_fun=nn.ReLU()) -> None:
        super(DefaultItemTower, self).__init__()
        
        self.out_emb_dim = hidden_dims[-1]
        
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
            layers.append(activation_fun)
            in_dim = h_dim
        self.nns =  nn.Sequential(*layers)

    def forward(self, x):
        # x: item_emb [B,N,K_item]
        B,N,K = x.shape
        h = self.nns(x.reshape(B*N, K)).reshape(B, N, self.out_emb_dim) # h [B, N, out_dim]
        return h

class DefaultUserTower(nn.Module):
    '''
        用户塔模型确保可以处理
            user_emb [B, K_user]
        保证用户塔和物品塔输出emb维度一致
    '''
    def __init__(self, in_dim, hidden_dims, activation_fun=nn.ReLU()) -> None:
        super(DefaultUserTower, self).__init__()
        
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
            layers.append(activation_fun)
            in_dim = h_dim
        self.nns =  nn.Sequential(*layers)

    def forward(self, x):
        # x: user_emb [B, K_user]
        return self.nns(x)
    

class DSSM(nn.Module):
    '''
        物品塔模型确保可以处理
            item_emb [B,N,K_item] 正样本N=0，其他样本N=1,2,3...
        用户塔模型确保可以处理
            user_emb [B, K_user]
        assert 用户塔和物品塔输出emb维度一致
        
        特别注意：输入的fake data是emb，所以模型不会对编码层做参数优化。
    '''
    def __init__(self, item_tower, user_tower) -> None:
        super(DSSM, self).__init__()
        self.item_tower = item_tower
        self.user_tower = user_tower

    def forward(self, x):
        # user_emb [B, K_user]
        # item_emb [B,N,K_item] 正样本N=0，其他样本N=1,2,3...
        user_emb, item_emb = x
        h_user = self.user_tower(user_emb) # [B, K_out]
        h_item = self.item_tower(item_emb) # [B, N, K_out]
        
        # 计算余弦相似度
        cos_sim = torch.einsum('bi,bji->bj', h_user, h_item) # 确认一下输出 [B, N]
        return cos_sim


# TODO notebook 可视化分析
class TripletHingeLoss(nn.Module):
    def __init__(self, m):
        super(TripletHingeLoss, self).__init__()
        self.m = m

    def forward(self, pos_cosin, neg_cosin):
        # pos_cosin [N], neg_cosin [N]
        assert pos_cosin.shape == neg_cosin.shape
        N = pos_cosin.shape[0]
                
        N_loss = torch.clamp(self.m - (pos_cosin-neg_cosin), min=0)
        loss = torch.sum(N_loss)/N
        return loss

# TODO notebook 可视化分析
class TripletLogisticLoss(nn.Module):
    def __init__(self, sigma):
        super(TripletLogisticLoss, self).__init__()
        self.sigma = sigma

    def forward(self, pos_cosin, neg_cosin):
        # pos_cosin [N], neg_cosin [N]
        assert pos_cosin.shape == neg_cosin.shape
        N = pos_cosin.shape[0]
                
        N_loss = torch.log(1 + torch.exp(self.sigma*(pos_cosin-neg_cosin)))
        loss = torch.sum(N_loss)/N
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x):
        # x:[B, N]
        N = x.shape[0]
        N_loss= -1*torch.log(torch.softmax(x, dim=1)[:,0]) # 第一列为正样本
        loss = torch.sum(N_loss)/N
        return loss