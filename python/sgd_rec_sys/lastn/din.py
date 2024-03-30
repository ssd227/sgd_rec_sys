import torch
from torch import nn

__all__ = ['DIN']

class DIN(nn.Module):
    '''
        实现思路与细节
            - ⽤加权平均代替平均(attention), 权重是候选物品与⽤户LastN物品的相似度。
                - 这里的加权weight不考虑一批数据的softmax
            
            - 只实现merge机制，不考虑完整ctr框架下的模型训练细节
                - merge后的特征作为用户特征的一部分输入ctr预估模型
            
            - 由于需要候选物品特征作为key参与attention交互，所以不适用于双塔召回
            
        个人想法：
            总觉得Dice的激活曲线完全可以用batchnorm取代
            计算效率问题，DIN原文里的逐个物品交互效率太低了，不如直接使用Transformer
            PReLU的图示就是极端版本的sigmoid(不会出现梯度问题吗？todo)
            
        
        注意：由于candidate Ad 和 Good i的交互是外积，允许编码维度不一致
    '''
    
    def __init__(self,
                 fea_dim, # 单个输入特征的维度
                 hidden_dim) -> None:
        super().__init__()
        self.actunit = ActUnit(fea_dim, hidden_dim)
    
    def forward(self, x):
        candidate_ad, goods = x # [B,K] [B,N,K]
        B,N,K = goods.shape
        
        cache = [] # item[B, K]
        for i in range(N):  # TODO 使用循环计算每个Good, 效率有点底下，不如Transformer
            weight_i = self.actunit(candidate_ad, goods[:,i,:].reshape(B,K))
            goodi_weight_emb = goods[:,i,:].reshape(B,K) * weight_i
            cache.append(goodi_weight_emb)
        
        # sum pooling
        wembs = torch.stack(cache, dim=1) # [B,N,K]
        out = torch.sum(wembs, dim=1) # [B,K]
        return out
        

class ActUnit(nn.Module):
    def __init__(self,
                 fea_dim, 
                 hidden_dim=36) -> None:
        super().__init__()
        # TODO PReLu/Dice(32)原论文细节未实现
        # TODO 外积是不是把特征扩张的太大了
        in_dim = int(fea_dim*(fea_dim+2)) # out_prod + k_dim, q_dim 
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1) # 输出单值作为权重
        self.sigmoid = nn.Sigmoid() # todo 原文使用的是PReLU和Dice
    
    def forward(self, k, q):
        assert k.shape == q.shape
        B = k.shape[0]
        out_product = torch.einsum('bi,bj->bij', k, q).reshape(B, -1)
        h1 = torch.cat((k,q,out_product), dim=1)
        out_weight = self.linear2(self.sigmoid(self.linear1(h1)))
        return out_weight
        




