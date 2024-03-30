
"""
未参考论文，个人理解版本
    大致遵循王树森推荐系统课件

-----------------------------------------
FM (因子分解机)

原理
    * POLY2 模型是手动2阶组合特征,然后进行LR预测。缺点就是模型膨胀,计算量复杂。
        类比于协同过滤中的二阶矩阵

    * FM模型为了减少模型特征参数,同时又希望又特征上的二阶交互,将参数拆分。
        类比于MF模型中用 Vu Vi 两个隐藏变量去拆分原始矩阵的做法。

使用场景
    能使用LR的地方皆可以使用FM
    
实现细节
    fm实现略脱离实际，基于emb的可用实现见deepfm.py
    
    假设
        连续特性bitwise
        类别特征onehot表示后bitwise
        
        所有特征拼接后是bitwise
        FM处理的特征
    
    二阶项 (wj1·Wj2)*xj1*xj2
        - 注意二阶交互不包含自身交互
            感觉不合理,也印证了fibinet里的f*(f-1)/2的个数
            本实现利用mask考虑下三角矩阵
        - 二阶项变换后得到 (wj1*xj1) · (wj2*xj2)
            便于矩阵批量化处理
        - 二阶项实现时使用了FF展开操作
            - 总work偏大
            - 内存占用考虑到底层实现，应该不会全展开
            - TODO 优化点: mask掉x_ji为0的项目, 剩下的子矩阵坐内积
                - 工程问题，以后再说

注意：
    "算力不足时代的产物。工业界召回、排序都不用了。"    --王树森
"""

import torch
from torch import nn

__all__ = ['FM',]

# 二阶项,使用了矩阵批处理的trick
class Order2(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(Order2, self).__init__()
        self.W = nn.Parameter(torch.empty((in_dim, emb_dim)))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        B = x.shape[0]
        F, K = self.W.shape
        
        # x:[B,F] => x:[B,F,1] , W:[F,K], x*W => wfxf:[B,F,K]     
        WX = x.unsqueeze(2) * self.W # [B,F,K] 按照filed  broadcast元素乘
        
        # 计算W·X每一行filed向量的内积交互, [B,F,K] op [B,F,K]=>[B,F,F]
        out = WX @ WX.reshape(B, K, F) # W·X=[field_1, field_2, ..., field_F], 内积后得到:[B,F,F]

        # 生成下三角部分的逻辑掩码
        mask = torch.tril(torch.ones(F, F, dtype=torch.bool)).reshape(-1)
        return out.reshape(B, F*F)[:,mask] # field cross out [B, (1+f)*f/2,]
        # 注意 out保留了下三角矩阵，原论文不包括 field_i自身的内积


class FM(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(FM, self).__init__()
        self.linear = nn.Linear(in_dim,1,bias=True)
        self.cross = Order2(in_dim, emb_dim)

        # TODO p1+p2可以用系数调节两者的值
        
    def forward(self, x):
        # order 1
        p1 = self.linear(x)
        # order 2
        p2 = torch.sum(self.cross(x),dim=1, keepdim=True) # reduce by line
        assert p1.shape == p2.shape
        return nn.Sigmoid()(p1+p2) # 暂定为ctr预估任务 

