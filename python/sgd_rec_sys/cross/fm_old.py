# !!! 两年后重写，此文档暂时作废
# TODO 待删除

"""
FM (因子分解机)

原理：
    * POLY2 模型是手动2阶组合特征,然后进行LR预测。缺点就是模型膨胀,计算量复杂。
        类比于协同过滤中的二阶矩阵

    * FM模型为了减少模型特征参数,同时又希望又特征上的二阶交互,将参数拆分。
        类比于MF模型中用 Vu Vi 两个隐藏变量去拆分原始矩阵的做法。

使用场景：
    能使用LR的地方皆可以使用FM
    
注意：
    "算力不足时代的产物。工业界召回、排序都不用了。"    --王树森
"""

# todo 下面的实现不好理解，而且还不一定是对的
# 参数数量上也对不上号。

import torch
from torch import nn

# 二阶项,使用了矩阵批处理的trick
class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super(FactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) **2
        sum_of_square = torch.sum(x**2, dim = 1) 
        
        '''
        思考：
        # 不减去就当成一个纯二阶项目,好像问题也不大。
        那实际上fm的意义就是把二阶项的参数W提前作用到x上, 参数数量上就少了很多
        二阶的意义就是merge, 然后开方。使用参数mu来调节 x项 和 x**2项上的量纲问题。(但是, 为什么不直接开方得了)
        '''
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FM(nn.Module):
    def __init__(self, input_len, output_len, hidden_len=4):
        super(FM, self).__init__()
        # for linear features
        self.W1 = nn.Parameter(torch.randn(input_len, output_len) * 1e-1)
        # for second order features
        self.W2 = nn.Parameter(torch.randn(1, hidden_len))
        self.mu = nn.Parameter(torch.randn(1, 1)*0.1)
        
        # init parameter
        nn.init.xavier_uniform_(self.W1.data)
        nn.init.xavier_uniform_(self.W2.data)
        
        # some functions 
        self.fm = FactorizationMachine()
        self.sigmod = nn.Sigmoid()
        
    def forward(self, x):
        # linear item
        part1 = x @ self.W1
        # cross item
        part2 = self.fm(x.reshape((-1, x.shape[1], 1))*self.W2) # TODO 这玩意写的也很丑
        return self.sigmod(part1 + self.mu * part2)

