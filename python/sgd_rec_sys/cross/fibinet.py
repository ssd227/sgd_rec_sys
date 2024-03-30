import torch
from torch import nn
from ..emb import EmbeddingLayer

__all__ = ['FiBiNet', 'SENet', 'BilinearInteractionLayer']

class FiBiNet(nn.Module):
    '''
    输入都是field特征(编码为K维)+一阶双线性交互
    senet 的Embedding向量维度可以不同, sum pooling可以处理可变长度
    
    
    与原论文的实现diff:
        1、双线性cross层, vi@W·vj的计算结果为[B,F,F,K], 只保留了[F,F]部分的下三角矩阵
            - 保留了vi@W·vi 自我交互的情况
            - 由于保留的是下三角矩阵(包含对角线元素)，[F,F]部分保留了 1+2+...+F = (1+F)*F/2
        2、在实现c.Field-Interaction Type时, 考虑到矩阵并行的高效性，实际Wij参数并不是n=f(f-1)/2
            - 实际使用的Wij=[F,F,K,K], 参数量为 (F*K)^2
            - 深度怀疑原论文里n=f(f-1)/2排除了Wii自我交互的因素
            
    与王树森讲义的实现diff:
        讲义中conbination layer融合了四项:
            1、离散特征 -> emb -> senet -> bilinear (已实现)
            2、离散特征 -> emb -> bilinear          (已实现)
            3、离散特征 -> emb                      (未添加)
            4、连续特征                             (未添加)
        
        
    TODO
    Q:是不是连续特征都离散化了，论文图示没有连续特征
        - 在Deep net处拼接上dense feature
        - 或者，对real-value data预处理时提前离散化(ctr相关数据多呈现指数长尾)
    Q:dropout、batchnorm等加速运算的优化都没有添加
        - 目前保证代码最大易读性, 贴合原论文框架 
        - 在真实大数据上测试模型时，这些辅助函数也容易添加
    
    '''
    def __init__(self,
                 senet_r,
                 fix_emb_dim, # fibinet所有特征维度固定，统一为设为K
                 deepnet_hidden_dims,
                 one_hot_fea_list,
                 multi_hot_fea_list,
                 ) -> None:
        super().__init__()
        self.emb_layer = EmbeddingLayer(one_hot_fea_list,
                                        fix_emb_dim,
                                        multi_hot_fea_list,
                                        fix_emb_dim)

        fields_num = len(one_hot_fea_list) + len(multi_hot_fea_list) # 离散特征数目
        F, K = fields_num, fix_emb_dim
        fields_dim = int(F*(1+F)/2 * K) #离散特征cross后,总编码维度
        
        self.senet = SENet(fields_num, senet_r)
        self.cross_layer = BilinearInteractionLayer(fields_num, fix_emb_dim, w_type='field-all')
        self.deep_net = DeepNet(fields_dim*2, deepnet_hidden_dims, activation_fun=nn.ReLU())
        
        # 分类layer
        self.linear = nn.Linear(deepnet_hidden_dims[-1], 1) # ctr预估，只有一个输出概率
        self.activation = nn.Sigmoid()
        
    def forward(self, X):
        
        # Shallow Part
        one_hot_x, multi_hot_x = X
        emb_x = self.emb_layer(one_hot_x, multi_hot_x) # 按照field [B,Field_i, emb_K]

        # senet
        senet_emb_x = self.senet(emb_x)
        
        # Bilinear Interaction Layer
        cross_p = self.cross_layer(emb_x)       # [B,F*(1+F)/2,K]
        cross_q = self.cross_layer(senet_emb_x) # [B,F*(1+F)/2,K]
        assert cross_p.shape == cross_q.shape
        B = cross_p.shape[0]
        
        # conbination layer
        h_cross_out = torch.cat([cross_p.reshape(B, -1), cross_q.reshape(B,-1)], dim=1) # [B, 2n] n=(1+F)*F/2
        
        # Deep Part
        h_deep = self.deep_net(h_cross_out)
        out = self.activation(self.linear(h_deep)) # LR
        
        return out

def DeepNet(in_dim, hidden_dims, activation_fun=nn.ReLU()):
    layers = []
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
        layers.append(activation_fun)
        in_dim = h_dim
    return nn.Sequential(*layers)


class SENet(nn.Module):
    def __init__(self,
                 fields_num,
                 reduction_ratio,
                 activation_fn1=nn.Sigmoid(),
                 activation_fn2=nn.Sigmoid()) -> None:
        super().__init__()
        
        F = fields_num
        R = reduction_ratio

        # Fex 双层线性映射不带bias，先压缩，后还原。类似auto encoder 
        reduction_dim = max(1, int(F/R))
        self.fex_w1 = nn.Linear(F, reduction_dim, bias=False)
        self.fex_w2 = nn.Linear(reduction_dim, F, bias=False)
        
        self.act1 = activation_fn1
        self.act2 = activation_fn2

    def forward(self, e):
        h_fsq = torch.sum(e, dim=2)   # e: [B, F, K], h_fsq: [B, F]
        h_fex = self.act2(self.fex_w2(
                        self.act1(self.fex_w1(h_fsq)))) # h_fex: [B, F]
        out = e * h_fex.unsqueeze(2) # broadcast *, [B,F,K]*[B,F,1] => [B,F,K]
        return out

class BilinearInteractionLayer(nn.Module):
    def __init__(self, fields_num, emb_dim, w_type='field-all') -> None:
        super().__init__()
        self.w_type = w_type
        
        F,K = fields_num, emb_dim
        
        if w_type == 'field-all':
            self.W = nn.Parameter(torch.zeros((K, K), dtype=torch.float32))
        elif w_type == 'field-each':
            self.W = nn.Parameter(torch.zeros((F, K, K), dtype=torch.float32))
        elif w_type == 'field-intercation': # 这个维度有点复杂
            self.W = nn.Parameter(torch.zeros((F, F, K, K), dtype=torch.float32))
        else:
            assert False, "wrong input of w_type"
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, e):
        # 矩阵处理方式保证并行计算的效率，相关trick示范见apps/cross/fibinet.ipynb末尾
        
        B,F,K = e.shape
        if self.w_type == 'field-all':
            # e[B, F, K] @ W [K, K] => [B, F, K]  
            viw = e@self.W   # [B, F, K]
            
            # 将矩阵 viW 和 vj 分别增加一个维度，然后执行元素乘积
            viw = viw.unsqueeze(2).permute((0,3,1,2))   # [B,K,F,1]
            vj = e.unsqueeze(1).permute((0,3,1,2))      # [B,K,1,F]
            out = (viw@vj).permute((0,2,3,1))           # [B,K,F,F]=>[B,F,F,K]
            # pij is item in [F, F] with emb dim K

        elif self.w_type == 'field-each':
            # e[B,F,1,K] @ W [F,K,K] => [B, F, 1, K]
            viw = e.reshape(B,F,1,K) @ self.W
            
            viw = viw.permute((0,3,1,2))            # [B,F,1,K]=>[B,K,F,1]
            vj = e.unsqueeze(1).permute((0,3,1,2))  # [B,K,1,F]
            out = (viw@vj).permute((0,2,3,1))       # @ =>[B,K,F,F]=>[B,F,F,K]
            # pij is item in [F, F] with emb dim K
            
        elif self.w_type == 'field-intercation':
            '''
            TODO 思考有点费劲，但是仿照上述操作，应该能确保正确
                    todo 可以用循环实现和矩阵实现跑数据验证正确性
                
            e[B,F,K] @ W [F,F,K,K] =>
            e[B,F,1,K] @ W [F,K,K*F] =>  先交换轴，后合并
            => [B, F, 1, K*F]
            => [B, F, F, K]
            '''
            vi = e.reshape(B,F,1,K)                         # [B,Fi,1,K]
            w = self.W.permute(0,2,3,1).reshape(F,K,K*F)    # [Fi,Fj,K,K] => [Fi,K,K*Fj]
            viw = (vi@w).reshape(B,F,F,K)                   # [B,Fi,1,K*Fj] => [B,Fi,Fj,K]
            
            vj = e.unsqueeze(1)                             # [B,Fj,K] => [B,1,Fj,K]
            out = (viw*vj).permute((0,2,3,1))               # [B,Fi,Fj,K]*[B,1,Fj,K]=>[B,Fi,Fj,K]
            # pij is item in [Fi, Fj] with emb dim K(自动broadcast)
        else:
            assert False, "wrong input of w_type"

        # 生成下三角部分的逻辑掩码
        mask = torch.tril(torch.ones(F, F, dtype=torch.bool)).reshape(-1)
        return out.reshape(B,F*F,K)[:,mask,:] # field cross out [B, (1+f)*f/2, K] 
            


        