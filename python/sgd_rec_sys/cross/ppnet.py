import torch
from torch import nn
from ..emb import EmbeddingLayer

__all__ = ['PPNet']

class PPNet(nn.Module):
    '''
    宏观思路
        conditional network
            原应用为LHUC，基于个人的语音生成
            control_net 使用了类似的操作
    
    原理图示见
        ./doc/cross/ppnet.md
    
    实现细节
        - 只处理离散特征
            - 特征编码维度可以不一致
        - 左右塔输出的label和原blog略有不同
            - 拼接输出并用last layer做ctr预估

        - [暂未实现] gate NN 接受左半部emb输入，但不计算路径上的反向传播（加速emb学习）
    '''
    def __init__(self,
                 tower_hidden_dims, # 保证左右塔相同，右塔单独做condition操作
                 one_hot_fea_list,
                 one_hot_fea_emb_dim,
                 multi_hot_fea_list,    # 每个特征的字典容量
                 multi_hot_fea_emb_dim, # 每个特征的编码维度
                 ) -> None:
        super().__init__()
        self.emb_layer = EmbeddingLayer(one_hot_fea_list,
                                        one_hot_fea_emb_dim,
                                        multi_hot_fea_list,
                                        multi_hot_fea_emb_dim)
        # 前三个特征为condition特征 uid pid aid
        # 分别表示user id，photo id，author id
        left_emb_dim = sum(one_hot_fea_emb_dim[3:]) + sum(multi_hot_fea_emb_dim)
        cond_emb_dim = sum(one_hot_fea_emb_dim[:3]) 
        
        self.left_tower = DeepNet(left_emb_dim, tower_hidden_dims, activation_fun=nn.ReLU())
        self.cond_tower = ConditionNet(left_emb_dim, cond_emb_dim, tower_hidden_dims, activation_fun=nn.ReLU())
        
        # last 分类layer
        self.prdict_layer = nn.Linear(tower_hidden_dims[-1]*2, 1) # ctr预估，只有一个输出概率
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        one_hot_x, multi_hot_x = X
        emb_raw = self.emb_layer(one_hot_x, multi_hot_x) # [B,F,K] batch,field,Emb
        B = emb_raw.shape[0]

        emb_cond = emb_raw[:,:3,:].reshape(B,-1) # 默认one hot 前三个位置为conditional emb
        emb_left = emb_raw[:,3:,:].reshape(B,-1)
        
        # two tower
        left_out = self.left_tower(emb_left)
        cond_out = self.cond_tower(emb_left, emb_cond)
        
        assert left_out.shape == cond_out.shape

        # predict out
        h_comb = torch.cat((left_out, cond_out), dim=1)
        out = self.sigmoid(self.prdict_layer(h_comb))
        return out


def DeepNet(in_dim, hidden_dims, activation_fun=nn.ReLU()):
    layers = []
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
        layers.append(activation_fun)
        in_dim = h_dim
    return nn.Sequential(*layers)


class ConditionNet(nn.Module):
    def __init__(self, emb_dim, cond_emb_dim, hidden_dims, activation_fun=nn.ReLU()):
        super().__init__()
        gate_in_dim = emb_dim + cond_emb_dim
        self.layer_num = len(hidden_dims)
        self.activ_fn = activation_fun
        self.linear_list = nn.ModuleList()
        self.gate_list = nn.ModuleList()
        
        in_dim = emb_dim
        for h_dim in hidden_dims:
            self.linear_list.append(nn.Linear(in_dim, h_dim))
            # TODO GateNN的模型容量需要调节hidden dim
            self.gate_list.append(GateNN(in_dim=gate_in_dim,
                                        hidden_dim=max(2, int(gate_in_dim/2))))
            in_dim = h_dim

    def forward(self, emb, emb_cond):
        h = emb
        gate_in = torch.cat((emb, emb_cond), dim=1)
        for i in range(self.layer_num):
            # emb_with_no_grad = emb.detach() # 这个细节有点费劲 TODO 要不暂时搁置
            h = self.activ_fn(
                    self.linear_list[i](
                        h * self.gate_list[i](gate_in)))
        return h
        

class GateNN(nn.Module):
    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()
        
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, 1, bias=True)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.linear1(x))
        out = 2 * self.sigmoid(self.linear2(h1))  # 取值范围[0,2], 默认值在1出
        return out  # TODO 类似残差的优化可以做，保证初始condition不变（和control_net类似的想法）