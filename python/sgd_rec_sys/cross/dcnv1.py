import torch
from torch import nn

from ..emb import EmbeddingLayer

__all__ = ['DCN1',]

class DCN1(nn.Module):
    def __init__(self,
                 in_dim,
                 cross_order,
                 hidden_dims,
                 one_hot_fea_list,
                 one_hot_fea_emb_dim,
                 multi_hot_fea_list,
                 multi_hot_fea_emb_dim,
                ) -> None:
        super().__init__()
        self.emb_layer = EmbeddingLayer(one_hot_fea_list,
                                        one_hot_fea_emb_dim,
                                        multi_hot_fea_list,
                                        multi_hot_fea_emb_dim,
                                        cat_emb=True)
        self.cross_net = CrossNet(in_dim, cross_order)
        self.deep_net = DeepNet(in_dim, hidden_dims, activation_fun=nn.ReLU())
        
        last_layer_dim = in_dim + hidden_dims[-1]
        self.linear = nn.Linear(last_layer_dim, 1) # ctr预估，只有一个输出概率
        self.activation = nn.Sigmoid()
        
    def forward(self, X):
        dense_x, one_hot_x, multi_hot_x = X
        emb_x = self.emb_layer(one_hot_x, multi_hot_x)
        x0 = torch.cat((dense_x, emb_x), dim=1)

        h_cross = self.cross_net(x0)
        h_deep = self.deep_net(x0)
        h_concate = torch.cat((h_cross, h_deep), dim=1)
        out = self.activation(
            self.linear(h_concate))
        
        return out

def DeepNet(in_dim, hidden_dims, activation_fun=nn.ReLU()):
    layers = []
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_features=in_dim, out_features=h_dim, bias=True))
        layers.append(activation_fun)
        in_dim = h_dim
    return nn.Sequential(*layers)


class CrossNet(nn.Module):
    def __init__(self, in_dim, cross_order) -> None:
        super().__init__()
        self.layers = nn.ModuleList([CrossLayer(in_dim) for _ in range(cross_order)])
    
    def forward(self, x0):
        xi = self.layers[0](x0, x0)
        if len(self.layers)>1:
            for cross_layer in self.layers[1:]:
                xi = cross_layer(x0, xi)
        return xi           


class CrossLayer(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros((in_dim, 1), dtype=torch.float32)) # 1d 参数较少
        self.b = nn.Parameter(torch.zeros((1, in_dim), dtype=torch.float32))
        nn.init.xavier_uniform_(self.w)
    
    def forward(self, x0, xi):
        assert x0.shape == xi.shape
        x0_ = x0.unsqueeze(2)
        xi_ = xi.unsqueeze(2)
        x0xi = torch.bmm(x0_, xi_.swapaxes(1,2))
        xi_next = (x0xi@self.w).squeeze() + self.b + xi
        return xi_next