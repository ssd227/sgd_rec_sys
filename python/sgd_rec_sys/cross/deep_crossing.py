import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F

# deep crossing 模型


class ResNet(nn.Module):
    def __init__(self, input_len, hidden_len):
        super().__init__()
        self.linear1 = Linear(input_len, hidden_len)
        self.linear2 = Linear(hidden_len, input_len)

    def forward(self, x):
        h1 = nn.ReLU()(self.linear1(x))
        h2 = self.linear2(h1)
        return nn.ReLU()(x+h2)


class Linear(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_len, output_len, requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(1, output_len, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.W.data)

    def forward(self, x):
        return x.matmul(self.W) + self.bias

class DeepCrossing(nn.Module):
    def __init__(self, field_num_list, output_len, embedding_len=2):
        super(DeepCrossing, self).__init__()
        self.field_num_list = field_num_list
        
        self.embed_fea_len = 0
        self.value_fea_len = 0
        for x in field_num_list:
            if x > 1:
                self.embed_fea_len += x
            else:
                self.value_fea_len += 1

        self.embedding = nn.Embedding(self.embed_fea_len, embedding_len)

        self.fea_len = self.embed_fea_len * embedding_len + self.value_fea_len
        
        self.resnet1 = ResNet(self.fea_len, self.fea_len//2)
        self.resnet2 = ResNet(self.fea_len, self.fea_len//2)
        self.lr = Linear(self.fea_len, 1)


    def forward(self, x_one_hot, x_value):
        # embedding and stack
        # 对稀疏特征，做embedding。 
        # 这里又两个特征，分别为 3维、4 维 one hot. 编码后统一降为2维的fea
        
        # print('\n\n'+'*******debug log********'+'\n\n')

        # ont hot features need embedding
        x_one_hot_emb = self.embedding(x_one_hot) # output N* sum(one hot len) * 2
        x_p1_reshape = torch.reshape(x_one_hot_emb, (x_one_hot.shape[0], -1))
        # value features no need processed
        h1 = torch.cat((x_p1_reshape, x_value), 1)

        # resnet unit
        h2 = self.resnet1(h1)
        h3 = self.resnet2(h2)

        # scoring
        h4 = self.lr(h3)
        return nn.Sigmoid()(h4)

def load_data():
    # 三个特征，前两个转成 one hot特征，并记录field_num. 最后一个为数值特征
    x = [[0,0,5.0],
        [1,1,1.0],
        [2,3,7.0]]

    y_ = [0.99, 0.12, 0.5]

    field_num = [3, 4, 1]
    # transform x to one hot
    x_ont_hot = []
    x_value =[]
    for a in x:
        tmp1 = []
        tmp2 = []
        for i in range(len(field_num)):
            if field_num[i] >1:
                one_hot = [0]*field_num[i]
                one_hot[a[i]] = 1
                tmp1 += one_hot
            else:
                tmp2.append(a[i])
        x_ont_hot.append(tmp1)
        x_value.append(tmp2)
    
    return torch.LongTensor(x_ont_hot), torch.Tensor(x_value), torch.Tensor(y_), field_num


def train():
    # data
    x_one_hot, x_value, y_, field_num_list = load_data()

    # core
    model = DeepCrossing(field_num_list, output_len=1)

    # print_all_model_parameters(core.lr)

    for epoch in range(1000):
        # 老实说sgd优化器对简单数据效果最好
        # optimizer = torch.optim.SGD(core.parameters(), lr=1e-1 * epoch / 100, weight_decay=0.1)
        optimizer = torch.optim.RMSprop(model.parameters(), weight_decay=0.1)
        # optimizer = torch.optim.Adam(core.parameters(), lr=1e-1 * epoch / 1000, weight_decay=0.1)

        y = model.forward(x_one_hot, x_value)
        loss = torch.sum(y_ * y + (1-y_) * (1-y))/len(y_)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch:', epoch, 'loss:', loss)


def print_all_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

if __name__ == '__main__':
    # load_data()
    train()
