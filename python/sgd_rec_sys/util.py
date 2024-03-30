import torch

def print_all_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
            
            
# 打印模型每行输入输出还是很方便的
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
print(layer.__class__.__name__,'output shape: \t',X.shape)