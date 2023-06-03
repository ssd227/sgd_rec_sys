# auc roc 简单实现
# 目前只支持 0、1 正负样本标签，可以是numpy or python原生格式

import numpy as np

def roc(y_hat, y):
    """
    输入:
        y_hat: 模型对每个样本的预测概率, 取值范围[0,1]
        y: 二分类任务的标签. 默认是0, 1
        
    roc 定义:
        y轴- # correct pos predictions / # pos examples
        x轴- # wrong pos predictions / # neg examples    
    """
    pos = 1
    neg = 0
    
    sample_points = []
    
    for i in range(0,101): # 通过100个点来刻画roc曲线
        theta = 0.01 * i
        xs = sum((y_hat >= theta) & (y == neg)) / sum(y==neg)
        ys = sum((y_hat >= theta) & (y == pos))  / sum(y==pos)
        sample_points.append((xs,ys))
    
    sample_points.sort(key=lambda x:x[0])
    
    assert len(sample_points) == 101
    
    return sample_points    
    
def auc(y_hat, y):
    """
    auc(计算roc曲线下的面积):度量模型对二分类的区分度,不能反映预测的值的准确程度
    输入:
        y_hat: 模型对每个样本的预测概率, 取值范围[0,1]
        y: 二分类任务的标签. 默认是0,1
        
    
    面积实现原理：
        简单的微分逼近
    """
    sample_points = roc(y_hat,y)
    area = 0
    for i in range(100):
        x1, y1 = sample_points[i]
        x2, y2 = sample_points[i+1]
        dx = x2-x1
        dy = y2-y1 
        area += dx*y1 + 0.5*dx*dy
    return area