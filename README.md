# 推荐系统 (pytorch version)


## 1 __项目目标__
搭建一个完备的



---

## 2 **sgd_rec_sys主要内容**
* 王树森（王老师） [推荐系统系列课程](https://space.bilibili.com/1369507485/channel/collectiondetail?sid=615109)为主要系统框架
* 推荐方向最新论文的整合
* [cs329-实用机器学习](https://space.bilibili.com/1567748478/channel/collectiondetail?sid=28144)-李沐（沐导）中的一些工程实践
    
---

## 3 推荐相关论文汇总

[docs/paper.md](./docs/papers.md)

---
## 4 各模块细节

## 4.1 召回(retrieval)
```
主流思路：基于向量召回，模型输出的user_emb, item_emb存储在向量数据库中，使用最近邻查找快速从底层库中取出

```


## 4.2 排序(rank)
```


```

## 4.3 特征交叉(feature cross)
```

```

## 4.4 行为序列() 
```
```

## 4.5 重排(reorder)
```
基于物品属性标签的相似性度量（）

```

## 4.6 物品冷启动(cold start)
```

```

## 4.7 模型指标(metrics)
* accuracy、precision、recall、f1
* roc、auc
* 参考文档 [[docs/mertics]](./docs/metrics/index.md)

---
## 5 其他

### 5.1 文档自动生成生成
```
TODO:
    python-sphinex

```


5.2 相关术语
![推荐系统整体框架](./terminology.webp)





