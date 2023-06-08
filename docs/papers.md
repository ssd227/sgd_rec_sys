# 推荐系统 相关论文

* Deep Neural Networks for YouTube Recommendations

        核心贡献：向量化召回 

        借鉴word2vec，将相似的user_emb和item_emb映射到相同的空间，达到聚类效果。使用向量检索快速返回topk候选样本

* Google-wide and deep
        感觉在水论文，实验数据没有说服力

* FM

* DeepFM

* deep and cross 
        号称可以设置任意次数的特征交叉cross？？

---
## 召回(recall)
基于向量召回，模型输出的user_emb, item_emb存储在向量数据库中，使用最近邻查找快速从底层库中取出


双塔模型训练

* Pointwise：独⽴看待每个正样本、负样本，做简单的⼆元分类。
* Pairwise：每次取⼀个正样本、⼀个负样本
  * Jui-Ting Huang et al. Embedding-based Retrieval in Facebook Search. InKDD, 2020. [[paper]()]
* Listwise：每次取⼀个正样本、多个负样本
  * Xinyang Yi et al. Sampling-Bias-Corrected Neural Modeling for Large Corpus Item
Recommendations. In RecSys, 2019. [[paper]()]


---
## 排序(rank)
---
## 特征交叉(feature cross)
---
## 行为序列()
---
## 重排

基于图文内容的物品向量表征
* [CLIP]-Learning transferable visual models from natural language
supervision. In ICML, 2021. [[paper](https://arxiv.org/abs/2103.00020)] [[code](https://github.com/openai/CLIP)] [[blog](https://openai.com/research/clip)]

漏斗多样性
* [MMR] 相关
* [DPP] Chen et al. Fast greedy map inference for determinantal point process to improve
recommendation diversity. In NIPS, 2018. [[paper](https://arxiv.org/pdf/1709.05135.pdf)]
---
## 物品冷启动
---
## 其他

---





