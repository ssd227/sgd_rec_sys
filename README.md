# sgd_rec_sys V_0.1  (pytorch)

## 1 项目目标
搭建一个完整可用的推荐系统，模块化实现推荐各阶段常用模型和算法。

参考:
* 王树森[推荐系统系列课程](https://space.bilibili.com/1369507485/channel/collectiondetail?sid=615109)-王树森 为主要系统框架
* [cs329-实用机器学习](https://space.bilibili.com/1567748478/channel/collectiondetail?sid=28144)-李沐 一些工程实践
* 推荐方向实用论文

进度&问题：
  - 相关论文大致实现了一遍，忽略了不少细节。
  - 某些召回模型，lastn模型依赖的工程细节比较多，没有实现
    - 不太方便用fake数据整合，更适合单开一个项目维护
  - 框架需要抽象，还做不到组合即用。
    - 编码层需要统一抽象处理

目前更适合用作个人的文档参考，快速利用现有知识搭建新系统。ctr预估的corss部分倒是可以快速复用。

立项时把工程问题想的太简单了, 系统不是几个模型和算法就能够快速部署的。数据存储预处理、emblayer设计和整合、向量数据库服务、推荐服务链都需要大量的整合工作。


---

## 2 主要内容
| 文件路径   |      作用    |
|---------- |:-------------:|
| [./python/sgd_rec_sys](./python/sgd_rec_sys/)  |  模型、算法实现 | 
|[./apps](./apps/)      |模型、算法使用样例| 
|[./docs](./docs/)      |推荐各阶段技术说明|
|[./papers](./papers/)  | 论文review | 


## 3 实现进度
- retrieval
  - [X] Item CF [[code]](./python/sgd_rec_sys/retrieval/item_cf.py) [[notebook]](./apps/retrieval/item_cf.ipynb)
  
  - [X] User CF [[code]](./python/sgd_rec_sys/retrieval/user_cf.py) [[notebook]](./apps/retrieval/user_cf.ipynb)
  
  - [X] swing [[code]](./python/sgd_rec_sys/retrieval/swing.py) [[notebook]](./apps/retrieval/swing.ipynb)
  
  - [X] MF (矩阵分解) [[code]](./python/sgd_rec_sys/retrieval/mf.py) [[notebook]](./apps/retrieval/mf.ipynb)
  
  - DSSM (双塔模型) [[code]](./python/sgd_rec_sys/retrieval/dssm.py)
    - [ ] pointwise
    - [X] pairwise [[notebook]](./apps/retrieval/dssm.ipynb)
    - [X] listwise [[notebook]](./apps/retrieval/dssm.ipynb)
    - [ ] batch内负采样
  - [ ] 双塔模型+自监督学习
  - [ ] Deep Retrieval
  - [ ] 其他召回通道
    -  GeoHahs、同城召回、关注作者召回、有交互作者召回、相似作者召回、缓存召回

- filter
  - [X] 曝光过滤 & Bloom Filter [[code]](./python/sgd_rec_sys/filter/bloom_filter.py) [[notebook]](./apps/filter/bloomFilter.ipynb)
  
- rank
  - [X] 多目标建模 [[code]](./python/sgd_rec_sys/rank/multitask.py) [[notebook]](./apps/rank/multitask.ipynb)
  - [X] MMOE [[code]](./python/sgd_rec_sys/rank/mmoe.py) [[notebook]](./apps/rank/mmoe.ipynb)
  - [X] 融合预估分数 [[code]](./python/sgd_rec_sys/rank/merge_score.py)
  - [ ] Youtube 视频播放建模
  - [ ] 粗排三塔

- cross
  - FM
    - [X] FM [[code]](./python/sgd_rec_sys/cross/fm.py) [[notebook]](./apps/cross/fm.ipynb)
    - [X] DeepFM [[code]](./python/sgd_rec_sys/cross/deepfm.py) [[notebook]](./apps/cross/deepfm.ipynb)
    - [ ] xDeepFM (太丑了pass)

  - [ ] Deep and Wide
  
  - Deep & Cross
    - [X] DCN V1 [[code]](./python/sgd_rec_sys/cross/dcnv1.py) [[notebook]](./apps/cross/dcn_v1.ipynb)
    - [X] DCN V2 [[code]](./python/sgd_rec_sys/cross/dcnv2.py) [[notebook]](./apps/cross/dcn_v2.ipynb)
  
  - LHUC
    - [X] PPNet (快手) [[code]](./python/sgd_rec_sys/cross/ppnet.py) [[notebook]](./apps/cross/ppnet.ipynb)

  - field 交叉
    - [X] FiBinet [[code]](./python/sgd_rec_sys/cross/fibinet.py) [[notebook]](./apps/cross/fibinet.ipynb)
      - SENet + Bilinear Cross

- lastn
  - [X] DIN [[code]](./python/sgd_rec_sys/lastn/din.py) [[notebook]](./apps/lastn/din.ipynb)
  - [ ] SIM (工程细节偏多)
  
- rerank
  - [X] MMR [[code]](./python/sgd_rec_sys/rerank/mmr.py) [[notebook]](./apps/reorder/mmr.ipynb)
    - Candi->Saved 贪心找max unsimilar item
  - [X] DPP [[code]](./python/sgd_rec_sys/rerank/dpp.py) [[notebook]](./apps/reorder/dpp.ipynb)
    - det^2 衡量相似度
  - [ ] MGS
    - 施密特正交法找基向量，类DPP

- cold start
  - 空白

- metrics
  - [X] accuracy、precision、recall、f1 [[code]](./python/sgd_rec_sys/metrics/basic.py) [[notebook]](./apps/metrics/basic.ipynb)
  - [X] AUC_ROC [[code]](./python/sgd_rec_sys/metrics/auc_roc.py) [[notebook]](./apps/metrics/auc_roc.ipynb) [[doc]](./docs/metrics/index.md)
---
## 4 各模块细节
- [[docs content]](./docs/index.md)


---
## 5 相关论文

---
### 5.1 召回(recall)
基于向量召回，模型输出的user_emb, item_emb存储在向量数据库中，使用最近邻查找快速从底层库中取出

双塔模型、向量召回
```
Pointwise：独⽴看待每个正样本、负样本，做简单的⼆元分类。
Pairwise：每次取⼀个正样本、⼀个负样本
Listwise：每次取⼀个正样本、多个负样本
```

- [X] [`EBR`] Jui-Ting Huang et al. Embedding-based Retrieval in Facebook Search. In KDD, 2020. [[paper]](https://arxiv.org/abs/2006.11632) [[简评]](./papers/erb.md) [⭐️⭐️⭐️`经典必读`]

- [ ] [DSSM-Mircosoft] [[paper]]()


- [X] [`Batch内负采样`] Xinyang Yi et al. Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. In RecSys, 2019. [[paper]](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) [[简评]](./papers/batch_neg.md) [⭐️]


- [X] [`自监督学习`]Tiansheng Yao et al. Self-supervised Learning for Large-scale Item Recommendations.In CIKM, 2021. [[paper]](https://arxiv.org/abs/2007.12865) [[简评]](./papers/self-supervised.md) [⭐️⭐️]

deep retrieval
- [ ] [`Deep Retrieval`] Gao et al. Deep Retrieval: Learning A Retrievable Structure for Large-Scale Recommendations. In CIKM, 2021. [[paper]](https://arxiv.org/abs/2007.07203)
- [ ] [`TDM`] Han Zhu et al. Learning Tree-based Deep Model for Recommender Systems. In KDD, 2018. [[paper]]()


曝光过滤
- [ ] [`Bloom Filter`] Burton H. Bloom. Space/time trade-offs in hash coding with allowable
errors. Communications of the ACM, 1970.
[[paper]](https://sci-hub.et-fine.com/10.1145/362686.362692)

更复杂的模型 (召回占比小，但有效)
- [ ] [`PDN`] Li et al. Path-based Deep Network for Candidate Item Matching in Recommenders. In SIGIR, 2021. [[paper]]()
- [ ] [`SINE`] Tan et al. Sparse-interest network for sequential recommendation. In WSDM, 2021. [[paper]]()
- [ ] [`M2GRL`] Wang et al. M2GRL: A multitask multi-view graph representation learning framework for webscale
recommender systems. In KDD, 2020. [[paper]]()

---
### 5.2 排序(rank)

排序模型框架
- [X] [`wide&deep`] Heng-Tze Cheng, et al. Wide & Deep Learning for Recommender Systems. In DLRS, 2016. [[paper]](https://arxiv.org/abs/1606.07792) [[简评]](./papers/wide%26deep.md) [⭐️⭐️`经典工程`]

视频播放建模
- [X] [`Youtube Video`] Paul Covington, Jay Adams, & Emre Sargin. Deep Neural Networks for YouTube Recommendations. In RecSys, 2016. [[paper]](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf) [[简评]](./papers/youtube_video.md) [`经典`]


多目标模型预估值校准
- [ ] Xinran He et al. Practical lessons from predicting clicks on ads at Facebook. In the 8th
International Workshop on Data Mining for Online Advertising. [[paper]]()

多目标预估
```
- 基于基座输出的向量，同时预估点击率等多个⽬标。
- 改进1：增加新的预估⽬标，并把预估结果加⼊融合公式。
- 改进2：`MMoE`、`PLE`等结构可能有效，但往往无效。
- 改进3：`纠正position bias`可能有效，也可能无效。
```

- [ ] [`MMOE`] Jiaqi Ma et al. Modeling Task Relationships in Multi-task Learning with
Multi-gate Mixture-of-Experts. InKDD, 2018. [[paper]](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)

- [ ] [`PLE`] Tang et al. Progressive layered extraction (PLE): A novel multi-task learning (MTL)
model for personalized recommendations. In RecSys, 2020. [[paper]]()

- [ ] [`纠正position bias`] Zhe Zhao et al. Recommending What Video to Watch Next: A Multitask
Ranking System. In RecSys, 2019. [[paper]](https://daiwk.github.io/assets/youtube-multitask.pdf)

粗排三塔模型
- [ ] [`COLD`] Zhe Wang et al. COLD: Towards the Next Generation of Pre-Ranking System. In DLPKDD, 2020.[[paper]](https://arxiv.org/abs/2007.16122)
---

### 5.3 特征交叉(feature cross)

FM系列模型（过时）

- [X] [`FM`] Steffen Rendle. Factorization machines. In ICDM, 2010. [[paper]](https://sci-hub.yncjkj.com/10.1109/icdm.2010.127) [[简评]](./papers/fm.md) [⭐️⭐️⭐️`经典`]

- [X] [`DeepFM`]Huifeng Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. In cs.IR, 2017.[[paper]](https://arxiv.org/abs/1703.04247) [[简评]](./papers/deepfm.md) [⭐️]

- [X] [`xDeepFM`] Jianxun Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. In , 2018. [[paper]](https://arxiv.org/pdf/1803.05170.pdf) [[简评]](./papers/xdeepfm.md) [⭐️`丑陋`] 


Cross Network
- [Deep And Cross] 可设置任意次数的特征交叉
- [X] [`DCN V1`] Ruoxi Wang et al. Deep & Cross Network for Ad Click Predictions. In ADKDD, 2017. [[paper]](https://arxiv.org/abs/1708.05123) [[简评]](./papers/dcnv1.md) [⭐️⭐️⭐️]
  
- [X] [`DCN V2`] Ruoxi Wang, et al. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems. InWWW, 2021. [[paper]](https://arxiv.org/abs/2008.13535) [[简评](./papers/dcnv2.md)] [⭐️⭐️⭐️]
  
LHUC
- [X] [`PPNet`] Parameter Personalized Net-快⼿落地万亿参数推荐精排模型，2021。 [[Blog](https://ai.51cto.com/art/202102/644214.html)] [[简评]](./papers/ppnet.md) [⭐️⭐️]
  - [ ] [LHUC]Pawel Swietojanski, Jinyu Li, & Steve Renals. Learning hidden unit contributions for unsupervised acoustic model adaptation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2016. [[paper]]()

SENet & Bilinear Cross
- [X] [`FiBiNet`] Tongwen Huang, Zhiqi Zhang, and Junlin Zhang. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction. In RecSys, 2019. [[paper]](https://arxiv.org/abs/1905.09433) [[简评]](./papers/fibinet.md) [⭐️⭐️⭐️]
  - [ ] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-Excitation Networks. In CVPR, 2018. [[paper]]()

---

### 5.4行为序列(lastn)
- [X] [`DIN`] Guorui Zhou, et al. Deep Interest Network for Click-Through Rate Prediction. In KDD, 2018. [[paper]](https://arxiv.org/abs/1706.06978) [[简评]](./papers/din.md) [⭐️⭐️]

- [X] [`SIM`] Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction In CIKM, 2020. [[paper]](https://arxiv.org/abs/2006.05639) [[简评]](./papers/sim.md) [⭐️⭐️]

---

### 5.5 重排

基于图文内容的物品向量表征
- [ ] [`CLIP`] Learning transferable visual models from natural language
supervision. In ICML, 2021. [[paper]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/openai/CLIP) [[blog]](https://openai.com/research/clip)

漏斗多样性
- [X] [`MMR`] 相关
  - (来自检索算法) 从候选集C中，逐个贪心找出与当前S最不相似的item
- [X] [`DPP`] Chen et al. Fast greedy map inference for determinantal point process to improve recommendation diversity. In NIPS, 2018. [[paper]](https://arxiv.org/pdf/1709.05135.pdf) [[简评]](./papers/dpp.md) [⭐️⭐️⭐️]

---

### 5.6 Cold Start (物品冷启动)

---

### 5.7 A/B测试
- [ ] [Google] Tang et al. Overlapping experiment infrastructure: more, better, faster
experimentation. InKDD, 2010. [[paper]]()

---

### 5.8 其他论文汇总
* guyulongcs
  * [Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising](https://github.com/guyulongcs/Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising?tab=readme-ov-file#multi-modal)
* Doragd
  * [Algorithm-Practice-in-Industry](https://github.com/Doragd/Algorithm-Practice-in-Industry)
* wangzhe
  * [Reco-papers](https://github.com/wzhe06/Reco-papers)
  * [Ad-papers](https://github.com/wzhe06/Ad-papers)
  * [Real-Time Bidding](https://github.com/wzhe06/rtb-papers)

总体感觉推荐方向的论文写的都挺扯淡的。论证角度不是很高，工业细节也不够细。想法和贡献大多来自CV、NLP。


---
## 6 其他
### 6.1 相关术语

![推荐系统整体框架](./terminology.webp)

### 6.2 问题：
- 如何保证全链路的排序一致性？（疑问且重要）
  - 召回，粗排，精排
  - 粗排模型的设计方向有那些。

- 编码层，对于各个物品，用户emb
  - 那些编码是多阶段共用的
  - 每个emb在各阶段的训练中是如何更新+同步的
    - 感觉需要有一个全局的emb系统
    - 怎么去设计这样一个系统
      - 方便模型调用+训练

### 6.3 TODO
- fibnet 矩阵交叉计算的数值验证
  - 学了个新操作 `torch.einsum`
  - fibnet的复杂操作可以简化
- multi-hot的定长批处理实现的比较丑
- 提高mmoe模型的并发效率（参考RNN里四个门的计算过程）

- 文档自动生成生成
  - python-sphinex

---

注释符号:❌✅⭐️★☆⚡️ ❤️ ☀️ ☁️☔️ ☃️ ✈️⚽️⚓️⌛️☎️✉️✨☮️☯️ 

