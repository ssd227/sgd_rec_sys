# 推荐系统 相关论文


## 召回(recall)
基于向量召回，模型输出的user_emb, item_emb存储在向量数据库中，使用最近邻查找快速从底层库中取出

双塔模型、向量召回
```
Pointwise：独⽴看待每个正样本、负样本，做简单的⼆元分类。
Pairwise：每次取⼀个正样本、⼀个负样本
Listwise：每次取⼀个正样本、多个负样本
```

- [ ] [DSSM-Mircosoft] [[paper]()]

- [X] [经典] [`EBR`] Jui-Ting Huang et al. Embedding-based Retrieval in Facebook Search. In KDD, 2020. [[paper](https://arxiv.org/abs/2006.11632)]
  - EBR 基于语义表示emb vec的召回技术
    - Briefly, embedding-based retrieval is a technique to use
  embeddings to represent query and documents, and then convert
  the retrieval problem into a nearest neighbor (NN) search problem
  in the embedding space.
  - 正负样本的选取
    - 负样本：随机采样vs展现未点击（后者recall效果差）
    - 正样本：点击vs展现（效果差距不大，召回逼近rank结果）
  - ANN 近似最近邻查找
    - 粗粒度量化 Kmeans
      - IMI [11] and IVF [15] algorithms
    - 细粒度量化 product quantization
      - [ ] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization for Nearest Neighbor Search. IEEE Trans.
      - 局部敏感hash, MIPS等算法也可以做
        - [ ] Edith Cohen and David D. Lewis. [n.d.]. Approximating Matrix Multiplication for Pattern Recognition Tasks. In SODA 1997.
  - 后期优化点
    - 困难正、负样本挖掘
  - Embedding Ensemble-模型组合部署的一些优化点
   
  - [`评价`]
    - 文笔流畅，工程细节指导价值很高。多看几遍

- [X] Xinyang Yi et al. Sampling-Bias-Corrected Neural Modeling for Large Corpus Item
Recommendations. In RecSys, 2019. [[paper](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)]
  - 采用batch内负样本的方式训练，减少训练时的推理代价，但是引入了采样偏差。
  - 核心的采样偏差没有具体介绍，见bengio论文
    - [ ] Adaptive Importance Sampling to Accelerate Training of a Neural Probabilistic Language Model
  - 分布式（参数服务器）流数据的频率估计
  - 流处理，视频id改变，随着训练过程拟合最新的分布曲线
  - [`评价-疑问`]
    - 感觉写的一般，没完全理解。核心的纠偏效果也没有说明，论证近似的结果比原结果效果更好？
    - 纠偏sotfmax效果比plain效果更好？？提升了采样效率，还能提升效果？？

- [X] Tiansheng Yao et al. Self-supervised Learning for Large-scale Item Recommendations.In CIKM, 2021. [[paper](https://arxiv.org/abs/2007.12865)]
  - 通过对比学习，增强tail数据的表示，提升稀疏数据下的召回准确性
  - 数据增强通过CFM来实现
  - 优化目标融合：Loss_main + alpha(Loss_self_pred)
    - 自监督的数据分布和 main监督学习中的分布不同
  - [`评价`]
    - 比上述的纠偏论文要好点，主要背景知识交代的清楚，向量近似查找MIPS
    - 数据增强的细节还是没写清楚CFM，实验部分可以快速看一遍表格。

deep retrieval
- [ ] Weihao Gao et al. Learning A Retrievable Structure for Large-Scale Recommendations. In CIKM, 2021. [[paper]()]
- [ ] [`TDM`] Han Zhu et al. Learning Tree-based Deep Model for Recommender Systems. In KDD, 2018. [[paper]()]

曝光过滤
- [ ] [`Bloom Filter`] Burton H. Bloom. Space/time trade-offs in hash coding with allowable
errors. Communications of the ACM, 1970.
[[paper](https://sci-hub.et-fine.com/10.1145/362686.362692)]

更复杂的模型 (召回占比小，但有效)
- [ ] [`PDN`] Li et al. Path-based Deep Network for Candidate Item Matching in Recommenders. In SIGIR, 2021. [[paper]()]
- [ ] [Deep Retrieval] Gao et al. Learning an end-to-end structure for retrieval in large-scale recommendations. In CIKM, 2021. [[paper]()]
- [ ] [`SINE`] Tan et al. Sparse-interest network for sequential recommendation. In WSDM, 2021. [[paper]()]
- [ ] [`M2GRL`] Wang et al. M2GRL: A multitask multi-view graph representation learning framework for webscale
recommender systems. In KDD, 2020. [[paper]()]

---

## 排序(rank)

排序模型框架
- [X] [经典] [`wide&deep`] Heng-Tze Cheng, et al. Wide & Deep Learning for Recommender Systems. In DLRS, 2016. [[paper](https://arxiv.org/abs/1606.07792)]
  - 推荐系统相关的图示画的不错，推荐框架，pipline，特征处理都说的很清楚。
  - deep学习泛化特征+wide学习统计特征，就有点玄学了。
  - wide模型弥补deep的缺陷，只需要简单的一些特征交叉（业界不用了吧？？）
  - [`评价`]
    - 模型不太行，更像系统方向的论文。好在各个环节都介绍的很清楚。
    - 实验数据分析没有说服力

视频播放建模
- [X] [经典][Youtube] Paul Covington, Jay Adams, & Emre Sargin. Deep Neural Networks for YouTube Recommendations. In RecSys, 2016. [[paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)]
  - 核心贡献：向量化召回 
  - 借鉴word2vec，将相似的user_emb和item_emb映射到相同的空间，达到聚类效果。使用向量检索快速返回topk候选样本

多目标模型预估值校准
- [ ] Xinran He et al. Practical lessons from predicting clicks on ads at Facebook. In the 8th
International Workshop on Data Mining for Online Advertising. [[paper]()]

多目标预估
```
- 基于基座输出的向量，同时预估点击率等多个⽬标。
- 改进1：增加新的预估⽬标，并把预估结果加⼊融合公式。
- 改进2：`MMoE`、`PLE`等结构可能有效，但往往无效。
- 改进3：`纠正position bias`可能有效，也可能无效。
```

- [ ] [`MMOE`] Jiaqi Ma et al. Modeling Task Relationships in Multi-task Learning with
Multi-gate Mixture-of-Experts. InKDD, 2018. [[paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)]

- [ ] [`PLE`] Tang et al. Progressive layered extraction (PLE): A novel multi-task learning (MTL)
model for personalized recommendations. In RecSys, 2020. [[paper]()]

- [ ] [`纠正position bias`] Zhe Zhao et al. Recommending What Video to Watch Next: A Multitask
Ranking System. In RecSys, 2019. [[paper](https://daiwk.github.io/assets/youtube-multitask.pdf)]

粗排三塔模型
- [ ] [`COLD`] Zhe Wang et al. COLD: Towards the Next Generation of Pre-Ranking System. In DLPKDD, 2020.[[paper](https://arxiv.org/abs/2007.16122)]  [[paper](https://arxiv.org/abs/2007.16122)]
---

## 特征交叉(feature cross)

FM系列模型（低算力时代的产物，过时）

- [X] [FM] Steffen Rendle. Factorization machines. In ICDM, 2010. [[paper](https://sci-hub.yncjkj.com/10.1109/icdm.2010.127)]
  - 原理和实现都很简单，二阶特征交叉，通过分解系数矩阵降低参数量
  - 高效实现的等价思路：`TODO`

- [X] [DeepFM]
  - wide and deep 架构，把wide部分换成FM。
    - DeepFM automates the feature interaction learning in the wide component by adopting a FM model
  - [`评价`]
    - 写的挺难看的，甚至没有看懂
  
- [X] [xDeepFM]
  - 依旧是deep&wide老框架，cross部分提出CIN交叉网络
  - DCN的架构上，features interact at the vector-wise level
  - CIN的想法直接看Figure4猜就好了，计算代价高。
    - 外积升阶（编码维度D各自计算），
    - RNN的深度类比 vs 多项式的交叉级数
    - CNN卷积kernel类比 vs 新交叉层的特征维度
  - 总结，简单的想法就是通过外积升阶，然后做线性组合，通过kernel数量控制模型复杂度，最后通过sum pooling汇总各个特征（reduce by D）。
    - 每个cross维度，存在多个kernel特征(代表抽象出的多个兴趣)
    - corss维度逐层加深(表示在上一层K个维度上，进一步抽象出的新兴趣)
    - 模型的超参数调优是个问题（不如DCN V2来的直接）
  
  - [`评价`]
    - 模型太丑了，公式符号巨难看。这也能发KDD
    - 引入+背景介绍部分写的不如DCN v1、v2（有点啰嗦）
    - 下面这段论述就有点扯淡了，DCN v1的缺点是同层使用一个变换，使得模型容量较小，除了在多项式项前的系数组合上存在限制使得搜索空间较小，不影响模型对多项式的拟合。
      - The CrossNet can learn feature interactions very efficiently (the complexity is negligible compared with a DNN model), however the downsides are: 
        - (1) the output of CrossNet is limited in a special form, with each hidden layer is a scalar multiple of x0;
        - (2) interactions come in a bit-wise fashion.
    - CNN RNN都套上了，实现的真的很丑。2018年论文的局限性吗？

Cross Network
* [Deep And Cross] 可设置任意次数的特征交叉
- [X] [DCN V1] Ruoxi Wang et al. Deep & Cross Network for Ad Click Predictions. In ADKDD, 2017. [[paper](https://arxiv.org/abs/1708.05123)]
  - cross net 逼近高阶多项式近似，是FM的泛化，想法还挺有意思
  - 设x=(a,b,c) 直观理解是多项式用W_layer_i糅合后，构成完整的N阶多项式（包括每一项但是系数可调节），然后分别乘法操作(*a, *b, *c升阶)。next layer再用W_layer_i+1糅合
  - corss level：vector元素级别的交叉
  - [`评价`]
    - 行文和图示都挺好理解
    - 公式推导符号理解费劲，可用简单的例子辅助理解
  
- [X] [DCN V2] Ruoxi Wang et al. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems. InWWW, 2021. [[paper](https://arxiv.org/abs/2008.13535)]
  - V1在大数据量的工业数据上，cross部分的参数量不匹配deep，且W_l的参数共享限制使得模型特征corss的表达能力较弱。
  - cross思路：xl线性组合后,hamada乘升阶，补上低阶项(+xl)，bias拟合常数项，最终是x0*bl（也不是不行）
  - 使用MMOE和low-rank技术减小模型规模，trad-off between performance and efficiency
  - [`评价`]
    - wide&deep并行架构，相关工作将deepFM、DCN、xdeepFM串起来了。很好！
    - 疑问：low rank的思路就是奇异值分解，把奇异值低的正交基排除掉。但是模型直接上低秩模型也能训练的好吗？类似lora的fine tune？
    - 多项式逼近解释依旧跳过，感觉是为了数学而数学，公式死难看。
    - 实验部分写的挺好，充足的各种对比。
  
LHUC
- [X] [PPNet] Parameter Personalized Net-快⼿落地万亿参数推荐精排模型，2021。 [[Blog](https://ai.51cto.com/art/202102/644214.html)]
  - [ ] [LHUC]Pawel Swietojanski, Jinyu Li, & Steve Renals. Learning hidden unit contributions for unsupervised acoustic model adaptation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2016. [[paper]()]
  - 在语音识别领域中，LHUC算法(learning hidden unit contributions)核心思想是做说话人自适应(speaker adaptation)，在DNN网络中为每个说话人学习一个特定的隐式单位贡献(hidden unit contributions)，来提升不同说话人的语音识别效果。借鉴LHUC的思想，快手在精排模型上展开了尝试。PPNet于2019年上线后，显著提升了模型的CTR目标预估能力。

SENet & Bilinear Cross
- [X] [`FiBiNet`] Tongwen Huang, Zhiqi Zhang, and Junlin Zhang. FiBiNET: Combining Feature Importance
and Bilinear feature Interaction for Click-Through Rate Prediction. In RecSys, 2019. [[paper](https://arxiv.org/abs/1905.09433)]
  - [ ] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-Excitation Networks. In CVPR, 2018. [[paper]()]
  - SENET 类似attention机制，动态调整每个field的权重
  - 双线性交互, DCN v2结构和这篇类似
  - 基于field的emb，特征的编码维度固定，cross交互只有二阶部分
  - [`评价`]
    - 结构阐述清晰，比较好懂。
    - 实验分析一般，不同参数量的模型对比没有意义。
---

## 行为序列(lastn)
- [X] [DIN] Guorui Zhou, et al. Deep Interest Network for Click-Through Rate Prediction. In KDD, 2018. [[paper](https://arxiv.org/abs/1706.06978)]
  - 模型改进：对lastn做attention加权
  - 工程改进：模型训练trick 基于batch的norm + 基于输入分布的激活函数PReLU
  - L2正则项，实际上在optimizer里选择性操作是不是更容易实现。
  - 激活函数的改进，感觉用bathnorm是个更佳的选择
  - 没太看懂过滤掉低频商品，能避免过拟合。还是说L2正则只用在高频物品上。
    - Filter visited goods_id by occurrence frequency in samples and leave only the most frequent ones. In our setting,top 20 million good_ids are left
    - 还对比了其他过拟合方式 Regularization in DiFacto（没见过）
  - [`评价`]
    - 整体思路比较清晰，电商场景下可能有效。
    - 速览：快速看图，了解模型结构和核心思想
    - 工程实践可以借鉴，设计emb训练框架
    - 论文（2018）有时代背景的局限性，适度参考。

- [X] [SIM] Search-based User Interest Modeling with Lifelong Sequential
Behavior Data for Click-Through Rate Prediction In CIKM, 2020. [[paper](https://arxiv.org/abs/2006.05639)]
  - 居然还是阿里巴巴提出的。lastn方向做了不少工作。王树森说这两年业界应用的较多。
  - 看之前的想法
    - 为什么不给每个人显示维护一个兴趣分布，跟进用户的反馈，强硬+柔和的多种方式去调节这个用户的长期兴趣分布。强反馈类似于ui上的不感兴趣的button。软反馈类似于用户快速划走
      - 类似于贝叶斯分布，给定一个先验，然后用用户长期数据去调整后验
    - 电商兴趣建模还有个问题，就是用户买过后兴趣点丧失价值。比如组装电脑的用户
  - 提升还挺明显 （RPM是什么指标？）
    - Since 2019, SIM has been deployed in the display advertising system in Alibaba, bringing 7.1% CTR and 4.4% RPM lift
  - 用户行为长序列建模（with maximum length reaching up to 54000）
  - 核心思想：
    - stage1：通过 hard-search（类目召回）soft-search 向量召回，从长历史数据中返回150左右的结果，
    - stage2：后续阶段使用DIN或DIEN做精细的attention融合。
    - 向量表示是两个阶段共享的，可以同时进行训练。
    - ANN检索用的ALSH
    - 实验结果证明soft比hard要好一丢丢。
      
      - 但是这种机制不就相当于你搜了什么东西，大量推荐同质类的东西，对于电商推荐而言，用户体验还是那么的糟糕和无用。
      -  AUC的提升，能说明系统就比原来要好吗？
      -  以前打天池比赛，外卖推荐，直接一条强机制用户买过的商家再推荐，再次购买的机率很大（一条强规则干到了前50名）
      -  商品类目调细，并把互补、常共现物品补上，是不是ctr还得再上一个台阶。但是这种做法有任何意义吗？
  
  - [`评价`]
    - 摘要、简介有点啰嗦且无用，看完introduction最后的三点贡献，直接跳到模型正文。
    - 公式符号有点乱。
    - 看完后续上线结果，来了个按照类目检索，然后DIN，啧啧。王老师对这个SIM是不是过于赞誉了。
    
    - 创新点好像也就那么回事。


---

## 重排

基于图文内容的物品向量表征
- [ ] [CLIP-OpneAI] Learning transferable visual models from natural language
supervision. In ICML, 2021. [[paper](https://arxiv.org/abs/2103.00020)] [[code](https://github.com/openai/CLIP)] [[blog](https://openai.com/research/clip)]

漏斗多样性
- [X] [MMR] 相关
  - (来自检索算法) 从候选集C中，逐个贪心找出与当前S最不相似的item
- [X] [DPP] Chen et al. Fast greedy map inference for determinantal point process to improve
recommendation diversity. In NIPS, 2018. [[paper](https://arxiv.org/pdf/1709.05135.pdf)]
  - 矩阵det的平方作为相似度度量标准（反常：两2d向量钝角和锐角围成的面积相同，方向差别却很大）
  - Cholesky decomposition算法加速矩阵计算 
---

## coldstart-物品冷启动

---

## A/B测试
- [ ] [Google] Tang et al. Overlapping experiment infrastructure: more, better, faster
experimentation. InKDD, 2010. [[paper]()]

---

## 其他

---

总体感觉推荐方向的论文写的都挺扯淡的。论证角度不是很高，工业细节也不够细。想法和贡献大多来自CV、NLP