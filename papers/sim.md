## [SIM] Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction In CIKM, 2020. [[paper]](https://arxiv.org/abs/2006.05639)

---
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
---
  - [`评价`]
    - 摘要、简介有点啰嗦且无用，看完introduction最后的三点贡献，直接跳到模型正文。
    - 公式符号有点乱。
    - 看完后续上线结果，来了个按照类目检索，然后DIN，啧啧。王老师对这个SIM是不是过于赞誉了。
    
    - 创新点好像也就那么回事。