##  [`xDeepFM`] Jianxun Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. In , 2018. [[paper]](https://arxiv.org/pdf/1803.05170.pdf)

---
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
 
 --- 
- [`评价`]
    - 模型太丑了，公式符号巨难看。这也能发KDD
    - 引入+背景介绍部分写的不如DCN v1、v2（有点啰嗦）
    - 下面这段论述就有点扯淡了，DCN v1的缺点是同层使用一个变换，使得模型容量较小，除了在多项式项前的系数组合上存在限制使得搜索空间较小，不影响模型对多项式的拟合。
      - The CrossNet can learn feature interactions very efficiently (the complexity is negligible compared with a DNN model), however the downsides are: 
          - (1) the output of CrossNet is limited in a special form, with each hidden layer is a scalar multiple of x0;
          - (2) interactions come in a bit-wise fashion.
    - CNN RNN都套上了，实现的真的很丑。2018年论文的局限性吗？