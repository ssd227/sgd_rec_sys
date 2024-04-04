## [`Batch内负采样`] Xinyang Yi et al. Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. In RecSys, 2019. [[paper]](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)
  
---
- 采用batch内负样本的方式训练，减少训练时的推理代价，但是引入了采样偏差。
- 核心的采样偏差没有具体介绍，见bengio论文
    - [ ] Adaptive Importance Sampling to Accelerate Training of a Neural Probabilistic Language Model
- 分布式（参数服务器）流数据的频率估计
- 流处理，视频id改变，随着训练过程拟合最新的分布曲线

---
- [`评价-疑问`]
    - 感觉写的一般，没完全理解。核心的纠偏效果也没有说明，论证近似的结果比原结果效果更好？
    - 纠偏sotfmax效果比plain效果更好？？提升了采样效率，还能提升效果？？