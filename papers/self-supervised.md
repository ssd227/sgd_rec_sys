## [`自监督学习`]Tiansheng Yao et al. Self-supervised Learning for Large-scale Item Recommendations.In CIKM, 2021. [[paper]](https://arxiv.org/abs/2007.12865)

---
- 通过对比学习，增强tail数据的表示，提升稀疏数据下的召回准确性
- 数据增强通过CFM来实现
- 优化目标融合：Loss_main + alpha(Loss_self_pred)
  - 自监督的数据分布和 main监督学习中的分布不同
---
- [`评价`]
  - 比上述的纠偏论文要好点，主要背景知识交代的清楚，向量近似查找MIPS
  - 数据增强的细节还是没写清楚CFM，实验部分可以快速看一遍表格。