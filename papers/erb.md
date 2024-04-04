## [经典] [`EBR`] Jui-Ting Huang et al. Embedding-based Retrieval in Facebook Search. In KDD, 2020. [[paper]](https://arxiv.org/abs/2006.11632)

---
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
  
---
- [`评价`]
  - 文笔流畅，工程细节指导价值很高。多看几遍