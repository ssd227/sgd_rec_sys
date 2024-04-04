## [经典][Youtube] Paul Covington, Jay Adams, & Emre Sargin. Deep Neural Networks for YouTube Recommendations. In RecSys, 2016. [[pape]](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)

--- 
- 核心贡献：向量化召回 
- 借鉴word2vec，将相似的user_emb和item_emb映射到相同的空间，达到聚类效果。使用向量检索快速返回topk候选样本