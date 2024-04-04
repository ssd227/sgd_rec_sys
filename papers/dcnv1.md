## [DCN V1] Ruoxi Wang et al. Deep & Cross Network for Ad Click Predictions. In ADKDD, 2017. [[paper]](https://arxiv.org/abs/1708.05123)

---
- cross net 逼近高阶多项式近似，是FM的泛化，想法还挺有意思
- 设x=(a,b,c) 直观理解是多项式用W_layer_i糅合后，构成完整的N阶多项式（包括每一项但是系数可调节），然后分别乘法操作(*a, *b, *c升阶)。next layer再用W_layer_i+1糅合
- corss level：vector元素级别的交叉
- 缺点：cross layer模型容量小，对海量数据拟合不好，双塔参数量不均衡，结果偏向deepnet
---
- [`评价`]
    - 行文和图示都挺好理解
    - 公式推导符号理解费劲，可用简单的例子辅助理解