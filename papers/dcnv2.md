## [DCN V2] Ruoxi Wang, et al. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems. InWWW, 2021. [[paper]](https://arxiv.org/abs/2008.13535)

---
  - V1在大数据量的工业数据上，cross部分的参数量不匹配deep，且W_l的参数共享限制使得模型特征corss的表达能力较弱。
  - cross思路：xl线性组合后,hamada乘升阶，补上低阶项(+xl)，bias拟合常数项，最终是x0*bl（也不是不行）
  - 使用MMOE和low-rank技术减小模型规模，trad-off between performance and efficiency

---
- [`评价`]
    - wide&deep并行架构，相关工作将deepFM、DCN、xdeepFM串起来了。很好！
    - 疑问：low rank的思路就是奇异值分解，把奇异值低的正交基排除掉。但是模型直接上低秩模型也能训练的好吗？类似lora的fine tune？
    - 多项式逼近解释依旧跳过，感觉是为了数学而数学，公式死难看。
    - 实验部分写的挺好，充足的各种对比。