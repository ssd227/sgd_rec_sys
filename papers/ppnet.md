## [PPNet] Parameter Personalized Net-快⼿落地万亿参数推荐精排模型，2021。 [[Blog]](https://ai.51cto.com/art/202102/644214.html)
---
- [ ] [LHUC]Pawel Swietojanski, Jinyu Li, & Steve Renals. Learning hidden unit contributions for unsupervised acoustic model adaptation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2016. [[paper]]()
- 在语音识别领域中，LHUC算法(learning hidden unit contributions)核心思想是做说话人自适应(speaker adaptation)，在DNN网络中为每个说话人学习一个特定的隐式单位贡献(hidden unit contributions)，来提升不同说话人的语音识别效果。借鉴LHUC的思想，快手在精排模型上展开了尝试。PPNet于2019年上线后，显著提升了模型的CTR目标预估能力。