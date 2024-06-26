'''
• 长序列（长期兴趣）优于短序列（近期兴趣）。
• 注意⼒机制优于简单平均。
• Soft search还是hard search？取决于⼯程基建。
• 使⽤时间信息有提升。
    • SIM的序列长，记录⽤户长期⾏为。
    • 时间越久远，重要性越低。

SIM的主体思路和推荐漏斗一致，对lastn做topk召回
    hard search
        工程实现比较简单，通过类目过滤长用户交互序列(不在此处实现)
    Soft Search
        使用向量查找取topK时，需要注意用户交互物品和候选物品向量表示的学习方式

工程实现难点：共用一套物品emb, 同时训练两个部分模型
'''