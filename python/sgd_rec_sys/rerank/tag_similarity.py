# 基于物品属性标签的相似性度量
# todo, 这里仅实现了原理性代码。未在apps、tests里测试


# 一级类目、二级类目、品牌标签
def tag_sim(tag1, tag2, simfuns):
    # a1, a2, branda = tag1
    # b1, b2, brandb = tag2
    # simfuncs 是一个list, 含对应tag的三个计算相似度的函数

    sim_res = [f(a,b) for a,b, f in zip(tag1, tag2, simfuns)]
    return sim_res # 三级tag相似度，如何融合score靠下游自行判断