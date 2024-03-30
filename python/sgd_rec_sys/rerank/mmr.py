''' 
    MMR(Maximal Marginal Relevance)

    粗排序、精排后进入筛选漏斗阶段候选物品队列。
    既要保证候选物品排序分数高, 又要具有多样性

'''

import numpy as np    

class Item:
    def __init__(self, id) -> None:
        self.id = id
        self.reward = None
        self.emb = None
         
    def set_reward(self, x:float):
        self.reward = x
        
    def set_emb(self, e:np.array):
        self.emb = e
        
    def get_norm_emb(self):
        if self.emb is None:
            raise ValueError("not set variable emb for Item:[{}]", self.id)
        return self.emb / np.linalg.norm(self.emb, ord=2)

    def sim(self, other):
        # 物品向量的余弦相似度
        # 各自norm,然后计算dot
        a, b= self.emb, other.emb
        an = a / np.linalg.norm(a, ord=2)
        bn = b / np.linalg.norm(b, ord=2)
        return np.dot(an, bn)


def mmr(items, k, theta, w=0):
    '''
    mmr算法
    
    输入：
        items - 进入过滤漏斗的物品队列
        k - 多样性过滤漏斗保留的物品数
        theta - 调和reward和sim_score间的超参数
        w - 传入正整数参数后，已选中的物品范围缩小到最近w个item
        
    return:
        保留的物品ids。如果k< len(items)，不截断直接返回原始物品队列ids
        
    问题: 
        当keep items过多后,后续筛选物品的sim值都很高(逼近1)。
        集合r中的元素排序按照物品reward, 达不到多样性效果。
    '''
    # preprocess：corner case
    assert w >= 0
    if items is None or len(items) == 0 or k <=0:
        return set() # 候选队列为空， 返回空集
    if k >= len(items): # 物品数比采样数少， 不需要采样直接返回。
        return set([item.id for item in items])
    
    # step1: init
    index = {item.id: item for item in items}
    s = [] # candidate_items
    r = set(index.keys()) # not_choose_items
    
    # step2: move item with highest rewards[i] in r
    max_reward_id = items[0].id 
    max_reward = items[0].reward
    for id in r:
        cur_item = index[id]
        if cur_item.reward > max_reward:
            max_reward = cur_item.reward
            max_reward_id = cur_item.id
    s.append(max_reward_id)
    r.remove(max_reward_id)
    
    # step3: choose left k-1 items from r to s
    for _ in range(k-1):
        if w>0:
            choosed, candis = s[-w:], r
        else:
            choosed, candis = s, r
        
        # print(choosed, candis) # for debug
        max_mr_id = choose_item_id_with_max_mr(choosed=choosed,
                                               candis=candis,
                                               item_index=index,
                                               theta=theta)
        s.append(max_mr_id)
        r.remove(max_mr_id)
    return s


def choose_item_id_with_max_mr(choosed, candis, item_index, theta):
    """
    choosed - 已选中集合
    candis - 待选集合 
    """
    max_mr = None
    max_mr_id = None
    # log for debug
    # log_mrs = [] # log
    
    for rid in candis:
        # log_mr = []  # log
        # calc_max_sim for each item[id=rid]
        max_sim = None
        for sid in choosed:
            sim_score =  item_index[rid].sim(item_index[sid])
            if max_sim:
                max_sim = max(max_sim, sim_score)
            else:
                max_sim = sim_score
        # log_mr.append("item:"+ str(rid))  # log
        # log_mr.append(max_sim) # log
        # 计算当前候选item的mr值
        mr = theta * item_index[rid].reward - (1-theta) * max_sim
        # log_mr.append([(mr,theta * item_index[rid].reward, (1-theta) * max_sim), (item_index[rid].reward, max_sim)]) # log

        if max_mr:
            if mr > max_mr:
                max_mr = mr
                max_mr_id = rid
        else: # not init
            max_mr = mr
            max_mr_id = rid
    #     print("max_mr[{}], max_mrid[{}]".format(max_mr, max_mr_id)) # log
    #     log_mrs.append(log_mr) # log    
    # print("log_mrs", log_mrs) # log
    
    return max_mr_id