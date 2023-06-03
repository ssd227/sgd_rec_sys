''' 
    MMR(Maximal Marginal Relevance)

    粗排序、精排后进入筛选漏斗阶段候选物品队列。
    既要保证候选物品排序分数高, 又要具有多样性


'''

import numpy as np

def cossim(e1, e2):
    pass
    

class Item:
    def __init__(self, id) -> None:
        self.id = id
        self.reward = None
        self.emb = None
         
    def set_reward(self, x:float):
        self.reward = x
        
    def set_emb(self, e:np.array):
        self.emb = e

    def sim(self, other):
        # 使用物品向量的余弦相似度
        # 先各自norm，然后计算dot
        a, b= self.emb, b = other.emb
        an = a / np.linalg.norm(a, ord=2)
        bn = b / np.linalg.norm(b, ord=2)
        return np.dot(an, bn)


def mmr_vanilla(items, k,):
    # step1: init
    s = set() # candidate_items
    r = set() # not_choose_items
    
    # step2: move item with highest rewards[i] in r 
    
    # mr_i= theta * reward_i - (1-theta)* max(sim(i,j))   for j in s 

    #
    return s


def mmr_with_window():