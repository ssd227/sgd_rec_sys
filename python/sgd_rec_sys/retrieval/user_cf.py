import numpy as np
import math
from numpy.linalg import norm

"""
UserCF的基本思想：
• 如果⽤户𝑢𝑠𝑒𝑟1 跟⽤户𝑢𝑠𝑒𝑟2 相似，⽽且𝑢𝑠𝑒𝑟2喜欢某物品，
• 那么⽤户𝑢𝑠𝑒𝑟2也很可能喜欢该物品。


优点：
    每个人的兴趣点都很广泛，usercf可以快速的给每个用户发散出不同的兴趣点，比如热点新闻的推荐。
"""

class UserCF():
    def __init__(self, meta_info):
        self.meta_info = meta_info # 用户、物品、rate三张表的信息
        
        # 记录每个用户喜欢的物品集合 和 每个物品的喜欢人数
        self.user_dict, self.item_like_n = self.parser_rate(self.meta_info)

        
    def parser_rate(self, meta_data):
        rate_pairs = meta_data.rate_meta_info()['rate_pairs']  #  list[ list[uid, iid, rate] ]
        
        parsed_dict = dict() # 二级字典，方便查找公共元素
        for uid, iid, rate in rate_pairs:
            if uid not in parsed_dict:
                parsed_dict[uid] = dict()
                parsed_dict[uid][iid] = rate
            else:
                parsed_dict[uid][iid]=rate # 用户的新评分会覆盖旧评分（存在重复评分）
                
        # 统计每个物品的喜欢人数，来定义是否是热门物品
        like_n = dict()
        for uid, iid, rate in rate_pairs:
            if rate >=1:
                if iid in like_n:
                    like_n[iid] +=1
                else:
                    like_n[iid] = 1
            else:
                continue
        
        return parsed_dict, like_n
        
    
    # cos-sim: 可以兼顾喜欢和不喜欢的评分
    def sim(self, u1, u2): 
        j1 = self.user_dict[u1]
        j2 = self.user_dict[u2]
        # print("J1, J2", j1, j2)
        
        common =[]
        for jid in j1.keys():
            if jid in j2:
                common.append(jid)
                
        # print("common", common)
        
        p = sum([j1[id]*j2[id] for id in common])
        q = (sum([like**2 for like in j1.values()]) ** 0.5) * \
            (sum([like**2 for like in j2.values()]) ** 0.5) 
        return p/q
    

    # jarcard 只考虑喜欢物品的个数
    # 只统计喜欢，不考虑不喜欢
    # 不论冷门、热门，物品权重都是1。
    # 缺点：热门物品需要降权重
    def jarcard_sim(self, u1, u2): 
        
        j1 = dict()
        for iid, rate in self.user_dict[u1].items():
            if rate >= 1: j1[iid] = 1
        
        j2 = dict()
        for iid, rate in self.user_dict[u2].items():
            if rate >= 1: j2[iid] = 1

        print("J1, J2", j1, j2)
        
        common =[]
        for jid in j1.keys():
            if jid in j2:
                common.append(jid)
                
        print("common", common)
        
        p = len(common)
        q = (len(j1) ** 0.5) * \
            (len(j2) ** 0.5) 
        return p/q

    # 降低热门物品权重的jarcard_sim
    def jarcard_sim_with_suppressing_hot(self, u1, u2):
        j1 = dict()
        for iid, rate in self.user_dict[u1].items():
            if rate >= 1: j1[iid] = 1
        
        j2 = dict()
        for iid, rate in self.user_dict[u2].items():
            if rate >= 1: j2[iid] = 1

        print("J1, J2", j1, j2)
        
        common =[]
        for jid in j1.keys():
            if jid in j2:
                common.append(jid)
                
        print("common", common)
        
        p = sum([ math.log(1+self.item_like_n[itemid]) ** -1 for itemid in common])
        q = (len(j1) ** 0.5) * \
            (len(j2) ** 0.5) 
        return p/q
    


