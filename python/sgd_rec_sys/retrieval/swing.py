"""
基于item cf的原理的工程改进


感觉也叠加了user cf的一些改进
    基于用户喜欢的物品去计算overlap，来降低权重

"""



class Swing:
    """
        实现时，暂时只考虑用户喜欢的物品
        
    """

    def __init__(self, meta_info)-> None:
        self.meta_info = meta_info # 用户、物品、rate三张表的信息
        self.item_dict, self.user_dict = self.parser_rate(self.meta_info) # rate数据merge预处理，方便计算sim
        
    def parser_rate(self, meta_data):
        rate_pairs = meta_data.rate_meta_info()['rate_pairs']  #  list[ list[uid, iid, rate] ]
        
        # 基于物品
        item_dict = dict() # 二级字典，方便查找公共元素
        for uid, iid, rate in rate_pairs:
            if rate > 0: # 实现时，暂时只考虑用户喜欢的物品
                if iid not in item_dict:
                    item_dict[iid] = dict()
                    item_dict[iid][uid] = rate
                else:
                    item_dict[iid][uid]=rate # 用户的新评分会覆盖旧评分（存在重复评分）
        
        # 基于用户
        user_dict = dict() # 二级字典，方便查找公共元素
        for uid, iid, rate in rate_pairs:
            if rate > 0: # 实现时，暂时只考虑用户喜欢的物品
                if uid not in user_dict:
                    user_dict[uid] = dict()
                    user_dict[uid][iid] = rate
                else:
                    user_dict[uid][iid]=rate # 用户的新评分会覆盖旧评分（存在重复评分）
        
        return item_dict, user_dict
        
    
    def overlap(self, u1, u2):
        j1 = self.user_dict[u1]
        j2 = self.user_dict[u2]
        
        common_count = 0
        for iid in j1.keys():
            if iid in j2:
                common_count += 1
        
        return common_count
            
    # TODO: alpha 是超参数，需要设计和调参
    def sim(self, iid1, iid2, alpha=1): 
        w1 = self.item_dict[iid1]
        w2 = self.item_dict[iid2]
        print("w1, w2", w1, w2)
        
        common =[]
        for uid in w1.keys():
            if uid in w2:
                common.append(uid)
                
        print("common", common)
        
        sim_score = 0
        # TODO：相似度的计算完全不需要考虑w1和w2的大小吗？
        
        seen = set() # (u1, u2)只计算一遍，不重复计算(u2,u1)。考虑到common的数量N，带来N**2的sim值差异
        for u1 in common:
            for u2 in common:
                if (u1, u2) in seen or (u2, u1) in seen:
                    continue
                else:
                    sim_score += 1 / (alpha + self.overlap(u1, u2))
                    seen.add((u1,u2))
        return sim_score