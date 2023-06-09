"""
ItemCF 的基本思想：
    • 如果⽤户喜欢物品𝑖𝑡𝑒𝑚1，⽽且物品𝑖𝑡𝑒𝑚1 与𝑖𝑡𝑒𝑚2 相似，
    • 那么⽤户很可能喜欢物品𝑖𝑡𝑒𝑚2。

优势：兴趣点专一，只推荐用户交互历史中正反馈item的相似item   
"""

class ItemCF():
    def __init__(self, meta_info):
        self.meta_info = meta_info # 用户、物品、rate三张表的信息
        self.item_dict = self.parser_rate(self.meta_info)   # 记录每个物品的用户打分集合，方便计算相似度
        
    def parser_rate(self, meta_data):
        rate_pairs = meta_data.rate_meta_info()['rate_pairs']  #  list[ list[uid, iid, rate] ]
        
        parsed_dict = dict() # 二级字典，方便查找公共元素
        for uid, iid, rate in rate_pairs:
            if iid not in parsed_dict:
                parsed_dict[iid] = dict()
                parsed_dict[iid][uid] = rate
            else:
                parsed_dict[iid][uid]=rate # 用户的新评分会覆盖旧评分（存在重复评分）
        return parsed_dict
        
    
    # 计算物品间的相似度(cosin sim)
    def sim(self, iid1, iid2): 
        # 评分大多是稀疏数据，不直接使用向量计算
        w1 = self.item_dict[iid1]
        w2 = self.item_dict[iid2]
        print("w1, w2", w1, w2)
        
        common =[]
        for uid in w1.keys():
            if uid in w2:
                common.append(uid)
                
        print("common", common)
        
        p = sum([w1[id]*w2[id] for id in common])
        q = (sum([like**2 for like in w1.values()]) ** 0.5) * \
            (sum([like**2 for like in w2.values()]) ** 0.5) 
        return p/q