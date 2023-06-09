
class RateInfo:
    def __init__(self, user_file, item_file, rate_file):
        self.user_f = user_file
        self.item_f = item_file
        self.rate_f = rate_file
        
        self.user_meta = None 
        self.item_meta = None
        self.rate_meta = None
    
    def user_meta_info(self):
        if self.user_meta:
            return self.user_meta
        self.user_meta = self.load_user_info(self.user_f)
        return self.user_meta
    
    def item_meta_info(self):
        if self.item_meta:
            return self.item_meta
        self.item_meta = self.load_item_info(self.item_f)
        return self.item_meta
    
    def rate_meta_info(self):
        if self.rate_meta:
            return self.rate_meta
        self.rate_meta = self.load_rate_info(self.rate_f)
        return self.rate_meta
    

    # 构造用户对物品的评价
    def load_user_info(self, user_file):
        meta_info = dict()
        with open(user_file) as f:
            # 解析表头信息
            s = f.readline()
            if s.startswith("#"):
                col_name = [col.strip() for col in s[1:].strip().split('|')]
                # print("user table cols:", col_name)
                meta_info["col_name"] = col_name
            else:
                raise ValueError("not legal user file")
            
            id2name = dict()
            name2id = dict()
            # 解析用户信息
            for line in f.readlines():
                s= line.strip().split(';')
                if len(s) == len(meta_info["col_name"]):
                    id, name = int(s[0].strip()), s[1].strip()
                    id2name[id] = name
                    name2id[name] = id
            
            meta_info['id2name'] = id2name
            meta_info['name2id'] = name2id
        return meta_info
    
    def load_item_info(self, item_file):
        meta_info = dict()
        with open(item_file) as f:
            # 解析表头信息
            s = f.readline()
            if s.startswith("#"):
                col_name = [col.strip() for col in s[1:].strip().split('|')]
                # print("item table cols:", col_name)
                meta_info["col_name"] = col_name
            else:
                raise ValueError("not legal item file")
            
            id2name = dict()
            name2id = dict()

            # 解析物品信息
            for line in f.readlines():
                s= line.strip().split(';')
                if len(s) == len(meta_info["col_name"]):
                    id, name = int(s[0].strip()), s[1].strip()
                    id2name[id] = name
                    name2id[name] = id

            meta_info['id2name'] = id2name
            meta_info['name2id'] = name2id
        return meta_info
        
    
    def load_rate_info(self, rate_file):
        meta_info = dict()
        with open(rate_file) as f:
            # 解析表头信息
            s = f.readline()
            if s.startswith("#"):
                col_name = [col.strip() for col in s[1:].strip().split('|')]
                # print("rate table cols:", col_name)
                meta_info["col_name"] = col_name
            else:
                raise ValueError("not legal rate file")

            rate_pairs = []
            # 解析物品信息
            for line in f.readlines():
                s= line.strip().split(';')
                if len(s) == len(meta_info["col_name"]):
                    uid, iid, rate = s
                    rate_pairs.append([int(x.strip()) for x in s])
            meta_info['rate_pairs'] = rate_pairs
        return meta_info