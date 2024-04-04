
import pickle
import numpy as np
__all__ = ['FakeDssmDataFactory',]

class FakeDssmDataFactory:
    def __init__(self,
                n_samples,
                user_fea_dim,
                item_fea_dim,
                item_fea_num, # 至少有一个正样本
                dtype=np.float32) -> None:

        self.n_samples = n_samples
        self.use_fea_dim = user_fea_dim
        self.item_fea_dim = item_fea_dim
        self.item_fea_num = item_fea_num
        
        self.dtype = dtype
        self.made = False
        
        # 存放生成的伪数据
        self.user_fea_emb_cache = None
        self.item_fea_emb_cache = None
        
    def make(self):
        if not self.made:
            self.user_fea_emb_cache = self.make_user_fea_emb(self.n_samples, self.use_fea_dim).tolist()
            self.item_fea_emb_cache = self.make_item_fea_emb(self.n_samples, self.item_fea_dim, self.item_fea_num).tolist()
            self.made=True

    def make_user_fea_emb(self, n_samples, fea_dim):
        data = np.random.randn(n_samples, fea_dim).astype(self.dtype) # 均值方差 mu=0, sigma=1
        print('user feature embedding success, shape:', data.shape)
        return data
    
    def make_item_fea_emb(self, n_samples, fea_dim, fea_nums):
        data = np.random.randn(n_samples, fea_nums, fea_dim).astype(self.dtype)
        print('item feature embedding success, shape:', data.shape)
        return data
    
    # 保存为python array格式文件
    def presist(self, save_file='default_fake_dssm_data.pkl'):
        if not self.made:
            self.make()
        
        meta = {'n': self.n_samples}
        data = dict()
        data['user_fea_emb'] = self.user_fea_emb_cache
        data['item_fea_emb'] = self.item_fea_emb_cache

        samples = {'meta': meta, 'data': data}
        with open(save_file, 'wb') as f:
            pickle.dump(samples, f)