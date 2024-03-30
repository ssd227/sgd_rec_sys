
import pickle
import numpy as np
__all__ = ['FakeLastnDataFactory',]

class FakeLastnDataFactory:
    def __init__(self,
                n_samples,
                fea_dim,
                last_n,
                dtype=np.float32) -> None:

        self.n_samples = n_samples
        self.fea_dim = fea_dim
        self.last_n = last_n
        
        self.dtype = dtype
        self.made = False
        
        # 存放生成的伪数据
        self.user_behavior_cache = None
        self.candidate_ad_cache = None
        
    def make(self):
        if not self.made:
            self.user_behavior_cache = self.make_user_behavior(self.n_samples, self.last_n, self.fea_dim).tolist()
            self.candidate_ad_cache = self.make_candidate_ad(self.n_samples, self.fea_dim).tolist()
            self.made=True

    def make_user_behavior(self, n_samples, last_n, fea_dim):
        data = np.random.randn(n_samples, last_n, fea_dim).astype(self.dtype) # 均值方差 mu=0, sigma=1
        print('user_behavior success, shape:', data.shape)
        return data
    
    def make_candidate_ad(self, n_samples, fea_dim):
        data = np.random.randn(n_samples, fea_dim).astype(self.dtype)
        print('candidate_ad success, shape:', data.shape)
        return data
    
    # 保存为python array格式文件
    def presist(self, save_file='default_fake_data.pkl'):
        if not self.made:
            self.make()
        
        meta = {'n': self.n_samples}
        data = dict()
        data['user_behavior'] = self.user_behavior_cache
        data['candidate_ad'] = self.candidate_ad_cache

        samples = {'meta': meta, 'data': data}
        with open(save_file, 'wb') as f:
            pickle.dump(samples, f)