
import pickle
import numpy as np
__all__ = ['FakeMultiTaskDataFactory',]

class FakeMultiTaskDataFactory:
    def __init__(self,
                n_samples,
                fea_emb_dim,
                target_num,
                dtype=np.float32) -> None:

        self.n_samples = n_samples
        self.fea_emb_dim = fea_emb_dim
        self.target_num = target_num
        
        self.dtype = dtype
        self.made = False
        
        # 存放生成的伪数据
        self.fea_emb_cache = None
        self.targets_cache = None
        
    def make(self):
        if not self.made:
            self.fea_emb_cache = self.make_fea_emb(self.n_samples, self.fea_emb_dim).tolist()
            self.targets_cache = self.make_targets(self.n_samples, self.target_num).tolist()
            self.made=True

    def make_fea_emb(self, n_samples, fea_dim):
        data = np.random.randn(n_samples, fea_dim).astype(self.dtype) # 均值方差 mu=0, sigma=1
        print('feature embedding success, shape:', data.shape)
        return data
    
    def make_targets(self, n_samples, target_num):
        data = np.random.randint(0, 2, size=(n_samples, target_num)).astype(self.dtype)
        print('targets success, shape:', data.shape)
        return data
    

    # 保存为python array格式文件
    def presist(self, save_file='default_fake_multitask_data.pkl'):
        if not self.made:
            self.make()
        
        meta = {'n': self.n_samples}
        data = dict()
        data['fea_emb'] = self.fea_emb_cache
        data['targets'] = self.targets_cache

        samples = {'meta': meta, 'data': data}
        with open(save_file, 'wb') as f:
            pickle.dump(samples, f)