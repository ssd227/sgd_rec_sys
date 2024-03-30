import pickle
import numpy as np
__all__ = ['FakeCtrDataFactory',]

class FakeCtrDataFactory:
    def __init__(self,
                 n_samples,
                 n_dense_feas,
                 one_hot_fea_list,
                 multi_hot_fea_list,
                 dtype=np.float32) -> None:

        self.n_samples = n_samples
        self.n_dense_feas = n_dense_feas
        self.one_hot_fea_list = one_hot_fea_list
        self.multi_hot_fea_list = multi_hot_fea_list
        
        self.dtype = dtype
        self.made = False
        
        # 存放生成的伪数据
        self.dense_cache = None
        self.one_hot_cache = None
        self.multi_hot_cache = None
        self.label_cache = None
        
    def make(self):
        if not self.made:
            if self.n_dense_feas!=0:
                self.dense_cache = self.make_dense_fea(self.n_samples, self.n_dense_feas).tolist()
            if len(self.one_hot_fea_list)>0:
                self.one_hot_cache = self.make_one_hot_fea(self.n_samples, self.one_hot_fea_list).tolist()
            if len(self.multi_hot_fea_list)>0:
                self.multi_hot_cache = self.make_multi_hot_fea(self.n_samples, self.multi_hot_fea_list)
            self.label_cache = self.make_label(self.n_samples).tolist()
            self.made=True

    def make_label(self, n_samples):
        data = np.random.randn(n_samples, 1).astype(self.dtype).reshape(-1) # 均值方差 mu=0, sigma=1
        data = (data - data.min()) / (data.max() - data.min())
        print('label success, shape:', data.shape)
        return data
    
    def make_dense_fea(self, n_samples, n_feas):
        data = np.random.randn(n_samples, n_feas).astype(self.dtype) # 均值方差 mu=0, sigma=1
        print('densed feas success, shape:', data.shape)
        return data
    
    def make_one_hot_fea(self, n_samples, fea_list):
        n_cols = len(fea_list)
        data = []
        for i in range(n_cols):
            # 省去了字典index步骤, 比如共K个token，生成类别区间[1, K]
            col_data = np.random.randint(low=1, high=fea_list[i]+1, size=(n_samples, 1))
            data.append(col_data)
        
        data = np.stack(data, axis=1).reshape(n_samples, n_cols).astype(self.dtype)
        print('one-hot feas success, shape:', data.shape)
        return data
    
    def make_multi_hot_fea(self, n_samples, fea_list, max_len=4):
        n_cols = len(fea_list)
        data = []
        for i in range(n_cols):
            # 省去了字典index步骤, 比如共K个token，生成类别区间[1, K]
            one_col = []
            for _ in range(n_samples):
                rand_k = np.random.randint(1, max_len)  # 随机生成[1, max_len]个 multi-hot 值
                x = np.random.randint(low=1, high=fea_list[i]+1, size=(1, rand_k)).reshape(-1).tolist()
                # print(x, type(x))
                one_col.append(x)
            data.append(one_col)
        print('multi-hot feas success')  
        return list(zip(*data)) # 行列互换
    
    # 保存为python array格式文件
    def presist(self, save_file='default_fake_data.pkl'):
        if not self.made:
            self.make()
        
        meta = {'n': self.n_samples}
        data = dict()
        if self.dense_cache:
            meta['n_dense_feas'] = self.n_dense_feas
            data['dense'] = self.dense_cache
        if self.one_hot_cache:
            meta['one-hot_fea_list']=self.one_hot_fea_list
            data['one-hot'] = self.one_hot_cache
        if self.multi_hot_cache:
            meta['multi-hot_fea_list']=self.multi_hot_fea_list
            data['multi-hot'] = self.multi_hot_cache
        if self.label_cache:
            data['label'] = self.label_cache

        samples = {'meta': meta, 'data': data}
        
        with open(save_file, 'wb') as f:
            pickle.dump(samples, f)