"""
在召回、排序阶段，通常会对广告队列做过滤操作。
对已展现给用户的广告(30天内的广告), 不再次向用户推荐。
使用bloom filter 可以快速的判断某一广告是否已经推荐给用户。
    已展现的广告, 100%判定正确。
    未展现的广告， 可能存在误判。 (广告未展现, 但是误判为已展现, 被错误的过滤掉)

参数设置:
    error_rate: 误判概率
    capacity:
    m:

具体实现:
    需要进行位操作，
"""

from math import log
from .hash_help import MyHashFunctions
from bitarray import bitarray


class BloomFilter:
    def __init__(self, n, error_rate=0.01) -> None:
        self.error_rate = error_rate
        self.n = n  # capacity
        
        log_sigma_reverse = log(1/self.error_rate)
        self.k = int(1.44 * log_sigma_reverse)  # best nums_bits
        self.m = int(2 * n * log_sigma_reverse)  # best hash func nums
        
        self.bits = bitarray(self.m)
        self.bits[:] = 0 # setall(0)
        
        self.hash_funcs = MyHashFunctions(self.k, self.m) # k funcs map item id to [0, m-1]
    
    def add(self, item):
        for f  in self.hash_funcs:
            self.bits[f(item)] = 1
    
    def haveSeen(self, item):
        for f in self.hash_funcs:
            id = f(item)
            if self.bits[id] == 0:
                return False
            assert self.bits[id] == 1
        return True
        
    def __contains__(self, item):
        return self.haveSeen(item=item)
    
    def show_param_log(self):
        print("inner params, m:[{}], hash_func_num:[{}], capacity:[{}]".format(self.m, self.k, self.n))