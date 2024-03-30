

class Bandit:
    # 设计思路: 把算法做成基类问题也不大, 就不用使用插拔模式
    def __init__(self) -> None:
        pass
        self.state # todo 整合进具体算法类中
        self.alg = None # 选择的老虎机算法， 
        self.env_interface = None # 根据选择结果，待调用的外部更新参数， 需要返回探索结果 reward
    
    def update():
        # 根据现有状态，使用alg 选择下一步策略
        
        # 与env交互，拿到反馈结果。
        
        # 跟新现有状态
        pass
        
    
# 下面三种算法的快速实现参考 https://zhuanlan.zhihu.com/p/80261581
def epsilon_greedy(eps, candidates):
    pass

def thompson_sampling():
    pass

def ucb():
    pass
    