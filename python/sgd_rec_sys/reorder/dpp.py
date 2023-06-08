# DPP 多样性抽样
'''
原理
    利用向量矩阵A的det的特性

    det(A)可以表示为超平行体的体积
    标量值越大，表示这一组向量越不相关

    利用图文向量的item漏斗问题, 就可以转变为
    从N个物品中找出k个物品， 使得超平行体p(k)的面积最大。

优化：
    1、因为计算复杂上的问题，hulu的dpp论文里使用的贪心算法来求解p(k)
    2、为了减少每轮det(A)的计算复杂度,Hulu的论⽂设计了⼀种数值算法，
        仅需𝑂 𝑛*𝑑 + 𝑛𝑘*的时间从𝑛个物品中选出𝑘个物品。
        •给定向量𝒗!,⋯, 𝒗" ∈ ℝ#，需要𝑂 𝑛*𝑑 时间计算𝑨。
        •⽤𝑂 𝑛𝑘* 时间计算所有的⾏列式（利⽤Cholesky分解）

'''

import numpy as np

def dpp(items, k, theta, w=0):
    '''
    dpp算法
    
    输入：
        items - 进入过滤漏斗的物品队列
        k - 多样性过滤漏斗保留的物品数
        theta - 调和reward和sim_score间的超参数
        w - 传入正整数参数后，已选中的物品范围缩小到最近w个item
        
    return:
        保留的物品ids。如果k< len(items)，不截断直接返回原始物品队列ids
        
    问题: 
        当keep items过多后,后续筛选物品的sim值都很高(逼近1)。
        集合r中的元素排序按照物品reward, 达不到多样性效果。
    '''
    # preprocess：corner case
    assert w >= 0
    if items is None or len(items) == 0 or k <=0:
        return set() # 候选队列为空， 返回空集
    if k >= len(items): # 物品数比采样数少， 不需要采样直接返回。
        return set([item.id for item in items])
    
    # step1: init
    index = {item.id: item for item in items}
    s = [] # candidate_items
    r = set(index.keys()) # not_choose_items
    
    # step2: move item with highest rewards[i] in r
    max_reward_id = items[0].id 
    max_reward = items[0].reward
    for id in r:
        cur_item = index[id]
        if cur_item.reward > max_reward:
            max_reward = cur_item.reward
            max_reward_id = cur_item.id
    s.append(max_reward_id)
    r.remove(max_reward_id)
    
    L_As, det_L = calc_L(ids=s, item_index=index)
    
    # step3: choose left k-1 items from r to s
    for _ in range(k-1):
        if w>0:
            choosed, candis = s[-w:], r # 滑动窗口的As分解代价就大了, 需要重新计算LL分解
            L_As, det_L = calc_L(ids=choosed, item_index=index)
        else:
            choosed, candis = s, r
        
        # print(choosed, candis) # for debug
        max_score_id, L_As , det_L = choose_item_id_with_max_det(L_As= L_As,
                                                            det_L_As= det_L,
                                                            choosed=choosed,
                                                            candis=candis,
                                                            item_index=index,
                                                            theta=theta)
        s.append(max_score_id)
        r.remove(max_score_id)
    return s

def calc_L(ids, item_index):
    Vs = np.array([item_index[id].get_norm_emb() for id in ids])
    Vs = Vs.reshape(len(ids), -1)
    As = Vs @ Vs.T
    L = np.linalg.cholesky(As)
    
    # 计算矩阵的det
    det_L = np.linalg.det(L) # 直接用矩阵对角线的值相乘
    
    return L, det_L

    
def choose_item_id_with_max_det(L_As, det_L_As, choosed, candis, item_index, theta):
    """
    inputs:
        L_As - As矩阵的Cholesky分解L
        det_L_As - 上一轮计算好的det
        choosed - 已经选择好的items
        candis - 待选集合
        item_index - 映射{id: item}
        theta - score 调和超参数
    
    outputs:
        max_score_id - 本轮最大分数的item_id
        L_As_i - 下一轮得L_As
        det_L_As_i - 下一轮L_As的det值 
    
    todo: 
        det_L_Asi 的大小并不会对下一轮的det值得比较产生影响
        对分数重要得是利用上一轮L_As计算出得di值得大小。
    
        
    """
    max_score = None
    max_score_id = None
    L_As_i  = None
    det_L_As_i = None
    
    for rid in candis:
        print("\nrid:", rid, "***************")
        # 由L_s 递推到L_si 的中间步骤

        # 由As递推Asi ************************************************
        # 基于cholesky 分解， As = L @ L.T 
        ai = np.array([item_index[rid].get_norm_emb() @ item_index[sid].get_norm_emb() for sid in choosed])
        ai = ai.reshape(-1, 1)
        # print("ai.shape", ai.shape)
        
        # print("L and L_inv's shape", L_As.shape, np.linalg.inv(L_As).shape)
        
        ci = np.linalg.inv(L_As) @ ai
        # print("ci.shape", ci.shape)
        
        di = np.sqrt(1-ci.T@ci)
        
        # 计算当前候选item的det值************************************************
        # print("det_L_As:{}, di:{}".format(det_L_As, di))
        score = theta * item_index[rid].reward + (1-theta) * np.log((det_L_As * di)**2) # 计算当前候选item的det值
        print('rid:{}, score:{}'.format(rid, score))
      
        if max_score:
            if  score <= max_score: continue
        
        # 递推更新L ************************************************
        # 右边补上0
        m,n = L_As.shape
        L_As_pad = np.zeros((m,n+1))  
        L_As_pad[:,:n] = L_As
        # 下面补上一行（ci，di）
        # print("ci, di", ci,ci.shape, di )
        cidi = np.concatenate((ci.T, np.array(di)), axis=1)
        # print('(cidi.shape, L.shape',cidi.shape, L_As_pad.shape)
        
        next_L = np.concatenate((L_As_pad, cidi), axis=0)  
        
        max_score = score
        max_score_id = rid
        L_As_i = next_L
        det_L_As_i = det_L_As * di

    return max_score_id, L_As_i, det_L_As_i