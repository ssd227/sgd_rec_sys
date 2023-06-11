# 预估分数的融合


def weight_add(ps, ws):
    # ws 作为超参数需要A/B实验去调
    p_click, p_like, p_collect= ps[:3]
    w1, w2 = ws[:2]
    
    merge_score = p_click + w1* p_like + w2 * p_collect
    return merge_score

def meaning_weight_add(ps, ws):
    # ws 作为超参数需要A/B实验去调
    p_click, p_like, p_collect= ps[:3]
    w1, w2 = ws[:2]
    
    # 具有实际意义(只有用户点击了的物品才有喜欢、点赞、收藏、转发等后续操作)
    merge_score = p_click * (1 + w1*p_like + w2*p_collect)
    return merge_score

# 某海外短视频app融分公式(抖音？？)
def video_merge_score1(ps, ws, a_s):
    p_time, p_like = ps[:2]
    a1, a2 = a_s[:2]
    w1, w2= ws[:2]
    
    merge_score = (1+w1*p_time)**a1 *\
                (1+w2*p_like)**a2
    return merge_score


# 国内某短视频APP的融分公式(快手？？)
# n篇候选视频排序，由预估值(时长、点击、喜欢)转变为预估排序值
# 得分值定义为 # 1/(pow(r_x)**ax + bx)
def video_merge_score2(rs, ws, a_s, bs):
    r_time, r_click, r_like  = rs[:3]
    a1, a2, a3 = a_s[:3]
    b1, b2, b3 = bs[:3]
    w1, w2, w3 = ws[:3]
    
    merge_score = w1/(pow(r_time)**a1 + b1) +\
                w2/(pow(r_click)**a2 + b2) +\
                w3/(pow(r_like)**a3 + b3)
    return merge_score

# 某电商融分公式
# 电商的转化流程: 曝光->点击->加购物车->付款
# 融合分数符合最终的预估的成交额度
def e_commerce_merge_score(ps, a_s, price):
    p_click, p_cart, p_pay = ps[:3]
    a1, a2, a3, a4 = a_s[:4]

    merge_score = p_click** a1 * \
                p_cart**a2 * \
                p_pay**a3 * \
                price**a4        
    return merge_score




                
                    
    