import math
import pandas as pd

n = 943
m = 1682
training_data = pd.read_csv('../../dataset/ml-100k/u1.base.occf', delim_whitespace=True, index_col=False, header=None)
testing_data = pd.read_csv('../../dataset/ml-100k/u1.test.occf', delim_whitespace=True, index_col=False, header=None)
training_data_length = training_data.index.size
testing_data_length = testing_data.index.size

def initialization():
    mu = training_data_length / n / m

    r = {}
    r_item = {}
    I_u = {}
    I = []
    U = []
    for index, row in training_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        
        if not usr_id in U:
            U.append(usr_id)
        if not item_id in I:
            I.append(item_id)

        r.setdefault(usr_id)
        r_item.setdefault(item_id)
        I_u.setdefault(usr_id)

        if r[usr_id] == None:
            r[usr_id] = []
        if r_item[item_id] == None:
            r_item[item_id] = []
        if I_u[usr_id] == None:
            I_u[usr_id] = []
        
        if not item_id in r[usr_id]:
            r[usr_id].append(item_id)
        if not usr_id in r_item[item_id]:
            r_item[item_id].append(usr_id)
        if not item_id in I_u[usr_id]:
            I_u[usr_id].append(item_id)
    
    

    # as simple prediction
    b_item = {}
    for item_id in range(1, m + 1):
        if not item_id in r_item:
            b_item[item_id] = -mu
            continue
        b_item[item_id] = len(r_item[item_id]) / n - mu

    r_te = {}
    I_te = {}
    U_te = []
    I_re_u = list(dict(sorted(b_item.items(), key=lambda item: item[1], reverse=True)).keys())
    I_re = {}
    for index, row in testing_data.iterrows():
        usr_id = row[0]
        item_id = row[1]

        if not usr_id in U_te:
            U_te.append(usr_id)

        r_te.setdefault(usr_id)
        I_te.setdefault(usr_id)
        I_re.setdefault(usr_id)

        if r_te[usr_id] == None:
            r_te[usr_id] = []
        if I_te[usr_id] == None:
            I_te[usr_id] = []
        if I_re[usr_id] == None:
            I_re[usr_id] = []
            re_index = 0
            while re_index < len(I_re_u):
                if not I_re_u[re_index] in I_u[usr_id]:
                    I_re[usr_id].append(I_re_u[re_index])
                re_index += 1

        if not item_id in r_te[usr_id]:
            r_te[usr_id].append(item_id)
        if not item_id in I_te[usr_id]:
            I_te[usr_id].append(item_id)

    
    return r, I_u, I, U, r_te, I_te, U_te, I_re

def Pre_u(k, I_re_u, I_te_u):
    count = 0.0
    for i in range(k):
        if I_re_u[i] in I_te_u:
            count += 1
    return count / k

def Pre(k, U_te, I_re, I_te):
    count = 0.0
    len_u = len(U_te)
    for u in U_te:
        count += Pre_u(k, I_re[u], I_te[u])
    return count / len_u

def Rec_u(k, I_re_u, I_te_u):
    len_I_te_u = len(I_te_u)
    count = 0
    for i in range(k):  
        if I_re_u[i] in I_te_u:
            count += 1
    return count / len_I_te_u

def Rec(k, U_te, I_re, I_te):
    count = 0.0
    len_u = len(U_te)
    for u in U_te:
        count += Rec_u(k, I_re[u], I_te[u])
    return count / len_u

def F1_u(k, I_re_u, I_te_u):
    pre_u = Pre_u(k, I_re_u, I_te_u)
    rec_u = Rec_u(k, I_re_u, I_te_u)
    if pre_u == 0 and rec_u == 0:
        return 0
    return 2 * pre_u * rec_u / (pre_u + rec_u)

def F1(k, U_te, I_re, I_te):
    count = 0.0
    len_u = len(U_te)
    for u in U_te:
        count += F1_u(k, I_re[u], I_te[u])
    return count / len_u

def DCG_u(k, I_re_u, I_te_u):
    total_score = 0.0
    for i in range(k):
        if I_re_u[i] in I_te_u:
            # l = i + 1
            total_score += 1 / math.log(i+1 + 1)
    return total_score / k

def Z_u(k, I_re_u, I_te_u):
    best_score = 0.0
    for i in range(k):
        best_score += 1 / math.log(i+1 + 1)
    return best_score / k

def NDCG_u(k, I_re_u, I_te_u):
    return DCG_u(k, I_re_u, I_te_u) / Z_u(k, I_re_u, I_te_u)

def NDCG(k, U_te, I_re, I_te):
    count = 0.0
    len_u = len(U_te)
    for u in U_te:
        # 因为考虑了相对位置, 所以k应该小于等于测试集长度
        _k = min(len(I_te[u]), k)
        count += NDCG_u(_k, I_re[u], I_te[u])
    return count / len_u

def one_call_u(k, I_re_u, I_te_u):
    for i in range(k):
        if I_re_u[i] in I_te_u:
            return 1
    return 0

def one_call(k, U_te, I_re, I_te):
    count = 0.0
    len_u = len(U_te)
    for u in U_te:
        count += one_call_u(k, I_re[u], I_te[u])
    return count / len_u

def RR_u(I_re_u, I_te_u):
    for i in range(len(I_re_u)):
        if I_re_u[i] in I_te_u:
            return  1 / (i + 1)
    return 0

def MRR(U_te, I_re, I_te):
    count = 0.0
    len_u = len(U_te)
    for u in U_te:
        count += RR_u(I_re[u], I_te[u])
    return count / len_u

def AP_u(I_re_u, I_te_u):
    len_I_te_u = len(I_te_u)
    precision = 0.0
    for i in range(len_I_te_u):
        if not I_te_u[i] in I_re_u:
            continue
        p_ui = I_re_u.index(I_te_u[i]) + 1
        count = 1.0
        for j in range(len_I_te_u):
            if not I_te_u[j] in I_re_u:
                continue
            p_uj = I_re_u.index(I_te_u[j]) + 1
            if p_uj < p_ui:
                count += 1
        precision += count / p_ui
    # len_I_te_u should not be zero
    return precision / len_I_te_u

def MAP(U_te, I_re, I_te):
    count = 0.0
    len_u = len(U_te)
    for u in U_te:
        count += AP_u(I_re[u], I_te[u])
    return count / len_u

def RP_u(I_re_u, I_te_u):
    len_I_te_u = len(I_te_u)
    RP = 0.0
    for i in range(len_I_te_u):
        if not I_te_u[i] in I_re_u:
            continue
        p_ui = I_re_u.index(I_te_u[i]) + 1
        RP += p_ui / len(I_re_u)
    # len_I_te_u should not be zero
    return RP / len_I_te_u

def ARP(U_te, I_re, I_te):
    count = 0.0
    len_u = len(U_te)
    for u in U_te:
        count += RP_u(I_re[u], I_te[u])
    return count / len_u

def AUC_u(u, U_te, R, R_te, I, I_re_u):
    length_j = len(I) - len(R[u]) - len(R_te[u])
    # length of i * length of j
    len_R_te_u = len(R_te[u]) * (length_j)
    AUC_count = 0.0
    p_uis = []
    # 因为取相对位置, +1可以同时忽略
    for i in R_te[u]:
        p_ui = I_re_u.index(i) + 1
        p_uis.append(p_ui)
    
    for p_ui in p_uis:
        AUC_count += length_j - (p_ui - (p_uis.index(p_ui) + 1))

    return  AUC_count / len_R_te_u

def AUC_u_directly(u, U_te, R, R_te, I, I_re_u):
    len_R_te_u = 0.0
    AUC_count = 0.0
    for i in R_te[u]:
        p_ui = I_re_u.index(i) + 1
        for j in I:
            if j in R[u] or j in R_te[u]:
                continue
            len_R_te_u += 1
            p_uj = I_re_u.index(j) + 1
            if p_ui < p_uj:
                AUC_count += 1
    return  AUC_count / len_R_te_u

def AUC(U_te, R, R_te, I, I_re):
    count = 0.0
    len_u = len(U_te)
    for u in U_te:
        count += AUC_u(u, U_te, R, R_te, I, I_re[u])
    return count / len_u

def main():
    k = 5
    r, I_u, I, U, r_te, I_te, U_te, I_re = initialization()

    pre_score = Pre(k, U_te, I_re, I_te)
    print("Pre@" + str(k) + ": " + str(pre_score))

    rec_score = Rec(k, U_te, I_re, I_te)
    print("Rec@" + str(k) + ": " + str(rec_score))

    F1_score = F1(k, U_te, I_re, I_te)
    print("F1@" + str(k) + ": " + str(F1_score))

    NDCG_score = NDCG(k, U_te, I_re, I_te)
    print("NDCG@" + str(k) + ": " + str(NDCG_score))

    one_call_score = one_call(k, U_te, I_re, I_te)
    print("1-call@" + str(k) + ": " + str(one_call_score))

    MRR_score = MRR(U_te, I_re, I_te)
    print("MRR: " + str(MRR_score))

    MAP_score = MAP(U_te, I_re, I_te)
    print("MAP: " + str(MAP_score))
    
    ARP_score = ARP(U_te, I_re, I_te)
    print("ARP: " + str(ARP_score))

    AUC_score = AUC(U_te, r, r_te, I, I_re)
    print("AUC(restruct): " + str(AUC_score))

    return

if __name__ == '__main__':
    main()
