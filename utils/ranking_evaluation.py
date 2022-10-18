from math import log

def Pre_u(k, I_re_u, I_te_u):
    count = 0.0
    for i in range(k):
        if I_re_u[i] in I_te_u:
            count += 1
    return count / k

def Pre(k, U_te, I_re, I_te):
    count = 0.0
    for u in U_te:
        count += Pre_u(k, I_re[u], I_te[u])
    return count / len(U_te)

def Rec_u(k, I_re_u, I_te_u):
    count = 0.0
    for i in range(k):  
        if I_re_u[i] in I_te_u:
            count += 1
    return count / len(I_te_u)

def Rec(k, U_te, I_re, I_te):
    count = 0.0
    for u in U_te:
        count += Rec_u(k, I_re[u], I_te[u])
    return count / len(U_te)

def F1_u(k, I_re_u, I_te_u):
    pre_u = Pre_u(k, I_re_u, I_te_u)
    rec_u = Rec_u(k, I_re_u, I_te_u)
    if pre_u == 0 and rec_u == 0:
        return 0
    return 2 * pre_u * rec_u / (pre_u + rec_u)

def F1(k, U_te, I_re, I_te):
    count = 0.0
    for u in U_te:
        count += F1_u(k, I_re[u], I_te[u])
    return count / len(U_te)

def DCG_u(k, I_re_u, I_te_u):
    total_score = 0.0
    for i in range(k):
        if I_re_u[i] in I_te_u:
            # l = i + 1
            total_score += 1 / log(i+1 + 1)
    return total_score

def Z_u(k, I_re_u, I_te_u):
    best_score = 0.0
    for i in range(k):
        best_score += 1 / log(i+1 + 1)
    return best_score

def NDCG_u(k, I_re_u, I_te_u):
    return DCG_u(k, I_re_u, I_te_u) / Z_u(k, I_re_u, I_te_u)

def NDCG(k, U_te, I_re, I_te):
    count = 0.0
    for u in U_te:
        _k = min(len(I_re[u]), k)
        count += NDCG_u(_k, I_re[u], I_te[u])
    return count / len(U_te)

def one_call_u(k, I_re_u, I_te_u):
    for i in range(k):
        if I_re_u[i] in I_te_u:
            return 1
    return 0

def one_call(k, U_te, I_re, I_te):
    count = 0.0
    for u in U_te:
        count += one_call_u(k, I_re[u], I_te[u])
    return count / len(U_te)

def RR_u(I_re_u, I_te_u):
    for i in range(len(I_re_u)):
        if I_re_u[i] in I_te_u:
            return  1 / (i + 1)
    return 0

def MRR(U_te, I_re, I_te):
    count = 0.0
    for u in U_te:
        count += RR_u(I_re[u], I_te[u])
    return count / len(U_te)

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
    for u in U_te:
        count += AP_u(I_re[u], I_te[u])
    return count / len(U_te)

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
    for u in U_te:
        count += RP_u(I_re[u], I_te[u])
    return count / len(U_te)

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
    for u in U_te:
        count += AUC_u(u, U_te, R, R_te, I, I_re[u])
    return count / len(U_te)


