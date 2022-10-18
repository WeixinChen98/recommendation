import math
import pandas as pd

n = 943
m = 1682
training_data = pd.read_csv('../../dataset/ml-100k/u1.base.occf', delim_whitespace=True, index_col=False, header=None)
testing_data = pd.read_csv('../../dataset/ml-100k/u1.test.occf', delim_whitespace=True, index_col=False, header=None)
training_data_length = training_data.index.size
testing_data_length = testing_data.index.size


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

def solution(K):
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
    
    # Jaccard Index
    s_item = {}
    for k in range(1, m + 1):
        s_item.setdefault(k)
        if s_item[k] == None:
            s_item[k] = {}
        for j in range(1, m + 1):
            if (not k in r_item) or (not j in r_item):
                s_item[k][j] = 0
                continue
            s_item[k][j] = len(set(r_item[k]) & set(r_item[j])) / len(set(r_item[k]) | set(r_item[j]))
        s_item[k] = dict(sorted(s_item[k].items(), key=lambda item: item[1], reverse=True))
        

    s_usr = {}
    for k in range(1, n + 1):
        s_usr.setdefault(k)
        if s_usr[k] == None:
            s_usr[k] = {}
        for j in range(1, n + 1):
            if (not k in r) or (not j in r):
                s_usr[k][j] = 0
                continue
            s_usr[k][j] = len(set(r[k]) & set(r[j])) / len(set(r[k]) | set(r[j]))
        s_usr[k] = dict(sorted(s_usr[k].items(), key=lambda item: item[1], reverse=True))
    
    prediction_item_OCCF = {}
    for u in range(1, n + 1):
        prediction_item_OCCF.setdefault(u)
        if prediction_item_OCCF[u] == None:
            prediction_item_OCCF[u] = {}
        for j in range(1, m + 1):
            prediction_item_OCCF[u][j] = 0.0
            if not u in I_u:
                continue
            if j in I_u[u]:
                continue
            N_j = list(s_item[j].keys())[:K]
            for k in N_j:
                if not k in I_u[u]:
                    continue
                prediction_item_OCCF[u][j] += s_item[k][j]


    prediction_usr_OCCF = {}
    for u in range(1, n + 1):
        prediction_usr_OCCF.setdefault(u)
        if prediction_usr_OCCF[u] == None:
            prediction_usr_OCCF[u] = {}
        for j in range(1, m + 1):
            prediction_usr_OCCF[u][j] = 0.0
            if not j in r_item:
                continue
            if u in r_item[j]:
                continue
            N_u = list(s_usr[u].keys())[:K]
            for w in N_u:
                if not w in r_item[j]:
                    continue
                prediction_usr_OCCF[u][j] += s_usr[w][u]
    
    Hybrid = {}
    for u in range(1, n + 1):
        Hybrid.setdefault(u)
        if Hybrid[u] == None:
            Hybrid[u] = {}
        for j in range(1, m + 1):
            Hybrid[u][j] = 0.5 * prediction_item_OCCF[u][j] + 0.5 * prediction_usr_OCCF[u][j]
    
    item_rank = {}
    usr_rank = {}
    hybrid_rank = {}
    for u in range(1, n + 1):
        item_rank[u] = list(dict(sorted(prediction_item_OCCF[u].items(), key=lambda item: item[1], reverse=True)).keys())
        usr_rank[u] = list(dict(sorted(prediction_usr_OCCF[u].items(), key=lambda item: item[1], reverse=True)).keys())
        hybrid_rank[u] = list(dict(sorted(Hybrid[u].items(), key=lambda item: item[1], reverse=True)).keys())
    

    r_te = {}
    I_te = {}
    U_te = []
    for index, row in testing_data.iterrows():
        usr_id = row[0]
        item_id = row[1]

        if not usr_id in U_te:
            U_te.append(usr_id)

        r_te.setdefault(usr_id)
        I_te.setdefault(usr_id)

        if r_te[usr_id] == None:
            r_te[usr_id] = []
        if I_te[usr_id] == None:
            I_te[usr_id] = []

        if not item_id in r_te[usr_id]:
            r_te[usr_id].append(item_id)
        if not item_id in I_te[usr_id]:
            I_te[usr_id].append(item_id)
    
    return r, r_item, I_u, I, U, r_te, I_te, U_te, item_rank, usr_rank, hybrid_rank

def main():
    K = 50
    k = 5
    r_usr, r_item, I_u, I, U, r_te, I_te, U_te, item_rank, usr_rank, hybrid_rank = solution(K)

    print("item-based OCCF")
    pre_score = Pre(k, U_te, item_rank, I_te)
    print("Pre@" + str(k) + ": " + str(pre_score))

    rec_score = Rec(k, U_te, item_rank, I_te)
    print("Rec@" + str(k) + ": " + str(rec_score))

    print("user-based OCCF")
    pre_score = Pre(k, U_te, usr_rank, I_te)
    print("Pre@" + str(k) + ": " + str(pre_score))

    rec_score = Rec(k, U_te, usr_rank, I_te)
    print("Rec@" + str(k) + ": " + str(rec_score))

    print("Hybrid")
    pre_score = Pre(k, U_te, hybrid_rank, I_te)
    print("Pre@" + str(k) + ": " + str(pre_score))

    rec_score = Rec(k, U_te, hybrid_rank, I_te)
    print("Rec@" + str(k) + ": " + str(rec_score))


if __name__ == '__main__':
    main()