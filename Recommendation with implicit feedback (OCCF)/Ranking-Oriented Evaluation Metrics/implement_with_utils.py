import sys
sys.path.insert(1, '../../utils')
import math
import pandas as pd
from ranking_evaluation import Pre, Rec, F1, NDCG, one_call, MRR, MAP, ARP, AUC

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
