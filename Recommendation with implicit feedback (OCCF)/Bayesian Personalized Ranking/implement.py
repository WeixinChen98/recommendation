import math
import random
import numpy as np
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

def initialization(d):
    mu = training_data_length / n / m

    r = {}
    r_item = {}
    I_u = {}
    I = []
    for index, row in training_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        
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
    
    b_item = {}
    for item_id in range(1, m + 1):
        if not item_id in r_item:
            b_item[item_id] = -mu
            continue
        b_item[item_id] = len(r_item[item_id]) / n - mu

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
    
    U = (np.random.random((n,d)) - 0.5) * 0.01
    V = (np.random.random((m,d)) - 0.5) * 0.01

    return r, I_u, I, r_te, I_te, U_te, b_item, U, V

def prediction(b_i, U_u, V_i_T):
    return b_i + U_u @ V_i_T 

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def BPR(I, I_u, r, b_item, U, V, alpha_u, alpha_v, beta_v, gamma, T, d):
    # training
    for t in range(T):
        print(t)
        ui = [(k, v) for k,vs in I_u.items() for v in vs]
        random.shuffle(ui)
        for (u, i) in ui: 
            j = random.choice(list(set(I) - set(I_u[u])))

            # notation
            U_u = U[u - 1].reshape(1, d)
            V_i = V[i - 1].reshape(1, d)
            V_j = V[j - 1].reshape(1, d)
            b_i = b_item[i]
            b_j = b_item[j]
            V_i_T = V[i - 1].reshape(d, 1)
            V_j_T = V[j - 1].reshape(d, 1)

            # prediction
            r_p_ui = prediction(b_i, U_u, V_i_T)
            r_p_uj = prediction(b_j, U_u, V_j_T)

            # calculate gradient
            e_ui = sigmoid(r_p_uj - r_p_ui)
            delta_U_u = -e_ui * (V_i - V_j) + alpha_u * U_u
            delta_V_i = -e_ui * U_u + alpha_v * V_i
            delta_V_j = -e_ui * (-U_u) + alpha_v * V_j
            delta_b_i = -e_ui + beta_v * b_i
            delta_b_j = -e_ui * (-1) + beta_v * b_j

            # update parameters
            U[u - 1] -= gamma * delta_U_u.reshape(d)
            V[i - 1] -= gamma * delta_V_i.reshape(d)
            V[j - 1] -= gamma * delta_V_j.reshape(d)
            b_item[i] -= gamma *  delta_b_i
            b_item[j] -= gamma * delta_b_j

    # prediction matrix
    r_pre = {}
    for u in range(1, n + 1):
        r_pre.setdefault(u)
        if r_pre[u] == None:
            r_pre[u] = {}
        U_u = U[u - 1].reshape(1, d)
        for i in range(1, m + 1):
            b_i = b_item[i]
            V_i_T = V[i - 1].reshape(d, 1)
            r_pre[u][i] = prediction(b_i, U_u, V_i_T)

    # ranked recommandation matrix
    I_re = {}
    for u in range(1, n + 1):
        I_re.setdefault(u)
        if I_re[u] == None:
            I_re[u] = []
        I_re_u = list(dict(sorted(r_pre[u].items(), key=lambda item: item[1], reverse=True)).keys())
        if not u in I_u:
            I_re[u] = I_re_u
        else:
            I_re[u] = [i for i in I_re_u if i not in I_u[u]]

    return I_re


def main():
    alpha_u = alpha_v = beta_v = 0.01
    gamma = 0.01
    T = 500
    d = 20

    r_usr, I_u, I, r_te, I_te, U_te, b_item, U, V = initialization(d)

    I_re = BPR(I, I_u, r_usr, b_item, U, V, alpha_u, alpha_v, beta_v, gamma, T, d)

    k = 5
    pre_score = Pre(k, U_te, I_re, I_te)
    print("Pre@" + str(k) + ": " + str(pre_score))

    rec_score = Rec(k, U_te, I_re, I_te)
    print("Rec@" + str(k) + ": " + str(rec_score))


if __name__ == '__main__':
    main()