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

    I_u_com = {}
    for u in range(1, n + 1):
        I_u_com.setdefault(u)
        if I_u_com[u] == None:
            I_u_com[u] = []
        if not u in I_u:
            I_u_com[u] = range(1, m + 1)
            continue
        for i in range(1, m + 1):
            if not i in I_u[u]:
                I_u_com[u].append(i)
    
    b_usr = {}
    for usr_id in range(1, n + 1):
        if not usr_id in r:
            b_usr[usr_id] = -mu
            continue
        b_usr[usr_id] = len(r[usr_id]) / m - mu

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
    
    W = (np.random.random((m,d)) - 0.5) * 0.01
    V = (np.random.random((m,d)) - 0.5) * 0.01

    return r, I_u, I_u_com, I, r_te, I_te, U_te, b_usr, b_item, W, V

def prediction(b_i, U_minusi_i_u, V_i_T):
    return b_i + U_minusi_i_u @ V_i_T 

def FISM_auc(I, I_u, I_u_com, r, b_i, W, V, alpha, alpha_v, alpha_w, beta_v, gamma, T, d, rho):
    # training
    print("training... ")
    A_u_length = rho
    for t in range(T):
        print(t)
        for u, items in I_u.items():
            for i in items:
                minus_i_len = len(I_u[u]) - 1                
                U_minus_i_u = np.zeros([1, d])
                if minus_i_len > 0:
                    for minus_i in I_u[u]:
                        if minus_i == i:
                            continue
                        U_minus_i_u += W[minus_i - 1].reshape(1, d)
                    U_minus_i_u /= (minus_i_len ** alpha)

                i_len = len(I_u[u])
                U_i_u = np.zeros([1, d])
                for ii in I_u[u]:
                    U_i_u += W[ii - 1].reshape(1, d)
                U_i_u /= (i_len ** alpha)

                V_i = V[i - 1].reshape(1, d)
                V_i_T = V[i - 1].reshape(d, 1)
                W_i = W[i - 1].reshape(1, d)
                
                # prediction on (u, i)
                r_p_ui = prediction(b_i[i], U_minus_i_u, V_i_T)

                A_u = random.sample(set(I) - set(I_u[u]), k = A_u_length)
                delta_b_i = beta_v * b_i[i]
                delta_V_i = alpha_v * V_i

                delta_W_minus_i = np.zeros([m, d])
                delta_W_i = np.zeros([1, d])
                for ii in I_u[u]:
                    if ii != i:
                        delta_W_minus_i[ii - 1] = alpha_w * W[ii - 1].reshape(d)
                    else:
                        delta_W_i = alpha_w * W[ii - 1].reshape(1, d)

                for j in A_u:
                    V_j = V[j - 1].reshape(1, d)
                    V_j_T = V[j - 1].reshape(d, 1)

                    # prediction on (u, j)
                    r_p_uj = prediction(b_i[j], U_i_u, V_j_T)
                    e_uij = (1 - (r_p_ui - r_p_uj)) / A_u_length

                    delta_b_j = e_uij + beta_v * b_i[j]
                    delta_V_j = e_uij * U_i_u + alpha_v * V_j

                    delta_b_i += -e_uij
                    delta_V_i += -e_uij * U_minus_i_u

                    for ii in I_u[u]:
                        if ii != i:
                            delta_W_minus_i[ii - 1] += (-e_uij * (V_i / (minus_i_len ** alpha) - V_j / (i_len ** alpha))).reshape(d)
                        else:
                            delta_W_i += -e_uij * (-V_j) / (i_len ** alpha)

                    # update b_j and V_j
                    b_i[j] -= gamma * delta_b_j
                    V[j - 1] -= gamma * delta_V_j.reshape(d)

                # update b_i and V_i
                b_i[i] -= gamma * delta_b_i
                V[i - 1] -= gamma * delta_V_i.reshape(d)

                # update W_i and W_minus_i
                for ii in I_u[u]:
                    if ii != i:
                        W[ii - 1] -= gamma * delta_W_minus_i[ii - 1].reshape(d)
                    else:
                        W[ii - 1] -= gamma * delta_W_i.reshape(d)
        
    # prediction matrix
    print("prediction generating... ")
    r_pre = {}
    for u in range(1, n + 1):
        r_pre.setdefault(u)
        if r_pre[u] == None:
            r_pre[u] = {}
        for j in range(1, m + 1):
            if u in I_u and j in I_u[u]:
                continue

            U_i_u = np.zeros([1, d])
            if u in I_u:
                i_len = len(I_u[u])
                for i in I_u[u]:
                    U_i_u += W[i - 1].reshape(1, d)
                U_i_u /= (i_len ** alpha)

            V_j_T = V[j - 1].reshape(d, 1)
            r_pre[u][j] = prediction(b_i[j], U_i_u, V_j_T)

    # ranked recommandation matrix
    print("ranking... ")
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
    alpha = 0.5
    alpha_w = alpha_v = beta_v = 0.01
    gamma = 0.01
    # length of A_u
    rho = 3
    d = 20

    print("FISM_auc")

    r_usr, I_u, I_u_com, I, r_te, I_te, U_te, b_usr, b_i, W, V = initialization(d)
    
    # T = 100, 500, 1000
    Ts = [100, 400, 500]
    k = 5
    sum_T = 0
    for T in Ts:
        sum_T += T
        print("T = " + str(sum_T))
        I_re = FISM_auc(I, I_u, I_u_com, r_usr, b_i, W, V, alpha, alpha_v, alpha_w, beta_v, gamma, T, d, rho)

        pre_score = Pre(k, U_te, I_re, I_te)
        print("Pre@" + str(k) + ": " + str(pre_score))

        rec_score = Rec(k, U_te, I_re, I_te)
        print("Rec@" + str(k) + ": " + str(rec_score))


if __name__ == '__main__':
    main()