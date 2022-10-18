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
    
    U = (np.random.random((n,d)) - 0.5) * 0.01
    V = (np.random.random((m,d)) - 0.5) * 0.01

    return r, I_u, I_u_com, I, r_te, I_te, U_te, b_usr, b_item, U, V

def prediction(b_u, b_i, U_u, V_i_T):
    return b_u + b_i + U_u @ V_i_T 

def MFLogLoss(I, I_u, I_u_com, r, b_usr, b_item, U, V, alpha_u, alpha_v, beta_u, beta_v, gamma, T, d, rho):
    # training
    print('training... ')
    A_length = rho * training_data_length
    _P = [(u, i) for u, items in I_u_com.items() for i in items]
    for t in range(T):
        print(t)
        A = random.sample(_P, k = A_length)
        PAunion = A + [(u, i) for u, items in I_u.items() for i in items]
        random.shuffle(PAunion)
        for (u, i) in PAunion:
            # ?
            if u not in I_u:
                continue

            r_ui = 1
            if (u, i) in A:
                r_ui = -1
            
            U_u = U[u - 1].reshape(1, d)
            V_i = V[i - 1].reshape(1, d)
            b_u = b_usr[u]
            b_i = b_item[i]
            V_i_T = V[i - 1].reshape(d, 1)

            # prediction
            r_p_ui = prediction(b_u, b_i, U_u, V_i_T)

            # calculate gradient
            e_ui = r_ui / ( 1 +  math.exp(r_ui * r_p_ui))
            delta_b_u = -e_ui + beta_u * b_u
            delta_b_i = -e_ui + beta_v * b_i
            delta_V_i = -e_ui * U_u + alpha_v * V_i
            delta_U_u = -e_ui * V_i + alpha_u * U_u

            # update parameters
            b_item[i] -= gamma *  delta_b_i
            b_usr[u] -= gamma * delta_b_u
            V[i - 1] -= gamma * delta_V_i.reshape(d)
            U[u - 1] -= gamma * delta_U_u.reshape(d)
            
    # prediction matrix
    print('prediction generating... ')
    r_pre = {}
    for u in range(1, n + 1):
        r_pre.setdefault(u)
        if r_pre[u] == None:
            r_pre[u] = {}
        for i in range(1, m + 1):
            b_u = b_usr[u]
            b_i = b_item[i]
            U_u = U[u - 1].reshape(1, d)
            V_i_T = V[i - 1].reshape(d, 1)
            r_pre[u][i] = prediction(b_u, b_i, U_u, V_i_T)

    # ranked recommandation matrix
    print('ranking... ')
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
    alpha_u = alpha_v = beta_u = beta_v = 0.001
    gamma = 0.01
    rho = 3
    d = 20

    r_usr, I_u, I_u_com, I, r_te, I_te, U_te, b_usr, b_item, U, V = initialization(d)
    
    T = 100
    print("T = " + str(T))
    I_re = MFLogLoss(I, I_u, I_u_com, r_usr, b_usr, b_item, U, V, alpha_u, alpha_v, beta_u, beta_v, gamma, T, d, rho)

    k = 5
    pre_score = Pre(k, U_te, I_re, I_te)
    print("Pre@" + str(k) + ": " + str(pre_score))

    rec_score = Rec(k, U_te, I_re, I_te)
    print("Rec@" + str(k) + ": " + str(rec_score))


if __name__ == '__main__':
    main()