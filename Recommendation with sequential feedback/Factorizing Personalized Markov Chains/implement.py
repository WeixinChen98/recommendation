import numpy as np
import pandas as pd
import math
import random
import sys
sys.path.insert(1, '../../utils')
from ranking_evaluation import Pre, Rec, NDCG

def preprocess_data(data, n, m):
    # count rating times of user and item
    user_rating_count = [0] * (n + 1)
    item_rating_count = [0] * (m + 1)
    for index, row in data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        user_rating_count[usr_id] += 1
        item_rating_count[item_id] += 1
    
    U = []
    I = []
    for usr_id in range(1, n + 1):
        if user_rating_count[usr_id] >= 5:
            U.append(usr_id)
        
    for item_id in range(1, m + 1):
        if item_rating_count[item_id] >= 5:
            I.append(item_id)

    
    # I_u with timestamp
    I_u_t = [{} for i in range(n + 1)]
    for index, row in data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        timestamp = row[3]
        if usr_id in U and item_id in I:            
            I_u_t[usr_id][item_id] = timestamp

    P = []
    I_u = [[] for i in range(n + 1)]
    S_u = [[] for i in range(n + 1)]
    I_te = [[] for i in range(n + 1)]
    for u in U:
        tmp_list = sorted(I_u_t[u].items(), key=lambda item: item[1])
        last_timestamp = tmp_list[-1][1]
        for (k, v) in tmp_list:
            if v != last_timestamp:
                P.append((u, k))
                S_u[u].append(k)
                I_u[u].append(k)
            else:
                I_te[u].append(k)

    return  U, I, P, I_u, S_u, I_te

# Eq.(1)
def prediction(U_u, V_i_T, P_i, Q_i_T):
    return (U_u @ V_i_T + P_i @ Q_i_T)[0, 0]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# without validation
def FPMC(n, m, d, U, I, P, I_u, S_u, U_u, V_i, P_i, Q_i, T, gamma, alpha_u, alpha_v, alpha_p, alpha_q):
    # training
    for t_trainning in range(T):
        print(t_trainning)

        # select (u, i) in random order
        random.shuffle(P)

        for (u, i) in P:
            t = S_u[u].index(i)

            # from the second item to the second last item (the last item for testing)
            if t == 0:
                continue
            
            j = random.sample(set(I) - set(I_u[u]), k = 1)

            U_u_ = U_u[u].reshape(1, d)
            V_i_ = V_i[i].reshape(1, d)
            V_j_ = V_i[j].reshape(1, d)
            V_i_T_ = V_i[i].reshape(d, 1)
            V_j_T_ = V_i[j].reshape(d, 1)
            P_i_ = P_i[S_u[u][t-1]].reshape(1, d)
            Q_i_ = Q_i[i].reshape(1, d)
            Q_j_ = Q_i[j].reshape(1, d)
            Q_i_T_ = Q_i[i].reshape(d, 1)
            Q_j_T_ = Q_i[j].reshape(d, 1)

            # precition on (u, i) and (u, j)
            r_p_ui = prediction(U_u_, V_i_T_, P_i_, Q_i_T_)
            r_p_uj = prediction(U_u_, V_j_T_, P_i_, Q_j_T_)

            e_ui = sigmoid(r_p_uj - r_p_ui)
            delta_U_u = alpha_u * U_u_ - e_ui * (V_i_ - V_j_)
            delta_V_i = alpha_v * V_i_ - e_ui * U_u_
            delta_V_j = alpha_v * V_j_ + e_ui * U_u_
            delta_Q_i = alpha_q * Q_i_ - e_ui * P_i_
            delta_Q_j = alpha_q * Q_j_ + e_ui * P_i_
            delta_P_i = alpha_p * P_i_ - e_ui * (Q_i_ - Q_j_)

            U_u[u] -= gamma * delta_U_u.reshape(d)
            V_i[i] -= gamma * delta_V_i.reshape(d)
            V_i[j] -= gamma * delta_V_j.reshape(d)
            Q_i[i] -= gamma * delta_Q_i.reshape(d)
            Q_i[j] -= gamma * delta_Q_j.reshape(d)
            P_i[S_u[u][t-1]] -= gamma * delta_P_i.reshape(d)

    # prediction matrix
    r_p = [[] for i in range(n + 1)]
    for u in U:
        r_p[u] = [-100.0] * (m + 1) 
        for j in set(I) - set(I_u[u]):
            U_u_ = U_u[u].reshape(1, d)
            V_j_T_ = V_i[j].reshape(d, 1)
            P_i_ = P_i[S_u[u][-1]].reshape(1, d)
            Q_j_T_ = Q_i[j].reshape(d, 1)
            r_p[u][j] = prediction(U_u_, V_j_T_, P_i_, Q_j_T_)


    # ranked recommandation matrix
    I_re = [[] for i in range(n + 1)]
    for u in U:
        I_re[u] = np.argsort(np.array(r_p[u])).tolist()
        I_re[u].reverse()
    
    return I_re


def main():
    n = 943
    m = 1682
    data_file_path = '../../dataset/ml-100k/u.data'
    data = pd.read_csv(data_file_path, delim_whitespace=True, index_col=False, header=None) 

    # tradeoff parameters
    # set first T = 1 to debug
    Ts = [100, 500, 1000]
    d = 20
    gamma = 0.01
    alpha_u = alpha_v = alpha_p = alpha_q = 0.001 # 0.1, 0.01, 0.001
    k = 20

    # initialize parameters
    U, I, P, I_u, S_u, I_te = preprocess_data(data, n , m)
    U_u = (np.random.random((n + 1, d)) - 0.5) * 0.01
    V_i = (np.random.random((m + 1, d)) - 0.5) * 0.01
    P_i = (np.random.random((m + 1, d)) - 0.5) * 0.01
    Q_i = (np.random.random((m + 1, d)) - 0.5) * 0.01

    print('FPMC')
    print('d = ' + str(d))
    print('gamma = ' + str(gamma))
    print('alpha_u = ' + str(alpha_u))
    print('alpha_v = ' + str(alpha_v))
    print('alpha_p = ' + str(alpha_p))
    print('alpha_q = ' + str(alpha_q))
    print('k = ' + str(k))

    # training 
    T_trained = 0
    for T_objective in Ts:
        T = T_objective - T_trained
        print('T = ' + str(T_objective))

        I_re = FPMC(n, m, d, U, I, P, I_u, S_u, U_u, V_i, P_i, Q_i, T, gamma, alpha_u, alpha_v, alpha_p, alpha_q)

        pre_score = Pre(k, U, I_re, I_te)
        print("Pre@" + str(k) + ": " + str(pre_score))

        rec_score = Rec(k, U, I_re, I_te)
        print("Rec@" + str(k) + ": " + str(rec_score))

        ndcg_score = NDCG(k, U, I_re, I_te)
        print("NDCG@" + str(k) + ": " + str(ndcg_score))

        T_trained = T_objective


if __name__ == '__main__':
    main()