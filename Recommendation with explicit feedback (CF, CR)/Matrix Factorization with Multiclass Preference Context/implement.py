import math
import pandas as pd
import numpy as np
from numpy import mat

n = 943
m = 1682
data = pd.read_csv('../../dataset/ml-100k/u.data', delim_whitespace=True, index_col=False, header=None)
datas = np.array_split(data, 5)

def MF_MPC_initialization(d, training_data):
    training_data_length = training_data.index.size
    r_sum = training_data.sum(axis=0)[2]
    r_mean = r_sum / training_data_length

    r_usr = {}
    for index, row in training_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rate = row[2]
        r_usr.setdefault(usr_id)
        if r_usr[usr_id] == None:
            r_usr[usr_id] = {}
        r_usr[usr_id][item_id] = rate
    

    r_item = {}
    for index, row in training_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rate = row[2]
        r_item.setdefault(item_id)
        if r_item[item_id] == None:
            r_item[item_id] = {}
        r_item[item_id][usr_id] = rate
    

    b_usr = {}
    b_item = {}
    for usr_id in range(1, n + 1):
        b = 0
        if not usr_id in r_usr:
            b_usr[usr_id] = b
            continue
        
        b_len = len(r_usr[usr_id])
        for item_id in r_usr[usr_id]:
            b += r_usr[usr_id][item_id] - r_mean

        b /= b_len
        b_usr[usr_id] = b

    for item_id in range(1, m + 1):
        b = 0
        if not item_id in r_item:
            b_item[item_id] = 0
            continue

        b_len = len(r_item[item_id])
        for usr_id in r_item[item_id]:
            b += r_item[item_id][usr_id] - r_mean
        
        b /= b_len
        b_item[item_id] = b

    I_r_u = {}
    for index, row in training_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rate = row[2]
        I_r_u.setdefault(usr_id)
        if I_r_u[usr_id] == None:
            I_r_u[usr_id] = {}
        I_r_u[usr_id][item_id] = rate

    U = (np.random.random((n,d)) - 0.5) * 0.01
    V = (np.random.random((m,d)) - 0.5) * 0.01
    M_1 = (np.random.random((m,d)) - 0.5) * 0.01
    M_2 = (np.random.random((m,d)) - 0.5) * 0.01
    M_3 = (np.random.random((m,d)) - 0.5) * 0.01
    M_4 = (np.random.random((m,d)) - 0.5) * 0.01
    M_5 = (np.random.random((m,d)) - 0.5) * 0.01
    return r_mean, b_usr, b_item, U, V, M_1, M_2, M_3, M_4, M_5, I_r_u


def MF_MPC_prediction(mu, b_u, b_i, U_u, V_i_T, U_MPC_u):
    prediction = mu + b_u + b_i + U_u @ V_i_T + U_MPC_u @ V_i_T
    if prediction > 5: prediction = 5
    if prediction < 1: prediction = 1
    return prediction

def MF_MPC(alpha_u, alpha_v, alpha_w, beta_u, beta_v, gamma, d, T, training_data, testing_data):
    training_data_length = training_data.index.size
    testing_data_length = testing_data.index.size
    mu, b_u, b_i, U, V, M_1, M_2, M_3, M_4, M_5, I_r_u = MF_MPC_initialization(d, training_data)

    # training
    for t in range(0, T):
        # print(t)
        for index, row in training_data.iterrows():
            usr_id = row[0]
            item_id = row[1]
            rating = row[2]
            U_MPC_u = np.zeros([1, d])
            U_MPC_u_1 = np.zeros([1, d])
            U_MPC_u_2 = np.zeros([1, d])
            U_MPC_u_3 = np.zeros([1, d])
            U_MPC_u_4 = np.zeros([1, d])
            U_MPC_u_5 = np.zeros([1, d])
            l_1 = l_2 = l_3 = l_4 = l_5 = 0
            for v_i in I_r_u[usr_id]:
                if v_i == item_id: continue
                if I_r_u[usr_id][v_i] == 1:
                    U_MPC_u_1 += M_1.reshape(m, d)[v_i - 1]
                    l_1 += 1
                elif I_r_u[usr_id][v_i] == 2:
                    U_MPC_u_2 += M_2.reshape(m, d)[v_i - 1]
                    l_2 += 1
                elif I_r_u[usr_id][v_i] == 3:
                    U_MPC_u_3 += M_3.reshape(m, d)[v_i - 1]
                    l_3 += 1
                elif I_r_u[usr_id][v_i] == 4:
                    U_MPC_u_4 += M_4.reshape(m, d)[v_i - 1]
                    l_4 += 1
                elif I_r_u[usr_id][v_i] == 5:
                    U_MPC_u_5 += M_5.reshape(m, d)[v_i - 1]
                    l_5 += 1
                
            if l_1 > 0:
                U_MPC_u += U_MPC_u_1 / (l_1 ** 0.5)
            if l_2 > 0:
                U_MPC_u += U_MPC_u_2 / (l_2 ** 0.5)
            if l_3 > 0:
                U_MPC_u += U_MPC_u_3 / (l_3 ** 0.5)
            if l_4 > 0:
                U_MPC_u += U_MPC_u_4 / (l_4 ** 0.5)
            if l_5 > 0:
                U_MPC_u += U_MPC_u_5 / (l_5 ** 0.5)

            prediction = MF_MPC_prediction(mu, b_u[usr_id], b_i[item_id], U[usr_id - 1].reshape(1, d), V[item_id - 1].reshape(d, 1), U_MPC_u)
            e_ui = rating - prediction
            delta_mu = -e_ui
            delta_b_u = -e_ui + beta_u * b_u[usr_id]
            delta_b_i = -e_ui + beta_v * b_i[item_id]
            delta_U_u = -e_ui * V[item_id - 1] + alpha_u * U[usr_id - 1]
            delta_V_i = -e_ui * (U[usr_id - 1] + U_MPC_u) + alpha_v * V[item_id - 1]


            delta_M_i_v_1 = delta_M_i_v_2 = delta_M_i_v_3 = delta_M_i_v_4 = delta_M_i_v_5 = np.zeros([m, d])
            for v_i in I_r_u[usr_id]:
                if v_i == item_id: continue
                if I_r_u[usr_id][v_i] == 1:
                    delta_M_i_v_1[v_i - 1] = -e_ui / (l_1 ** 0.5) * V[item_id - 1] + alpha_w * M_1.reshape(m, d)[v_i - 1]
                elif I_r_u[usr_id][v_i] == 2:
                    delta_M_i_v_2[v_i - 1] = -e_ui / (l_2 ** 0.5) * V[item_id - 1] + alpha_w * M_2.reshape(m, d)[v_i - 1]
                elif I_r_u[usr_id][v_i] == 3:
                    delta_M_i_v_3[v_i - 1] = -e_ui / (l_3 ** 0.5) * V[item_id - 1] + alpha_w * M_3.reshape(m, d)[v_i - 1]
                elif I_r_u[usr_id][v_i] == 4:
                    delta_M_i_v_4[v_i - 1] = -e_ui / (l_4 ** 0.5) * V[item_id - 1] + alpha_w * M_4.reshape(m, d)[v_i - 1]
                elif I_r_u[usr_id][v_i] == 5:
                    delta_M_i_v_5[v_i - 1] = -e_ui / (l_5 ** 0.5) * V[item_id - 1] + alpha_w * M_5.reshape(m, d)[v_i - 1]
                
            mu -= gamma * delta_mu
            b_u[usr_id] -= gamma *  delta_b_u
            b_i[item_id] -= gamma * delta_b_i
            U[usr_id - 1] -= gamma * delta_U_u.reshape(d)
            V[item_id - 1] -= gamma * delta_V_i.reshape(d)
            for v_i in I_r_u[usr_id]:
                if v_i == item_id: continue
                if I_r_u[usr_id][v_i] == 1:
                    M_1[v_i - 1] -= gamma * delta_M_i_v_1[v_i - 1]
                elif I_r_u[usr_id][v_i] == 2:
                    M_2[v_i - 1] -= gamma * delta_M_i_v_2[v_i - 1]
                elif I_r_u[usr_id][v_i] == 3:
                    M_3[v_i - 1] -= gamma * delta_M_i_v_3[v_i - 1]
                elif I_r_u[usr_id][v_i] == 4:
                    M_4[v_i - 1] -= gamma * delta_M_i_v_4[v_i - 1]
                elif I_r_u[usr_id][v_i] == 5:
                    M_5[v_i - 1] -= gamma * delta_M_i_v_5[v_i - 1]

        gamma *= 0.9


    bias_sum = 0
    square_bias_sum = 0
    for index, row in testing_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rating = row[2]
        U_MPC_u = np.zeros([1, d])
        U_MPC_u_1 = np.zeros([1, d])
        U_MPC_u_2 = np.zeros([1, d])
        U_MPC_u_3 = np.zeros([1, d])
        U_MPC_u_4 = np.zeros([1, d])
        U_MPC_u_5 = np.zeros([1, d])
        l_1 = l_2 = l_3 = l_4 = l_5 = 0
        for v_i in I_r_u[usr_id]:
            if v_i == item_id: continue
            if I_r_u[usr_id][v_i] == 1:
                U_MPC_u_1 += M_1.reshape(m, d)[v_i - 1]
                l_1 += 1
            elif I_r_u[usr_id][v_i] == 2:
                U_MPC_u_2 += M_2.reshape(m, d)[v_i - 1]
                l_2 += 1
            elif I_r_u[usr_id][v_i] == 3:
                U_MPC_u_3 += M_3.reshape(m, d)[v_i - 1]
                l_3 += 1
            elif I_r_u[usr_id][v_i] == 4:
                U_MPC_u_4 += M_4.reshape(m, d)[v_i - 1]
                l_4 += 1
            elif I_r_u[usr_id][v_i] == 5:
                U_MPC_u_5 += M_5.reshape(m, d)[v_i - 1]
                l_5 += 1
            
        if l_1 > 0:
            U_MPC_u += U_MPC_u_1 / (l_1 ** 0.5)
        if l_2 > 0:
            U_MPC_u += U_MPC_u_2 / (l_2 ** 0.5)
        if l_3 > 0:
            U_MPC_u += U_MPC_u_3 / (l_3 ** 0.5)
        if l_4 > 0:
            U_MPC_u += U_MPC_u_4 / (l_4 ** 0.5)
        if l_5 > 0:
            U_MPC_u += U_MPC_u_5 / (l_5 ** 0.5)

        r_ui_prediction = MF_MPC_prediction(mu, b_u[usr_id], b_i[item_id], U[usr_id - 1].reshape(1, d), V[item_id - 1].reshape(d, 1), U_MPC_u)
        
        bias_sum += abs(r_ui_prediction - rating)
        square_bias_sum += (r_ui_prediction - rating) ** 2

    MAE = bias_sum / testing_data_length
    RMSE = math.sqrt(square_bias_sum / testing_data_length)
    
    return MAE, RMSE

def main():
    alpha_u = alpha_v = alpha_w = beta_u = beta_v = MFMPC_lambda = 0.001
    gamma = 0.01
    T = 50
    d = 20
    

    MFMPC_MAE = 0
    MFMPC_RMSE = 0
    print("MF-MPC")

    for i in range(5):
        print(i)
        training_data = pd.concat(datas[0:i] + datas[i+1:5])
        testing_data = datas[i]
        MAE, RMSE = MF_MPC(alpha_u, alpha_v, alpha_w, beta_u, beta_v, gamma, d, T, training_data, testing_data)
        print(MAE)
        print(RMSE)
        MFMPC_MAE += MAE
        MFMPC_RMSE += RMSE

    MFMPC_MAE /= 5.0
    MFMPC_RMSE /= 5.0
    print(MFMPC_MAE)
    print(MFMPC_RMSE)
    return

if __name__ == '__main__':
    main()