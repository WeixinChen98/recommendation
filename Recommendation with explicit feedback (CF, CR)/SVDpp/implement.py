
import pandas as pd
import math 
import numpy as np

n = 943
m = 1682
training_data = pd.read_csv('../../dataset/ml-100k/ua.base', delim_whitespace=True, index_col=False, header=None)
training_data_implicit = training_data.sample(frac = 0.5)
training_data_implicit_length = training_data_implicit.index.size

training_data = training_data.drop(training_data_implicit.index)
training_data_length = training_data.index.size

testing_data = pd.read_csv('../../dataset/ml-100k/ua.test', delim_whitespace=True, index_col=False, header=None)
testing_data_length = testing_data.index.size


def SVDpp_initialization(d):
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

    I_u = {}
    for index, row in training_data_implicit.iterrows():
        usr_id = row[0]
        item_id = row[1]
        I_u.setdefault(usr_id)
        if I_u[usr_id] == None:
            I_u[usr_id] = []
        I_u[usr_id].append(item_id)

    U = (np.random.random((n,d)) - 0.5) * 0.01
    V = (np.random.random((m,d)) - 0.5) * 0.01
    W = (np.random.random((m,d)) - 0.5) * 0.01
    return r_mean, b_usr, b_item, U, V, W, I_u

def SVDpp_prediction(mu, b_u, b_i, U_u, V_i_T, U_virtual_u):
    prediction = mu + b_u + b_i + U_u @ V_i_T + U_virtual_u @ V_i_T
    if prediction > 5: prediction = 5
    if prediction < 1: prediction = 1
    return prediction

def SVDpp(alpha_u, alpha_v, alpha_w, beta_u, beta_v, gamma, d, T):
    mu, b_u, b_i, U, V, W, I_u = SVDpp_initialization(d)

    print('training...')
    # training
    for t in range(0, T):
        print(t)
        for index, row in training_data.iterrows():
            usr_id = row[0]
            item_id = row[1]
            rating = row[2]
            U_virtual_u = np.zeros([1, d])
            for v_i in I_u[usr_id]:
                U_virtual_u += W.reshape(m, d)[v_i - 1]
            U_virtual_u /= math.sqrt(abs(sum(I_u[usr_id])))
            prediction = SVDpp_prediction(mu, b_u[usr_id], b_i[item_id], U[usr_id - 1].reshape(1, d), V[item_id - 1].reshape(d, 1), U_virtual_u)
            e_ui = rating - prediction
            delta_mu = -e_ui
            delta_b_u = -e_ui + beta_u * b_u[usr_id]
            delta_b_i = -e_ui + beta_v * b_i[item_id]
            delta_U_u = -e_ui * V[item_id - 1] + alpha_u * U[usr_id - 1]
            delta_V_i = -e_ui * (U[usr_id - 1] + U_virtual_u) + alpha_v * V[item_id - 1]
            delta_W_i_v = np.zeros([m, d])
            for v_i in I_u[usr_id]:
                delta_W_i_v[v_i - 1] = -e_ui / math.sqrt(abs(sum(I_u[usr_id]))) * V[item_id - 1] + alpha_w * W.reshape(m, d)[v_i - 1]
            
            
            mu -= gamma * delta_mu
            b_u[usr_id] -= gamma *  delta_b_u
            b_i[item_id] -= gamma * delta_b_i
            U[usr_id - 1] -= gamma * delta_U_u.reshape(d)
            V[item_id - 1] -= gamma * delta_V_i.reshape(d)
            for v_i in I_u[usr_id]:
                W[v_i - 1] -= gamma * delta_W_i_v[v_i - 1]
        gamma *= 0.9


    # save params
    # ...
    
    print('testing...')
    # testing
    bias_sum = 0
    square_bias_sum = 0
    for index, row in testing_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rating = row[2]
        U_virtual_u = np.zeros([1, d])
        for v_i in I_u[usr_id]:
            U_virtual_u += W.reshape(m, d)[v_i - 1]
        U_virtual_u /= math.sqrt(abs(sum(I_u[usr_id])))
        r_ui_prediction = SVDpp_prediction(mu, b_u[usr_id], b_i[item_id], U[usr_id - 1].reshape(1, d), V[item_id - 1].reshape(d, 1), U_virtual_u)
        
        bias_sum += abs(r_ui_prediction - rating)
        square_bias_sum += (r_ui_prediction - rating) ** 2

    MAE = bias_sum / testing_data_length
    RMSE = math.sqrt(square_bias_sum / testing_data_length)

    return MAE, RMSE


def main():
    alpha_u = alpha_v = alpha_w = beta_u = beta_v = 0.01
    gamma = 0.01
    T = 100
    d = 20

    print("SVD++")
    MAE, RMSE = SVDpp(alpha_u, alpha_v, alpha_w, beta_u, beta_v, gamma, d, T)
    print(MAE)
    print(RMSE)

    # T = 0
    # print(T)
    # MAE, RMSE = SVDpp(alpha_u, alpha_v, alpha_w, beta_u, beta_v, gamma, d, T)
    # print(MAE)
    # print(RMSE)

    # T = 1
    # print(T)
    # MAE, RMSE = SVDpp(alpha_u, alpha_v, alpha_w, beta_u, beta_v, gamma, d, T)
    # print(MAE)
    # print(RMSE)

    # T = 5
    # print(T)
    # MAE, RMSE = SVDpp(alpha_u, alpha_v, alpha_w, beta_u, beta_v, gamma, d, T)
    # print(MAE)
    # print(RMSE)

    # T = 10
    # print(T)
    # MAE, RMSE = SVDpp(alpha_u, alpha_v, alpha_w, beta_u, beta_v, gamma, d, T)
    # print(MAE)
    # print(RMSE)

    # T = 15
    # print(T)
    # MAE, RMSE = SVDpp(alpha_u, alpha_v, alpha_w, beta_u, beta_v, gamma, d, T)
    # print(MAE)
    # print(RMSE)

    return

if __name__ == '__main__':
    main()