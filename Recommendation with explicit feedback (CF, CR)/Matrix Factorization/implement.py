
import math
import pandas as pd
import numpy as np
from numpy import mat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

n = 943
m = 1682
training_data = pd.read_csv('../../dataset/ml-100k/u1.base', delim_whitespace=True, index_col=False, header=None)
testing_data = pd.read_csv('../../dataset/ml-100k/u1.test', delim_whitespace=True, index_col=False, header=None)
training_data_length = training_data.index.size
testing_data_length = testing_data.index.size


def to_filled_matrix(data, n, m, x_mean):
    data_mat = mat(np.zeros([n, m]), dtype=np.float32)
    for index, row in data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rating = row[2]
        data_mat[usr_id - 1, item_id - 1] = rating

    to_be_filled = np.argwhere(data_mat == 0)
    for pos in to_be_filled:
        x = pos[0]
        y = pos[1]
        data_mat[x, y] = x_mean[x + 1]

    return data_mat

def Pure_SVD(r_usr_mean, d):
    R = to_filled_matrix(training_data, n, m, r_usr_mean)
    filled_usr_mean = R.mean(1)
    R1 = np.subtract(R, filled_usr_mean @ mat(np.ones([1, m])))

    u, s, vt = svds(R1, k=d)
    s = np.diag(s)
    # u, s, vt = np.linalg.svd(R1)
    # u = u[:, :d]
    # s = np.diag(s[:d])
    # vt = vt[:d,:]

    # r_prediction = np.matmul(np.matmul(u,s), vt)
    r_prediction = u @ s @ vt

    bias_sum = 0
    square_bias_sum = 0
    for index, row in testing_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rating = row[2]
        r_ui_prediction = filled_usr_mean[usr_id - 1] + r_prediction[usr_id - 1, item_id - 1]
        bias_sum += abs(r_ui_prediction - rating)
        square_bias_sum += (r_ui_prediction - rating) ** 2

    MAE = bias_sum / testing_data_length
    RMSE = math.sqrt(square_bias_sum / testing_data_length)
    return MAE, RMSE

def RSVD_initialization(d):
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

    U = (np.random.random((n,d)) - 0.5) * 0.01
    V = (np.random.random((m,d)) - 0.5) * 0.01

    return r_mean, b_usr, b_item, U, V

def RSVD_prediction(mu, b_u, b_i, U_u, V_i_T):
    return mu + b_u + b_i + U_u @ V_i_T

def RSVD(alpha_u, alpha_v, beta_u, beta_v, gamma, d, T):
    # initialization
    mu, b_u, b_i, U, V = RSVD_initialization(d)

    # training
    for t in range(0, T):
        print(t)
        for index, row in training_data.iterrows():
            usr_id = row[0]
            item_id = row[1]
            rating = row[2]
            prediction = RSVD_prediction(mu, b_u[usr_id], b_i[item_id], U[usr_id - 1].reshape(1, d), V[item_id - 1].reshape(d, 1))
            e_ui = rating - prediction
            delta_mu = -e_ui
            delta_b_u = -e_ui + beta_u * b_u[usr_id]
            delta_b_i = -e_ui + beta_v * b_i[item_id]
            delta_U_u = -e_ui * V[item_id - 1] + alpha_u * U[usr_id - 1]
            delta_V_i = -e_ui * U[usr_id - 1] + alpha_v * V[item_id - 1]
            
            mu -= gamma * delta_mu
            b_u[usr_id] -= gamma *  delta_b_u
            b_i[item_id] -= gamma * delta_b_i
            U[usr_id - 1] -= gamma * delta_U_u.reshape(d)
            V[item_id - 1] -= gamma * delta_V_i.reshape(d)
        gamma *= 0.9
        
    # testing
    bias_sum = 0
    square_bias_sum = 0
    for index, row in testing_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rating = row[2]
        r_ui_prediction = RSVD_prediction(mu, b_u[usr_id], b_i[item_id], U[usr_id - 1].reshape(1, d), V[item_id - 1].reshape(d, 1))
        bias_sum += abs(r_ui_prediction - rating)
        square_bias_sum += (r_ui_prediction - rating) ** 2

    MAE = bias_sum / testing_data_length
    RMSE = math.sqrt(square_bias_sum / testing_data_length)

    return MAE, RMSE



def main():
    r_usr = {}
    for index, row in training_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rating = row[2]
        r_usr.setdefault(usr_id)
        if r_usr[usr_id] == None:
            r_usr[usr_id] = {}
        r_usr[usr_id][item_id] = rating

    r_usr_mean = {}
    for usr_id in r_usr:
        r_usr_mean[usr_id] = sum(r_usr[usr_id].values()) / len(r_usr[usr_id])

    alpha_u = alpha_v = beta_u = beta_v = 0.01
    gamma = 0.01
    T = 100
    d = 20
    print("Pure_SVD")
    MAE, RMSE = Pure_SVD(r_usr_mean, d)
    print(MAE)
    print(RMSE)

    print("RSVD")
    MAE, RMSE = RSVD(alpha_u, alpha_v, beta_u, beta_v, gamma, d, T)
    print(MAE)
    print(RMSE)

    return

if __name__ == '__main__':
    main()