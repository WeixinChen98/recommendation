import numpy as np
import pandas as pd
import math
import random
import argparse
import tensorflow as tf
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


# without validation
def eALS(n, m, d, U, I, k, I_te, P, I_u, S_u, U_u, V_i, P_i, Q_i, T, gamma, alpha_u, alpha_v, alpha_p, alpha_q):
    # disable eager execution
    tf.compat.v1.disable_eager_execution()

    # training
    user_u = tf.compat.v1.placeholder(tf.int32, name='user_u')
    item_i_t = tf.compat.v1.placeholder(tf.int32, name='item_i_t')
    item_i_t_minus_1 = tf.compat.v1.placeholder(tf.int32, name='item_i_t_minus_1')
    item_j = tf.compat.v1.placeholder(tf.int32, name='item_j')
    learning_rate_setter = tf.compat.v1.placeholder(tf.float32, name='learning_rate_setter')
    U_ = tf.Variable(U_u)
    V_ = tf.Variable(V_i)
    P_ = tf.Variable(P_i)
    Q_ = tf.Variable(Q_i)
    U_u_ = tf.nn.embedding_lookup(U_, user_u)
    V_i_t_ = tf.nn.embedding_lookup(V_, item_i_t)
    V_j_ = tf.nn.embedding_lookup(V_, item_j)
    P_i_t_minus_1_ = tf.nn.embedding_lookup(P_, item_i_t_minus_1)
    Q_i_t_ = tf.nn.embedding_lookup(Q_, item_i_t)
    Q_j_ = tf.nn.embedding_lookup(Q_, item_j)

    r_p_ui = tf.matmul(tf.reshape(U_u_, shape = [1, d]), tf.reshape(V_i_t_, shape = [d, 1])) + tf.matmul(tf.reshape(P_i_t_minus_1_, shape = [1, d]), tf.reshape(Q_i_t_, [d, 1]))
    r_p_uj = tf.matmul(tf.reshape(U_u_, shape = [1, d]), tf.reshape(V_j_, shape = [d, 1])) + tf.matmul(tf.reshape(P_i_t_minus_1_, shape = [1, d]), tf.reshape(Q_j_, [d, 1]))
    loss =  tf.negative(tf.math.log(tf.sigmoid(r_p_ui - r_p_uj))) + \
            tf.multiply(0.5 * alpha_u, tf.reduce_sum(tf.square(U_u_))) + \
            tf.multiply(0.5 * alpha_v, tf.reduce_sum(tf.square(V_i_t_))) + \
            tf.multiply(0.5 * alpha_v, tf.reduce_sum(tf.square(V_j_))) + \
            tf.multiply(0.5 * alpha_p, tf.reduce_sum(tf.square(P_i_t_minus_1_))) + \
            tf.multiply(0.5 * alpha_q, tf.reduce_sum((tf.square(Q_i_t_)))) + \
            tf.multiply(0.5 * alpha_q, tf.reduce_sum((tf.square(Q_j_))))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate_setter).minimize(loss)

    # train
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for t_training in range(T):
            print('training... t = ' + str(t_training))
            random.shuffle(P)
            for (u, i) in P:
                t = S_u[u].index(i)

                # start from the second item
                if t == 0:
                    continue
                
                i_t_minus_1 = S_u[u][t-1]
                j = random.sample(set(I) - set(I_u[u]), k = 1)
                
                # gradient descent
                sess.run(optimizer, feed_dict={learning_rate_setter:gamma, user_u: u, item_i_t: i, item_i_t_minus_1: i_t_minus_1, item_j: j})

            # test
            if t_training + 1 == T:
                print('testing...')
                r_p = [[] for i in range(n + 1)]
                for u in U:
                    r_p[u] = [-100.0] * (m + 1)
                    for j in set(I) - set(I_u[u]):
                        i_t_minus_1 = S_u[u][-1]
                        r_p[u][j] = sess.run(r_p_uj, feed_dict = {user_u: u, item_i_t_minus_1: i_t_minus_1, item_j: j})

                # ranked recommandation matrix
                I_re = [[] for i in range(n + 1)]
                for u in U:
                    I_re[u] = np.argsort(np.array(r_p[u], dtype = 'float32')).tolist()
                    I_re[u].reverse()
                
                pre_score = Pre(k, U, I_re, I_te)
                print("Pre@" + str(k) + ": " + str(pre_score))

                rec_score = Rec(k, U, I_re, I_te)
                print("Rec@" + str(k) + ": " + str(rec_score))

                ndcg_score = NDCG(k, U, I_re, I_te)
                print("NDCG@" + str(k) + ": " + str(ndcg_score))
                    


def main():
    parser = argparse.ArgumentParser(description = 'manual to this script')
    parser.add_argument("--load_dir", type = str, default = '../../dataset/ml-100k/u.data')
    parser.add_argument("--T", type = int, default = 100)
    parser.add_argument("--user_length", type = int, default = 943)
    parser.add_argument("--item_length", type = int, default = 1682)
    parser.add_argument("--latent_dim", type = int, default = 20)
    parser.add_argument("--learning_rate", type = float, default = 0.01) 
    parser.add_argument("--alpha", type = float, default = 0.01) # 0.1, 0.01, 0.001
    parser.add_argument("--k", type = int, default = 20)

    args = parser.parse_args()

    data_file_path = args.load_dir
    n = args.user_length
    m = args.item_length

    # tradeoff parameters
    T = args.T
    d = args.latent_dim
    gamma = args.learning_rate
    alpha_u = alpha_v = alpha_p = alpha_q = args.alpha 
    k = args.k

    # initialize parameters
    data = pd.read_csv(data_file_path, delim_whitespace=True, index_col=False, header=None) 
    U, I, P, I_u, S_u, I_te = preprocess_data(data, n , m)
    U_u = (np.random.random((n + 1, d)) - 0.5).astype('float32') * 0.01
    V_i = (np.random.random((m + 1, d)) - 0.5).astype('float32') * 0.01
    P_i = (np.random.random((m + 1, d)) - 0.5).astype('float32') * 0.01
    Q_i = (np.random.random((m + 1, d)) - 0.5).astype('float32') * 0.01

    print('FPMC')
    print('d = ' + str(d))
    print('gamma = ' + str(gamma))
    print('alpha_u = ' + str(alpha_u))
    print('alpha_v = ' + str(alpha_v))
    print('alpha_p = ' + str(alpha_p))
    print('alpha_q = ' + str(alpha_q))
    print('k = ' + str(k))

    eALS(n, m, d, U, I, k, I_te, P, I_u, S_u, U_u, V_i, P_i, Q_i, T, gamma, alpha_u, alpha_v, alpha_p, alpha_q)


if __name__ == '__main__':
    main()