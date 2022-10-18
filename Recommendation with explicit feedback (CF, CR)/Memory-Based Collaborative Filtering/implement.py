import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np


n = 943
m = 1682
K = 50
lambda_UCF = 0.5
training_data = pd.read_csv('../../dataset/ml-100k/u1.base', delim_whitespace=True, index_col=False, header=None)
testing_data = pd.read_csv('../../dataset/ml-100k/u1.test', delim_whitespace=True, index_col=False, header=None)
training_data_length = training_data.index.size
testing_data_length = testing_data.index.size


def similarity_u(r_uk, r_wk, r_u_mean, r_w_mean) -> float: 
    numerator = 0
    denominator_p1 = 0
    denominator_p2 = 0
    for k in r_uk:
        if not k in r_wk:
            continue
        numerator += (r_uk[k] - r_u_mean) * (r_wk[k] - r_w_mean)
        denominator_p1 += (r_uk[k] - r_u_mean) ** 2
        denominator_p2 += (r_wk[k] - r_w_mean) ** 2
    if denominator_p1 == 0 or denominator_p2 == 0:
        return 0
    denominator = math.sqrt(denominator_p1) * math.sqrt(denominator_p2)
    return numerator / denominator

def similarity_i(r_usr, k, j, r_usr_mean) -> float: 
    numerator = 0
    denominator_p1 = 0
    denominator_p2 = 0
    for u in r_usr:
        if not k in r_usr[u] or not j in r_usr[u]:
            continue
        numerator += (r_usr[u][k] - r_usr_mean[u]) * (r_usr[u][j] - r_usr_mean[u])
        denominator_p1 += (r_usr[u][k] - r_usr_mean[u]) ** 2
        denominator_p2 += (r_usr[u][j] - r_usr_mean[u]) ** 2
    if denominator_p1 == 0 or denominator_p2 == 0:
        return 0
    denominator = math.sqrt(denominator_p1) * math.sqrt(denominator_p2)
    return numerator / denominator

def top_k_u(usr_id, r_usr, r_usr_mean, item_id, K):
    if not usr_id in r_usr:
        return {}
    r_uk = r_usr[usr_id]
    r_u_mean = r_usr_mean[usr_id]
    pcc = {}
    for r_w in r_usr:
        if r_w == usr_id or not item_id in r_usr[r_w]:
            continue
        r_wk = r_usr[r_w]
        r_w_mean = r_usr_mean[r_w]
        _pcc = similarity_u(r_uk, r_wk, r_u_mean, r_w_mean)
        if _pcc > 0:
            pcc[r_w] = _pcc

    top_k_pcc = dict(sorted(pcc.items(), key=lambda item: item[1], reverse=True)[:K])
    return top_k_pcc
    

def top_k_i(usr_id, r_usr, r_item, r_usr_mean, item_id, K):
    if not item_id in r_item:
        return {}
    similarity = {}
    for k in r_item:
        if k == item_id or not k in r_usr[usr_id]:
            continue
        _similarity = similarity_i(r_usr, k, item_id, r_usr_mean)
        if _similarity > 0:
            similarity[k] = _similarity

    top_k = dict(sorted(similarity.items(), key=lambda item: item[1], reverse=True)[:K])
    return top_k

def main() -> None:
    r_usr = {}
    r_item = {}
    for index, row in training_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rating = row[2]
        r_usr.setdefault(usr_id)
        r_item.setdefault(item_id)
        if r_usr[usr_id] == None:
            r_usr[usr_id] = {}
        r_usr[usr_id][item_id] = rating
        if r_item[item_id] == None:
            r_item[item_id] = {}
        r_item[item_id][usr_id] = rating

    r_usr_mean = {}
    for usr_id in r_usr:
        r_usr_mean[usr_id] = sum(r_usr[usr_id].values()) / len(r_usr[usr_id]) 

    r_item_mean = {}
    for item_id in r_item:
        r_item_mean[item_id] = sum(r_item[item_id].values()) / len(r_item[item_id])
    

    UCF_prediction_bias_sum = 0
    ICF_prediction_bias_sum = 0
    HCF_prediction_bias_sum = 0
    UCF_prediction_square_bias_sum = 0
    ICF_prediction_square_bias_sum = 0
    HCF_prediction_square_bias_sum = 0
    for index, row in testing_data.iterrows():
        print(index)
        usr_id = row[0]
        item_id = row[1]
        rating = row[2]

        UCF_top_k_pcc = top_k_u(usr_id, r_usr, r_usr_mean, item_id, K)
        UCF = r_usr_mean[usr_id]
        UCF_bias_numerator = 0
        UCF_bias_denominator = sum(UCF_top_k_pcc.values())            
        for w in UCF_top_k_pcc:
            UCF_bias_numerator += UCF_top_k_pcc[w]*(r_usr[w][item_id] - r_usr_mean[w])
        
        if UCF_bias_denominator != 0:
            UCF_bias = UCF_bias_numerator / UCF_bias_denominator
            UCF += UCF_bias
        if UCF > 5: UCF = 5
        if UCF < 1: UCF = 1
        UCF_prediction_bias_sum += abs(UCF - rating)
        UCF_prediction_square_bias_sum += (UCF - rating) ** 2

        ICF_top_k_similarity = top_k_i(usr_id, r_usr, r_item, r_usr_mean, item_id, K)
        ICF = r_usr_mean[usr_id]
        ICF_numerator = 0
        ICF_bias_denominator = sum(ICF_top_k_similarity.values())          
        for k in ICF_top_k_similarity:
            ICF_numerator += ICF_top_k_similarity[k]*(r_usr[usr_id][k])
        
        if ICF_bias_denominator != 0:
            ICF = ICF_numerator / ICF_bias_denominator
        if ICF > 5: ICF = 5
        if ICF < 1: ICF = 1
        ICF_prediction_bias_sum += abs(ICF - rating)
        ICF_prediction_square_bias_sum += (ICF - rating) ** 2

        HCF = lambda_UCF * UCF + (1 - lambda_UCF) * ICF
        HCF_prediction_bias_sum += abs(HCF - rating)
        HCF_prediction_square_bias_sum += (HCF - rating) ** 2


    
    UCF_MAE = UCF_prediction_bias_sum / testing_data_length
    UCF_RMSE = math.sqrt(UCF_prediction_square_bias_sum / testing_data_length)

    ICF_MAE = ICF_prediction_bias_sum / testing_data_length
    ICF_RMSE = math.sqrt(ICF_prediction_square_bias_sum / testing_data_length)

    HCF_MAE = HCF_prediction_bias_sum / testing_data_length
    HCF_RMSE = math.sqrt(HCF_prediction_square_bias_sum / testing_data_length)

    print(UCF_MAE)
    print(UCF_RMSE)

    print(ICF_MAE)
    print(ICF_RMSE)

    print(HCF_MAE)
    print(HCF_RMSE)



    

if __name__ == '__main__':
    main()