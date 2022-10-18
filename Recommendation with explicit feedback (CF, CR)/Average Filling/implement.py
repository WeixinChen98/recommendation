import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np


n = 943
m = 1682
training_data = pd.read_csv('../../dataset/ml-100k/u1.base', delim_whitespace=True, index_col=False, header=None)
testing_data = pd.read_csv('../../dataset/ml-100k/u1.test', delim_whitespace=True, index_col=False, header=None)
training_data_length = training_data.index.size
testing_data_length = testing_data.index.size



def main() -> None: 

    r_sum = training_data.sum(axis=0)[2]
    r_mean = r_sum / training_data_length

    r_usr = {}
    r_usr_mean = {}
    for index, row in training_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rate = row[2]
        r_usr.setdefault(usr_id)
        if r_usr[usr_id] == None:
            r_usr[usr_id] = {}
        r_usr[usr_id][item_id] = rate

    r_te_usr = {}
    for index, row in testing_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rate = row[2]
        r_te_usr.setdefault(usr_id)
        if r_te_usr[usr_id] == None:
            r_te_usr[usr_id] = {}
        r_te_usr[usr_id][item_id] = rate
    
    for usr_id in r_usr:
        r_usr_mean[usr_id] = sum(r_usr[usr_id].values()) / len(r_usr[usr_id]) 
    
    for usr_id in range(1, n):
        r_usr_mean.setdefault(usr_id)
        if r_usr_mean[usr_id] == None:
            r_usr_mean[usr_id] = r_mean

    r_item = {}
    r_item_mean = {}
    for index, row in training_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rate = row[2]
        r_item.setdefault(item_id)
        if r_item[item_id] == None:
            r_item[item_id] = {}
        r_item[item_id][usr_id] = rate
    
    for item_id in r_item:
        r_item_mean[item_id] = sum(r_item[item_id].values()) / len(r_item[item_id])

    for item_id in range(1, m):
        r_item_mean.setdefault(item_id)
        if r_item_mean[item_id] == None:
            r_item_mean[item_id] = r_mean

    b_usr = {}
    b_item = {}
    for usr_id in range(1, n):
        b = 0
        if not usr_id in r_usr:
            b_usr[usr_id] = b
            continue
        
        b_len = len(r_usr[usr_id])
        for item_id in r_usr[usr_id]:
            b += r_usr[usr_id][item_id] - r_item_mean[item_id]

        b /= b_len
        b_usr[usr_id] = b
    

    for item_id in range(1, m):
        b = 0
        if not item_id in r_item:
            b_item[item_id] = 0
            continue

        b_len = len(r_item[item_id])
        for usr_id in r_item[item_id]:
            b += r_item[item_id][usr_id] - r_usr_mean[usr_id]
        
        b /= b_len
        b_item[item_id] = b

    # maybe using array ?
    prediction_r_usr_mean_bias_sum = 0
    prediction_r_item_mean_bias_sum = 0
    prediction_r_half_half_bias_sum = 0
    prediction_b_usr_bias_sum = 0
    prediction_b_item_bias_sum = 0
    prediction_b_half_half_bias_sum = 0

    prediction_r_usr_mean_squares_bias_sum = 0
    prediction_r_item_mean_squares_bias_sum = 0
    prediction_r_half_half_squares_bias_sum = 0
    prediction_b_usr_squares_bias_sum = 0
    prediction_b_item_squares_bias_sum = 0
    prediction_b_half_half_squares_bias_sum = 0
    for index, row in testing_data.iterrows():
        usr_id = row[0]
        item_id = row[1]
        rate = row[2]

        prediction_r_usr_mean = r_usr_mean[usr_id]
        prediction_r_item_mean = r_item_mean[item_id]
        prediction_r_half_half = r_usr_mean[usr_id] / 2 + r_item_mean[item_id] / 2
        prediction_b_usr = r_item_mean[item_id] + b_usr[usr_id]
        prediction_b_item = r_usr_mean[usr_id] + b_item[item_id]
        prediction_b_half_half = r_mean + b_usr[usr_id] + b_item[item_id]

        prediction_r_usr_mean_bias_sum += abs(rate - prediction_r_usr_mean)
        prediction_r_item_mean_bias_sum += abs(rate - prediction_r_item_mean)
        prediction_r_half_half_bias_sum += abs(rate - prediction_r_half_half)
        prediction_b_usr_bias_sum += abs(rate - prediction_b_usr)
        prediction_b_item_bias_sum += abs(rate - prediction_b_item)
        prediction_b_half_half_bias_sum += abs(rate - prediction_b_half_half)

        prediction_r_usr_mean_squares_bias_sum += (rate - prediction_r_usr_mean) ** 2
        prediction_r_item_mean_squares_bias_sum += (rate - prediction_r_item_mean) ** 2
        prediction_r_half_half_squares_bias_sum += (rate - prediction_r_half_half) ** 2
        prediction_b_usr_squares_bias_sum += (rate - prediction_b_usr) ** 2
        prediction_b_item_squares_bias_sum += (rate - prediction_b_item) ** 2
        prediction_b_half_half_squares_bias_sum += (rate - prediction_b_half_half) ** 2

    mae_r_usr_mean = prediction_r_usr_mean_bias_sum / testing_data_length
    mae_r_item_mean = prediction_r_item_mean_bias_sum / testing_data_length
    mae_r_half_half = prediction_r_half_half_bias_sum / testing_data_length
    mae_b_usr = prediction_b_usr_bias_sum / testing_data_length
    mae_b_item = prediction_b_item_bias_sum / testing_data_length
    mae_b_half_half = prediction_b_half_half_bias_sum / testing_data_length

    rmse_r_usr_mean = math.sqrt(prediction_r_usr_mean_squares_bias_sum / testing_data_length)
    rmse_r_item_mean = math.sqrt(prediction_r_item_mean_squares_bias_sum / testing_data_length)
    rmse_r_half_half = math.sqrt(prediction_r_half_half_squares_bias_sum / testing_data_length)
    rmse_b_usr = math.sqrt(prediction_b_usr_squares_bias_sum / testing_data_length)
    rmse_b_item = math.sqrt(prediction_b_item_squares_bias_sum / testing_data_length)
    rmse_b_half_half = math.sqrt(prediction_b_half_half_squares_bias_sum / testing_data_length)

    result1_mae = []
    result1_mae.append(mae_r_usr_mean)
    result1_mae.append(mae_r_item_mean)
    result1_mae.append(mae_r_half_half)
    result1_mae.append(mae_b_usr)
    result1_mae.append(mae_b_item)
    result1_mae.append(mae_b_half_half)

    result1_rmse = []
    result1_rmse.append(rmse_r_usr_mean)
    result1_rmse.append(rmse_r_item_mean)
    result1_rmse.append(rmse_r_half_half)
    result1_rmse.append(rmse_b_usr)
    result1_rmse.append(rmse_b_item)
    result1_rmse.append(rmse_b_half_half)
    result1 = {'mae': result1_mae, 'rmse': result1_rmse}
    result1_df = pd.DataFrame(data=result1)
    result1_df.to_csv(path_or_buf='./results/result1.csv')

    usr_segments = [[], [], []]
    for usr_id in r_te_usr:
        rating_num = len(r_te_usr[usr_id])
        if rating_num <= 20:
            usr_segments[0].append(usr_id)
        elif rating_num <= 50:
            usr_segments[1].append(usr_id)
        else:
            usr_segments[2].append(usr_id)


    mae_usr_segments = []
    rmse_usr_segments = []
    for usr_segment in usr_segments:
        prediction_bias_sum = [0, 0, 0, 0, 0, 0]
        prediction_squares_bias_sum = [0, 0, 0, 0, 0, 0]
        mae = [0, 0, 0, 0, 0, 0]
        rmse = [0, 0, 0, 0, 0, 0]
        data_length = 0
        for usr_id in usr_segment:
            testing_data_segment = testing_data.loc[testing_data[0] == usr_id]
            data_length += testing_data_segment.index.size
            for index, row in testing_data_segment.iterrows():
                item_id = row[1]
                rate = row[2]

                prediction_r_usr_mean = r_usr_mean[usr_id]
                prediction_r_item_mean = r_item_mean[item_id]
                prediction_r_half_half = r_usr_mean[usr_id] / 2 + r_item_mean[item_id] / 2
                prediction_b_usr = r_item_mean[item_id] + b_usr[usr_id]
                prediction_b_item = r_usr_mean[usr_id] + b_item[item_id]
                prediction_b_half_half = r_mean + b_usr[usr_id] + b_item[item_id]

                prediction_bias_sum[0] += abs(rate - prediction_r_usr_mean)
                prediction_bias_sum[1] += abs(rate - prediction_r_item_mean)
                prediction_bias_sum[2] += abs(rate - prediction_r_half_half)
                prediction_bias_sum[3] += abs(rate - prediction_b_usr)
                prediction_bias_sum[4] += abs(rate - prediction_b_item)
                prediction_bias_sum[5] += abs(rate - prediction_b_half_half)

                prediction_squares_bias_sum[0] += (rate - prediction_r_usr_mean) ** 2
                prediction_squares_bias_sum[1] += (rate - prediction_r_item_mean) ** 2
                prediction_squares_bias_sum[2] += (rate - prediction_r_half_half) ** 2
                prediction_squares_bias_sum[3] += (rate - prediction_b_usr) ** 2
                prediction_squares_bias_sum[4] += (rate - prediction_b_item) ** 2
                prediction_squares_bias_sum[5] += (rate - prediction_b_half_half) ** 2

        for i in range(len(mae)):
            mae[i] = prediction_bias_sum[i] / data_length
            
        for i in range(len(rmse)):
            rmse[i] = math.sqrt(prediction_squares_bias_sum[i] / data_length)
        
        mae_usr_segments.append(mae)
        rmse_usr_segments.append(rmse)

    plt.figure(1)
    fig1_labels = ['1', '2', '3']
    fig1_data = [[],[],[],[],[],[]]
    for i in range(0, 6):
        fig1_data[i].append(rmse_usr_segments[0][i])
        fig1_data[i].append(rmse_usr_segments[1][i])
        fig1_data[i].append(rmse_usr_segments[2][i])
    x = np.arange(len(fig1_labels))  # the label locations
    width = 0.1  # the width of the bars

    
    fig1, ax1 = plt.subplots()
    fig1_rects1 = ax1.bar(x - width * 5 / 2, fig1_data[0], width, label='user average')
    fig1_rects2 = ax1.bar(x - width * 3 / 2, fig1_data[1], width, label='item average')
    fig1_rects3 = ax1.bar(x - width * 1 / 2, fig1_data[2], width, label='mean of user average and item average')
    fig1_rects4 = ax1.bar(x + width * 1 / 2, fig1_data[3], width, label='user bias and item average')
    fig1_rects5 = ax1.bar(x + width * 3 / 2, fig1_data[4], width, label='user average and item bias')
    fig1_rects6 = ax1.bar(x + width * 5 / 2, fig1_data[5], width, label='global average, user bias and item bias')
    # Add some text for fig1_labels, title and custom x-ax_1is tick fig1_labels, etc.
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE on different user segments')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fig1_labels)
    ax1.legend()

    ax1.bar_label(fig1_rects1)
    ax1.bar_label(fig1_rects2)
    ax1.bar_label(fig1_rects3)
    ax1.bar_label(fig1_rects4)
    ax1.bar_label(fig1_rects5)
    ax1.bar_label(fig1_rects6)

    fig1.savefig('./results/rmse_on_user_segments.png', dpi=fig1.dpi)

    plt.figure(2)
    fig2_labels = ['1', '2', '3']
    fig2_data = [[],[],[],[],[],[]]
    for i in range(0, 6):
        fig2_data[i].append(mae_usr_segments[0][i])
        fig2_data[i].append(mae_usr_segments[1][i])
        fig2_data[i].append(mae_usr_segments[2][i])
    x = np.arange(len(fig2_labels))  # the label locations
    width = 0.1  # the width of the bars

    
    fig2, ax2 = plt.subplots()
    fig2_rects1 = ax2.bar(x - width * 5 / 2, fig2_data[0], width, label='user average')
    fig2_rects2 = ax2.bar(x - width * 3 / 2, fig2_data[1], width, label='item average')
    fig2_rects3 = ax2.bar(x - width * 1 / 2, fig2_data[2], width, label='mean of user average and item average')
    fig2_rects4 = ax2.bar(x + width * 1 / 2, fig2_data[3], width, label='user bias and item average')
    fig2_rects5 = ax2.bar(x + width * 3 / 2, fig2_data[4], width, label='user average and item bias')
    fig2_rects6 = ax2.bar(x + width * 5 / 2, fig2_data[5], width, label='global average, user bias and item bias')

    ax2.set_ylabel('MAE')
    ax2.set_title('MAE on different user segments')
    ax2.set_xticks(x)
    ax2.set_xticklabels(fig2_labels)
    ax2.legend()

    ax2.bar_label(fig2_rects1)
    ax2.bar_label(fig2_rects2)
    ax2.bar_label(fig2_rects3)
    ax2.bar_label(fig2_rects4)
    ax2.bar_label(fig2_rects5)
    ax2.bar_label(fig2_rects6)

    fig2.savefig('./results/mae_on_user_segments.png', dpi=fig2.dpi)

if __name__ == '__main__':
    main()