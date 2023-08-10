import os
import pandas as pd
import numpy as np
import pickle
import builtins 

from multiprocessing import current_process, Pool
from datetime import datetime, timedelta
from tqdm import tqdm

# return date_list table_list
def read_table(dir):
    file_list = os.listdir(dir)
    print(file_list)
    table_list = []
    date_list = []
    for file in tqdm(file_list):
        date_list.append(datetime.strptime(file.split('.')[0], "%Y-%m-%d"))
        table_list.append(pd.read_feather(dir + file))
    
    return date_list, table_list

# input numpy shape (x, y), normalize x
# x_i / x_0
def time_norm(table):
    first = table[:, 0].reshape(-1, 1)
    result = table / first
    return result

# input numpy shape (x, y), normalize y
def zscore_norm(table):
    return (table - np.mean(table, axis=0)) / np.std(table, axis=0)

# generate daypoint using all stock info
# 0-29: feature
# target0: 39 close - 29 close
# target1: 40 close - 30 close
# target2: 35 close - 30 close
# args = [
# date_list: containing all date 
# table_list: containing all table dataframe
# i: the selected date index
# ]
# stock datetime    target  open    high    low close   vwap    volume
# 1     1           3       30     30     30 30     30     30
def process_daypoint(args):
    # print("Process ID:", current_process().pid)
    date_list, table_list, i = args
    stock_set = set(table_list[i]["code"])
    basic_info = []
    target = []
    open = []
    high = []
    low = []
    close = []
    vwap = []
    volume = []

    # get the included stock code & group table by code
    j = i - 29
    group_table = []
    while j <= i: # 0-29
        stock_set = stock_set & set(table_list[j]["code"])
        group_table.append(table_list[j].groupby('code'))
        j += 1
    while j <= i + 11: #29-40
        stock_set = stock_set & set(table_list[j]["code"])
        j += 1

    # generate daypoint for each stock
    for stock in stock_set:
        # get basic info [stock, date]
        basic_info.append([stock, date_list[i].strftime('%Y-%m-%d')])

        # get target
        T0_close = table_list[i][table_list[i]["code"] == stock]["close"].item()[-1]
        T1_close = table_list[i + 1][table_list[i + 1]["code"] == stock]["close"].item()[-1]
        T6_close = table_list[i + 6][table_list[i + 6]["code"] == stock]["close"].item()[-1]
        T10_close = table_list[i + 10][table_list[i + 10]["code"] == stock]["close"].item()[-1]
        T11_close = table_list[i + 11][table_list[i + 11]["code"] == stock]["close"].item()[-1]
        target0 = (T10_close - T0_close) / T0_close
        target1 = (T11_close - T1_close) / T1_close
        target2 = (T6_close - T1_close) / T1_close
        target.append([target0, target1, target2])

        # get feature
        result = [[], [], [], [], [], []]
        for iter_table in group_table:
            stock_table = iter_table.get_group(stock).reset_index(drop=True)
            result[0].extend(list(stock_table["open"].tolist()[0]))
            result[1].extend(list(stock_table["high"].tolist()[0]))
            result[2].extend(list(stock_table["low"].tolist()[0]))
            result[3].extend(list(stock_table["close"].tolist()[0]))
            result[4].extend(list(stock_table["vwap"].tolist()[0]))
            tmp_v = list(stock_table["volume"].tolist()[0]) # set volume to 1 if it is 0
            result[5].extend([1 if x == 0 else x for x in tmp_v]) 

        open.append(result[0])
        high.append(result[1])
        low.append(result[2])
        close.append(result[3])
        vwap.append(result[4])
        volume.append(result[5])

    # normalize feature
    volume = np.array(volume)
    open = np.array(open)
    high = np.array(high)
    low = np.array(low) 
    close = np.array(close)
    vwap = np.array(vwap)
    basic_info = np.array(basic_info)

    volume = volume / volume[:, 0].reshape((-1, 1))
    open = open / close[:, 0].reshape((-1, 1))
    high = high / close[:, 0].reshape((-1, 1))
    low = low / close[:, 0].reshape((-1, 1))
    close = close / close[:, 0].reshape((-1, 1))
    vwap = vwap / close[:, 0].reshape((-1, 1))


    # volume = time_norm(np.array(volume))
    # open = time_norm(np.array(open))
    # high = time_norm(np.array(high))
    # low = time_norm(np.array(low))
    # close = time_norm(np.array(close))
    # vwap = time_norm(np.array(vwap))
    # basic_info = np.array(basic_info)

    # open = zscore_norm(open)
    # high = zscore_norm(high)
    # low = zscore_norm(low)
    # close = zscore_norm(close)
    # vwap = zscore_norm(vwap)
    # volume = zscore_norm(volume)
    # target = zscore_norm(target)

    day_data = np.concatenate([basic_info, target, open, high, low, close, vwap, volume], axis=1)
    # with builtins.open(f"./full_data/day_datapoint/{date_list[i].strftime('%Y-%m-%d')}.pickle", 'wb') as f:
    #     pickle.dump(day_data, f)
    np.savetxt(f"./data/day_datapoint/{date_list[i].strftime('%Y-%m-%d')}.txt", day_data, fmt="%s")
    print(date_list[i].strftime('%Y-%m-%d'))


# generate day datapoint
def generate_daypoint():
    date_list, table_list = read_table("./data/day_table/")
    print(len(date_list))
    print(date_list[29:-11])

    # serial 
    # for i in range(29, len(date_list) - 11):
    #     process_daypoint([date_list, table_list, i])

    args_list = [( date_list, table_list, i) for i in range(29, len(date_list) - 11)]
    with Pool(processes=4) as pool:
        pool.map(process_daypoint, args_list)



if __name__ == "__main__":
    generate_daypoint()



