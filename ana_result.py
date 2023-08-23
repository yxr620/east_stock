import pandas as pd
import os
import argparse

from utils import get_file_list, day_dataset, day_dataset_para
from tqdm import tqdm
from scipy.stats import pearsonr


def cal_ic(result, dir):
    grouped = result.groupby('date')
    result['pred_rank'] = result.groupby('date')['pred'].rank()
    result['target_rank'] = result.groupby('date')['true'].rank()
    ic_values = grouped.apply(lambda x: pearsonr(x['pred'], x['true'])[0])
    rank_ic_values = grouped.apply(lambda x: pearsonr(x['pred_rank'], x['target_rank'])[0])

    # IC mean etc
    ic_mean = ic_values.mean()
    ic_std = ic_values.std()
    ic_ir = ic_mean / ic_std
    postive_ic = len(ic_values[ic_values > 0]) / len(ic_values)

    f = open(dir + "test.log", 'a')
    f.write(f"\nIC Mean: {ic_mean:.5f}")
    f.write(f"\nIC Std: {ic_std:.5f}")
    f.write(f"\nRank IC: {rank_ic_values.mean():.5f}")
    f.write(f"\nIC_IR: {ic_ir:.5f}")
    f.write(f"\npostive_ic: {postive_ic:.5f}")

if __name__ == "__main__":
    # command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="result dir", default="./data/result1/")
    args = parser.parse_args()

    # load true table and pred table
    pred_file = os.path.join(args.dir, "concate_result.feather")
    true_file = "./data/result1/true_table.feather"
    pred_table = pd.read_feather(pred_file)
    if os.path.exists(true_file):
        true_table = pd.read_feather(true_file)
    else:
        file_list_init = get_file_list('./data/day_datapoint/')
        file_list = []
        for i in range(len(file_list_init)):
            # print(file_list_init[i].split('/')[-1].split('.')[0])
            if file_list_init[i].split('/')[-1].split('.')[0] > "2016-12-31":
                file_list.append(file_list_init[i])

        # get the pred result and the original data
        all_data = day_dataset_para(file_list)

        # get the true table
        true_table = []
        for i in tqdm(range(len(all_data))):
            info = all_data.get_info(i)
            feature, target = all_data[i]
            true_table.append([info[0], info[1], float(target[0])])
        del all_data

        true_table = pd.DataFrame(true_table, columns=["stock_code", "date", "true"])
        true_table.to_feather("./data/result1/true_table.feather")

    # group table by date
    total_table = pd.concat([true_table, pred_table["pred"]], axis=1)
    print(true_table)
    print(pred_table)
    print(total_table)

    total_group = total_table.groupby("date")
    pred_group = pred_table.groupby("date")
    true_group = true_table.groupby("date")

    cal_ic(total_table, "./data/result1/")
    # for key in total_group.groups.keys():
    #     tmp_table = total_group.get_group(key)





