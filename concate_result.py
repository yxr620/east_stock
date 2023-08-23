import os
import pandas as pd

from tqdm import tqdm

def concate_result(dir):
    file_list_init = os.listdir(dir)
    file_list = []
    table_list = []
    for file in (file_list_init):
        if file[-3:] == 'csv':
            table = pd.read_csv(dir + file)
            # table = table.drop(table.columns[0], axis=1)
            table_list.append(table)
            file_list.append(file)
            print(f"{file} {table.shape}")
            # group = table.groupby("date")
            # keys = group.groups.keys()
            # for key in keys:
            #     print(f"{key} {group.get_group(key).shape}")
            # print(keys)
            # exit()


    result = pd.concat(table_list, axis=0).reset_index(drop=True)
    print(result)
    result.to_feather(dir + "concate_result.feather")

if __name__ == "__main__":
    dir0 = "./data/result0/"
    dir1 = "./data/result1/"
    concate_result(dir0)
