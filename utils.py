import torch
import numpy as np
import pandas as pd
import os

from multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# get day datapoint [40 * 6]
def get_day_feature(datapoint):
    info = [datapoint[0], datapoint[1]]
    target = datapoint[2:8].astype(np.double)
    feature = datapoint[8:].astype(np.double).reshape(6, 30).T
    # print(datapoint[8:].astype(np.double))
    # print(feature)
    # exit()
    return info, feature, target


def get_file_list(dir):
    file_list = os.listdir(dir)
    for i in range(len(file_list)):
        file_list[i] = dir + file_list[i]
    return file_list

def loss_fn(y_pred, y_true):
    y = torch.cat((y_pred.view(1, -1), y_true.view(1, -1)), dim=0)
    corr = torch.corrcoef(y)[0, 1]
    return -corr

# delete stock whose adjfactor is empty, resulting in nan for open, high, low, close
def fix_table():
    day_dir = "./data/day_table/2019-12-13.feather"
    stock = "000043.SZ"
    day_table = pd.read_feather(day_dir)

    day_table = day_table[day_table['code'] != stock].reset_index(drop=True)
    day_table.to_feather(day_dir)

# the file name of required datapoint. Only the name needed not the entire dir
class day_dataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.target = []
        self.feature = []
        self.info = []

        for file in tqdm(file_list):
            table = np.loadtxt(file, dtype=str)
            for i in range(table.shape[0]):
                info, feature, target = get_day_feature(table[i])
                self.info.append(info)
                self.feature.append(feature)
                self.target.append(target)

        self.feature = np.array(self.feature)
        self.target = np.array(self.target)
        # the torch type must match the model type
        self.feature = torch.tensor(self.feature, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        print(self.feature.shape)

    def __getitem__(self, index):
        x = self.feature[index]
        y = self.target[index]
        return x, y
    
    def __len__(self):
        return len(self.feature)
    
    def get_info(self, index):
        return self.info[index]

class day_dataset_para(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.target = []
        self.feature = []
        self.info = []

        with Pool(processes=8) as pool:
            results = pool.map(self.process_file, self.file_list)

        # Collect the results
        for info, feature, target in results:
            self.info.extend(info)
            self.feature.extend(feature)
            self.target.extend(target)


        self.feature = np.array(self.feature)
        self.target = np.array(self.target)
        # the torch type must match the model type
        self.feature = torch.tensor(self.feature, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        print(self.feature.shape)

    # Define a helper function for parallel processing
    def process_file(self, file):
        # print(file)
        table = np.loadtxt(file, dtype=str)
        features, targets, infos = [], [], []
        for i in range(table.shape[0]):
            info, feature, target = get_day_feature(table[i])
            infos.append(info)
            features.append(feature)
            targets.append(target)
        return infos, features, targets

    def __getitem__(self, index):
        x = self.feature[index]
        y = self.target[index]
        return x, y
    
    def __len__(self):
        return len(self.feature)
    
    def get_info(self, index):
        return self.info[index]