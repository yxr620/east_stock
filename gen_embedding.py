import argparse
import torch
import numpy as np

from utils import get_file_list, day_dataset_para
from model import GRU_multi
from torch.utils.data import DataLoader
from tqdm import tqdm

# python gen_embedding.py --start 2020-01-01 --endtrain 2020-04-30 --endtest 2020-05-31 --target 0
# python gen_embedding.py --start 2016-01-01 --endtrain 2016-12-31 --endtest 2017-12-31 --target 0
# python gen_embedding.py --start 2017-01-01 --endtrain 2017-12-31 --endtest 2018-12-31 --target 0
# python gen_embedding.py --start 2018-01-01 --endtrain 2018-12-31 --endtest 2019-12-31 --target 0
# python gen_embedding.py --start 2019-01-01 --endtrain 2019-12-31 --endtest 2020-12-31 --target 0
# python gen_embedding.py --start 2020-01-01 --endtrain 2020-12-31 --endtest 2021-12-31 --target 0
# python gen_embedding.py --start 2021-01-01 --endtrain 2021-12-31 --endtest 2022-12-31 --target 0
# python gen_embedding.py --start 2022-01-01 --endtrain 2022-12-31 --endtest 2023-12-31 --target 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="start date of training", default="2020-01-01")
    parser.add_argument("--endtrain", type=str, help="end of training data, start of testing", default="2020-04-30")
    parser.add_argument("--endtest", type=str, help="end of testing data ", default="2020-05-31")
    parser.add_argument("--target", type=int, help="specify which target to be used", default=0)
    args = parser.parse_args()

    input_size = 6 # feature number of each day
    num_layers = 1
    hidden_size = 64
    output_size = 30
    num_epochs = 100
    learning_rate = 0.0001
    batch_size = 1024
    L2_weight = 0.1

    file_list_init = get_file_list('./data/day_datapoint/')
    file_list = []
    train_start = args.start
    train_end = args.endtrain
    test_end = args.endtest
    target_index = args.target

    for i in range (len(file_list_init)):
        file_name = file_list_init[i].split('/')[-1]
        if file_name.split('.')[0] <= test_end and file_name.split('.')[0] >= train_start:
            file_list.append(file_list_init[i])
    file_list = file_list

    dataset = day_dataset_para(file_list)
    loader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device('cpu')
    model = GRU_multi(input_size, hidden_size, output_size, num_layers, device)
    state_dict = torch.load(f"./data/result{target_index}/{train_end}_gru.pt")
    model.load_state_dict(state_dict)

    total_embedding = []
    total_target = []

    with torch.no_grad():
        for data, target in loader:
            embedding = model(data)
            total_embedding.append(np.array(embedding))
            total_target.append(np.array(target))
    total_embedding = np.concatenate(total_embedding, axis=0)
    total_target = np.concatenate(total_target, axis=0)

    result = []
    for i in tqdm(range(len(dataset))):
        stock, date = dataset.get_info(i)
        result.append([stock, date])
    result = np.concatenate([result, total_target, total_embedding], axis=1)

    print(result.shape)
    np.savetxt(f"./data/result0/{train_start}_{test_end}.txt", result, fmt="%s")


