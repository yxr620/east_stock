import torch
import torch.nn as nn
import argparse

from utils import get_file_list, day_dataset, day_dataset_para
from torch.utils.data import DataLoader
from model import GRUModel, LSTMModel, AGRUModel, GRU_multi
from tqdm import tqdm

# correctness checked
# @para pred: torch.tensor of shape (x)
# @para true: torch.tensor of shape (x)
# retur -corr between pred and true
def cal_negcorr(pred, true):
    y = torch.cat((pred.view(1, -1), true.view(1, -1)), dim=0)
    corr = torch.corrcoef(y)[0, 1]
    return -corr

def loss_fn(pred, true, L2_weight):
    pred_mean = torch.mean(pred, dim=1)
    neg_ic = cal_negcorr(pred_mean, true)
    corr_mat = torch.corrcoef(pred.T)
    L2norm = torch.norm(corr_mat) # L2norm of corr mat, ie penalty

    return neg_ic + L2_weight * L2norm, neg_ic, L2norm

# python main_multi.py --end 2020-05-31 --target 0
# python main_multi.py --end 2016-12-31 --target 0
# python main_multi.py --end 2017-12-31 --target 0
# python main_multi.py --end 2018-12-31 --target 0
# python main_multi.py --end 2019-12-31 --target 0
# python main_multi.py --end 2020-12-31 --target 0
# python main_multi.py --end 2021-12-31 --target 0
# python main_multi.py --end 2022-12-31 --target 0
# python main_multi.py --end 2023-12-31 --target 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=str, help="end date", default="2016-12-32")
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
    test_end = args.end # '2020-07-01'
    target_index = args.target
    target_map = [5, 10, 20, 6, 11, 21]

    for i in range (len(file_list_init)):
        file_name = file_list_init[i].split('/')[-1]
        if file_name.split('.')[0] <= test_end:
            file_list.append(file_list_init[i])

    # prevent data leaking
    file_list = file_list[:-target_map[target_index]]
    train_len = int(len(file_list) * 4 / 5)
    train_list = file_list[:train_len]
    test_list = file_list[train_len:]
    print(train_list)
    print(test_list)

    train_dataset = day_dataset_para(train_list)
    test_dataset = day_dataset_para(test_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gru = GRU_multi(input_size, hidden_size, output_size, num_layers, device)
    model = gru
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_test_loss = 100
    best_train_loss = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        neg_ic = 0
        l2_norm = 0
        for data, target in (train_loader):
            data, target = data.to(device), target[:, target_index].to(device)
            optimizer.zero_grad()
            # print("shit")
            # print(data.shape)

            output = model(data)

            loss, neg_ic_, l2_norm_ = loss_fn(output, target, L2_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            neg_ic += neg_ic_
            l2_norm += l2_norm_
            # print("shit")

        train_loss = train_loss / len(train_loader)
        neg_ic = neg_ic / len(train_loader)
        l2_norm = l2_norm / len(train_loader)
        print(f'Epoch: {epoch+1}, neg ic {neg_ic:.4f}, l2_norm {l2_norm:.4f}')
        if best_train_loss > train_loss: best_train_loss = train_loss
        # print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, train_loss))
        # 在每个epoch结束后对测试数据进行预测
        model.eval()
        valid_loss = 0
        test_output = []
        test_target = []
        neg_ic = 0
        l2_norm = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target[:, target_index].to(device)
                output = model(data)
                test_output.append(output)
                test_target.append(target)

            pred = torch.concat(test_output, dim=0)
            true = torch.concat(test_target).squeeze()
            valid_loss, neg_ic_, l2_norm_ = loss_fn(pred, true, L2_weight)
            neg_ic += neg_ic_
            l2_norm += l2_norm_
        print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f} neg ic {neg_ic:.4f}, l2_norm {l2_norm:.4f}')

        if best_test_loss > valid_loss:
            best_test_loss = valid_loss
            torch.save(model.state_dict(), f"./data/result{target_index}/{test_end}_gru.pt")
    