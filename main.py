import torch
import torch.nn as nn
import argparse

from utils import get_file_list, day_dataset, loss_fn
from torch.utils.data import DataLoader
from model import GRUModel, LSTMModel, AGRUModel

def train_model(name, model, train_loader, valid_loader, dir, target_index, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cpu')
    model.to(device)

    best_test_loss = 0
    best_train_loss = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target[:, target_index].to(device)
            optimizer.zero_grad()
            # print("shit")
            # print(data.shape)

            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        if best_train_loss > train_loss: best_train_loss = train_loss
        print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, train_loss))
        # 在每个epoch结束后对测试数据进行预测
        model.eval()
        test_loss = 0
        test_output = []
        test_target = []
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target[:, target_index].to(device)
                output = model(data)
                test_output.append(output)
                test_target.append(target)

            pred = torch.concat(test_output).squeeze()
            true = torch.concat(test_target).squeeze()
            test_loss = loss_fn(pred, true)
        if(epoch % 10 == 0):
            y = torch.cat((pred.view(1, -1), true.view(1, -1)), dim=0)
            print(pred)
            print(true)
        print('Epoch: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))

        if best_test_loss > test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f"{dir}{target_index}/{test_end}_{name}.pt")
    


# python main.py --end 2020-07-01 --target 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=str, help="end date", default="John")
    parser.add_argument("--target", type=int, help="specify which target to be used", default=0)
    args = parser.parse_args()

    # 参数
    input_size = 6 #每天的特征维度
    num_layers = 1 
    hidden_size = 64
    num_classes = 1 
    num_epochs = 100
    learning_rate = 0.0001
    batch_size = 128

    file_list_init = get_file_list('./data/day_datapoint/')[:20]
    file_list = []
    test_end = args.end # '2020-07-01'
    target_index = args.target

    for i in range (len(file_list_init)):
        if file_list_init[i][-14:] <= test_end:
            file_list.append(file_list_init[i])
    file_list = file_list[:-2]

    train_len = int(len(file_list) * 4 / 5)
    train_list = file_list[:train_len]
    test_list = file_list[train_len:]
    print(train_list)
    print(test_list)

    train_dataset = day_dataset(train_list)
    test_dataset = day_dataset(test_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=batch_size)

    gru = GRUModel(input_size, hidden_size, num_layers, num_classes)
    lstm = LSTMModel(input_size, hidden_size, num_layers, num_classes)
    agru = AGRUModel(input_size, hidden_size, num_layers, num_classes)

    train_model("gru", gru, train_loader, valid_loader, "./data/result_single", target_index, num_epochs)
    train_model("lstm", lstm, train_loader, valid_loader, "./data/result_single", target_index, num_epochs)
    train_model("agru", agru, train_loader, valid_loader, "./data/result_single", target_index, num_epochs)
