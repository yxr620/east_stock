import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import argparse

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from utils import loss_fn

def add_month(date):
    # Add a month
    date = date + relativedelta(months=2)
    date = date.replace(day=1) - relativedelta(days=1)

    return date

def pred_month(start, endtrain, starttest, endtest, target, data):

    # Step 1: Load and Prepare the Data
    print(data.shape)

    info = data[:, :2]
    y = data[:, 2:8].astype(float)
    X = data[:, 8:].astype(float)

    print(y.shape)
    print(X.shape)

    # get the index which are used to split dataset
    train_index = 0
    test_index = 0
    test_start_index = 0
    for row in data:
        if row[1] > endtrain:
            break
        train_index += 1
    for row in data:
        if row[1] > endtest:
            break
        test_index += 1
    for row in data:
        if row[1] > starttest: break
        test_start_index += 1


    print(train_index)
    split_index = int(train_index * 0.8)
    # exit()
    X_train, X_val, X_test = X[:split_index, :], X[split_index: train_index, :], X[test_start_index:test_index, :]
    y_train, y_val, y_test = y[:split_index, :], y[split_index: train_index, :], y[test_start_index:test_index, :]
    info_test = info[test_start_index:test_index, :]

    # Step 2: Create a LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train[:, target])
    val_data = lgb.Dataset(X_val, label=y_val[:, target])

    # Step 3: Set Up Parameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # Step 4: Train the Model
    num_boost_round = 100
    model = lgb.train(params, train_data, num_boost_round=num_boost_round, valid_sets=[val_data])

    # Step 5: Save the Model
    model.save_model(f"./data/result{target}/{start}_{endtest}_lgb.txt")

    # Step 6: Load the Model
    model = lgb.Booster(model_file=f"./data/result{target}/{start}_{endtest}_lgb.txt")

    # Step 7: Make Predictions on New Data
    # Assuming you have new data stored in a DataFrame called 'new_data'
    y_pred = model.predict(X_val)
    print(y_pred.shape)
    print(y_val[:, 0].shape)
    print(loss_fn(torch.tensor(y_pred), torch.tensor(y_val[:, target])))

    y_test_pred = model.predict(X_test)
    test_loss = loss_fn(torch.tensor(y_test_pred), torch.tensor(y_test[:, target]))
    print(test_loss)

    result = np.concatenate([info_test, y_test_pred.reshape((-1, 1))], axis=1)
    print(result.shape)
    table = pd.DataFrame(result, columns=["stock_code", "date", "pred"])
    group = table.groupby("date")
    keys = group.groups.keys()
    for key in keys:
        print(f"{key} {group.get_group(key).shape}")

    # exit()
    table.to_csv(f"./data/result{target}/{starttest}_{endtest}_pred.csv", index=False)




# python gen_embedding.py --start 2020-01-01 --endtrain 2020-04-30 --endtest 2020-05-31 --target 0
# python lgb_train.py --start 2016-01-01 --endtrain 2016-12-31 --endtest 2017-12-31 --target 0
# python lgb_train.py --start 2017-01-01 --endtrain 2017-12-31 --endtest 2018-12-31 --target 0
# python lgb_train.py --start 2018-01-01 --endtrain 2018-12-31 --endtest 2019-12-31 --target 0
# python lgb_train.py --start 2019-01-01 --endtrain 2019-12-31 --endtest 2020-12-31 --target 0
# python lgb_train.py --start 2020-01-01 --endtrain 2020-12-31 --endtest 2021-12-31 --target 0
# python lgb_train.py --start 2021-01-01 --endtrain 2021-12-31 --endtest 2022-12-31 --target 0
# python lgb_train.py --start 2022-01-01 --endtrain 2022-12-31 --endtest 2023-12-31 --target 0
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="start date of training", default="2020-01-01")
    parser.add_argument("--endtrain", type=str, help="end of training data, start of testing", default="2020-04-30")
    parser.add_argument("--endtest", type=str, help="end of testing data ", default="2020-05-31")
    parser.add_argument("--target", type=int, help="specify which target to be used", default=0)
    args = parser.parse_args()

    start = args.start
    endtrain = args.endtrain
    endtest = args.endtest
    target_index = args.target
    target_map = [5, 10, 20, 6, 11, 21]

    data = np.loadtxt(f'./data/result{target_index}/{start}_{endtest}.txt', delimiter=' ', dtype=str)
    pre_date = datetime.strptime(endtrain, "%Y-%m-%d")
    iter_date = add_month(pre_date)
    end_date = datetime.strptime(endtest, "%Y-%m-%d")
    while iter_date <= end_date:
        end_train = pre_date - relativedelta(days=target_map[target_index]) # avoid data leaking
        start_test = pre_date
        end_test = iter_date
        print(f"start {start}, end_train {end_train}, start_test{start_test},  end_test {end_test}")
        pred_month(start, end_train.strftime("%Y-%m-%d"), start_test.strftime("%Y-%m-%d"), end_test.strftime("%Y-%m-%d"), target_index, data)

        # update pre_date and iter_date 
        pre_date = iter_date
        iter_date = add_month(iter_date)
