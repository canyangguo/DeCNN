import pandas as pd
import numpy as np
import json
import torch
import torch.utils.data as Data
from utils.setting import log_string


def load_data(config_name, log):
    with open(config_name, 'r') as f:
        config = json.loads(f.read())
    data = pd.read_csv(config['data_path'], header=None)
    data = data.iloc[:, :]  # X + Y + CELL + WIFI
    data = data.values[1:, :]
    data = pd.DataFrame(data, dtype='float32').values  # string to chat

    max = [np.max(data[:, 0: 1]),
           np.max(data[:, 1: config['gps_num']]),
           np.max(data[:, config['gps_num']:config['gps_num'] + config['cell_num']]),
           np.max(data[:, config['gps_num'] + config['cell_num']:])]
    min = [np.min(data[:, 0: 1]),
           np.min(data[:, 1: config['gps_num']]),
           np.min(data[:, config['gps_num']:config['gps_num'] + config['cell_num']]),
           np.min(data[:, config['gps_num'] + config['cell_num']:])]

    lat = maxmin(data[config['time_step']-1:, 0:1], max[0], min[0])
    lng = maxmin(data[config['time_step']-1:, 1:config['gps_num']], max[1], min[1])  # max-min normalized
    normalized_cell = maxmin(data[:, config['gps_num']:config['gps_num'] + config['cell_num']], max[2], min[2])
    normalized_wifi = maxmin(data[:, config['gps_num'] + config['cell_num']:], max[3], min[3])
    normalized = np.c_[normalized_cell, normalized_wifi]
    normalized = append_sequence(normalized, config['time_step'])
    normalized_data = np.c_[lat, lng, normalized]

    X, Y, Z, R = [], [], [], []

    for i in range(0, normalized_data.shape[0]):
        if i % 2 == 0:
            X.append(normalized_data[i, 2:])
            Y.append(normalized_data[i, :2])
        else:
            Z.append(normalized_data[i, 2:])
            R.append(normalized_data[i, :2])

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    R = np.array(R)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    Z = torch.tensor(Z, dtype=torch.float32)
    R = torch.tensor(R, dtype=torch.float32)

    log_string(log, 'training_input: {}, training_output:{}, testing_input: {}, testing_output: {}\n'.format(
        X.shape, Y.shape, Z.shape, R.shape))

    train_data = Data.TensorDataset(X, Y)
    train_data = Data.DataLoader(dataset=train_data,
                                 batch_size=config['batch_size'],
                                 shuffle=True)

    return train_data, Z, R, max[0:2], min[0:2]


def demaxmin(X, max, min):
    return X * (max - min) + min


def maxmin(X, max, min):
    return (X-min) / (max-min)


def append_sequence(X, t):
    table = X[:len(X) - t + 1]
    for i in range(1, t):
        temp = X[i:len(X) - (t - i - 1)]
        table = np.c_[table, temp]
    return np.array(table)


def Pearsonr(y_ob, y_pred):
    Epsilon = 1e-6
    Cov = torch.sum(((y_pred - torch.mean(y_pred))+Epsilon)
                    * (Epsilon + (y_ob - torch.mean(y_ob))))
    Std1 = (torch.sum(((y_ob - torch.mean(y_ob)) + Epsilon) ** 2) ** 0.5)
    Std2 = (torch.sum(((y_pred - torch.mean(y_pred)) + Epsilon) ** 2) ** 0.5)
    return (Cov / (Std1 * Std2))






