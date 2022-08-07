import time
import torch
import numpy as np
from utils import data_process, evaluate
from utils.setting import log_string


def train(model, train_data, epochs, criterion, optimizer, device, gamma, alpha, log):
    T = time.time()
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        t = time.time()
        with torch.enable_grad():
            for i, (X, Y) in enumerate(train_data):
                X, Y = X.to(device), Y.to(device)
                H, P = model(X)
                cory = []
                if epoch < gamma:
                    for h in range(H.shape[1]):
                        for p in range(P.shape[1]):
                            cory.append(data_process.Pearsonr(P[:, p], H[:, h]))
                    P_loss = torch.max(torch.abs(torch.stack(cory)))
                    loss = criterion(P, Y) + alpha * P_loss
                else:
                    loss = criterion(P, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if (epoch + 1) % 100 == 0:
            log_string(log, 'Epoch: {:06d}'.format(epoch + 1) + '/{:06d}, '.format(epochs) +
                       'loss_train: {:.6f}, '.format(loss.item()) +
                       'time: {:.6f}s'.format(time.time() - t))

    log_string(log, 'Optimization Finished!')
    log_string(log, 'total_time: {:.6f}s.\n'.format(time.time() - T))


def test(model, Z, R, max, min, device):
    with torch.no_grad():  # without computing gd
        Z = Z.to(device)
        H, P = model(Z)
        cory = []
        for h in range(H.shape[1]):
            for p in range(P.shape[1]):
                cory.append(data_process.Pearsonr(P[:, p], H[:, h]))
        mean_cor = np.mean(np.abs(np.array(torch.stack(cory).detach().to('cpu'))))
        max_cor = np.max(np.abs(np.array(torch.stack(cory).detach().to('cpu'))))

        P = P.to('cpu')
        P = np.array(P)
        P[:, 0] = data_process.demaxmin(P[:, 0], max[0], min[0])
        P[:, 1] = data_process.demaxmin(P[:, 1], max[1], min[1])

        R[:, 0] = data_process.demaxmin(R[:, 0], max[0], min[0])
        R[:, 1] = data_process.demaxmin(R[:, 1], max[1], min[1])
        error = evaluate.dis(R, P)

    return error, mean_cor, max_cor
