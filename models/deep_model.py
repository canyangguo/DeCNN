import torch.nn as nn
import torch


class FNN(nn.Module):
    def __init__(self, input_num, transmission_num, output_num, t):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_num, transmission_num)
        self.fc2 = nn.Linear(transmission_num, output_num*16)
        self.fc3 = nn.Linear(output_num*16, output_num*4)
        self.fc4 = nn.Linear(output_num*4, output_num)
        self._t = t

    def forward(self, x):
        x = x.reshape(x.shape[0], self._t, -1)
        batch, t, node = x.shape
        x = x[:, -1, :]
        h = torch.sigmoid(self.fc1(x))
        y = torch.sigmoid(self.fc2(h))
        y = torch.sigmoid(self.fc3(y))
        y = self.fc4(y)
        y = y.reshape(batch, -1)
        return h, y


class RNN(nn.Module):
    def __init__(self, input_num, transmission_num, output_num, t):
        super(RNN, self).__init__()
        self._t = t
        self.rnn = nn.RNN(input_num, transmission_num, 1, batch_first=True, bias=True)
        self.fc2 = nn.Linear(transmission_num, output_num*16)
        self.fc3 = nn.Linear(output_num*16, output_num*4)
        self.fc4 = nn.Linear(output_num*4, output_num)
        self._t = t

    def forward(self, x):
        x = x.reshape(x.shape[0], self._t, -1)
        batch, t, node = x.shape
        y, _ = self.rnn(x)
        y = y[:, -1, :]
        h = torch.sigmoid(y)
        y = torch.sigmoid(self.fc2(h))
        y = torch.sigmoid(self.fc3(y))
        # h = torch.sigmoid(self.fc4(y))
        y = self.fc4(y)
        y = y.reshape(batch, -1)
        return h, y


class LSTM(nn.Module):
    def __init__(self, input_num, transmission_num, output_num, t):
        super(LSTM, self).__init__()
        self._t = t
        self.lstm = nn.LSTM(input_num, transmission_num, 1, batch_first=True, bias=True)
        self.fc2 = nn.Linear(transmission_num, output_num*16)
        self.fc3 = nn.Linear(output_num*16, output_num*4)
        self.fc4 = nn.Linear(output_num*4, output_num)
        self._t = t

    def forward(self, x):
        x = x.reshape(x.shape[0], self._t, -1)
        batch, t, node = x.shape
        y, _ = self.lstm(x)
        y = y[:, -1, :]
        h = torch.sigmoid(y)
        y = torch.sigmoid(self.fc2(h))
        y = torch.sigmoid(self.fc3(y))
        y = self.fc4(y)
        y = y.reshape(batch, -1)
        return h, y


class GRU(nn.Module):
    def __init__(self, input_num, transmission_num, output_num, t):
        super(GRU, self).__init__()
        self._t = t
        self.gru = nn.GRU(input_num, 8, 1, batch_first=True, bias=True)
        self.fc2 = nn.Linear(transmission_num, output_num*16)
        self.fc3 = nn.Linear(output_num*16, output_num*4)
        self.fc4 = nn.Linear(output_num*4, output_num)
        self._t = t

    def forward(self, x):
        x = x.reshape(x.shape[0], self._t, -1)
        batch, t, node = x.shape
        y, _ = self.gru(x)
        y = y[:, -1, :]
        h = torch.sigmoid(y)
        y = torch.sigmoid(self.fc2(h))
        y = torch.sigmoid(self.fc3(y))
        y = self.fc4(y)
        y = y.reshape(batch, -1)
        return h, y

