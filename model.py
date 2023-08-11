import torch
import torch.nn as nn

from torch.nn.utils.weight_norm import weight_norm


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class GRU_multi(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(torch.device('cuda'))
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.bn(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class AGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.gru = weight_norm.weight_norm(nn.GRU(input_size, hidden_size, num_layers, batch_first=True))
        self.gru = weight_norm(
            nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
