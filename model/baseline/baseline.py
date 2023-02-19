import numpy as np
import torch
import torch.nn as nn


class LstmModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.memory_cell = config.memory_cell
        self.body_dim = config.body_dim
        self.n = config.n
        self.lstm1 = nn.LSTM(1, self.memory_cell, batch_first=True)
        self.lstm2 = nn.LSTM(1, self.memory_cell, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(self.memory_cell*2+self.body_dim, config.n),
            nn.ReLU(),
            nn.Linear(config.n, 1),
        )

    def forward(self, x, b):
        #  self.lstm(input_seq.reshape(seq_len序列长度, batch批量大小, input_size特征维度), (h0, c0))
        x1, (hn, cn) = self.lstm1(x[..., 0].unsqueeze(-1))
        x2, (hn, cn) = self.lstm2(x[..., 1].unsqueeze(-1))
        x1 = x1[:, -1]
        x2 = x2[:, -1]
        x = torch.cat((x1, x2), dim=1)
        x = torch.cat((x, b), dim=1)
        x = self.fc(x)

        return x

