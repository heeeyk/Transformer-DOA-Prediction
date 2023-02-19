import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer('q', torch.tensor(config.q))

    def forward(self, predictions, targets):
        diff = predictions - targets
        ql = (1-self.q)*F.relu(diff) + self.q*F.relu(-diff)
        losses = ql.view(-1, ql.shape[-1]).mean(1)
        return losses
