import torch
import torch.nn as nn
import torch.nn.functional as F

class MilLoss(nn.Module):
    def __init__(self):
        super(MilLoss, self).__init__()

    def forward(self, logit, label):
        scores = F.log_softmax(logit, dim=-1)
        loss = (scores * label).sum(-1) / label.sum(-1)
        loss = -loss.mean()
        return loss, {'MIL Loss': loss.item()}
