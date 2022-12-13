import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq, logP, rewards):
        mask = seq > 0
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        rewards = rewards.view(-1, 1).expand_as(logP)
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        loss = torch.mean(-logP * rewards)
        return loss

class ObjRewardCriterion(nn.Module):
    def __init__(self):
        super(ObjRewardCriterion, self).__init__()

    def forward(self, nce_logit, obj_seq, rewards):
        obj_seq = obj_seq.type(torch.cuda.LongTensor)
        mask = obj_seq > 0
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        scores = F.log_softmax(nce_logit, dim=-1)
        logP = torch.gather(scores, -1, obj_seq.unsqueeze(-1)).squeeze(-1)
        rewards = rewards.view(-1, 1).expand_as(logP)
        
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        loss = torch.mean(-logP * rewards)
        return loss
