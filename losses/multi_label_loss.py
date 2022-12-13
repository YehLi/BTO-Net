import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelLoss(nn.Module):
    def __init__(self):
        super(MultiLabelLoss, self).__init__()
        self.criterion = torch.nn.MultiLabelMarginLoss(reduction='none')

    def forward(self, logit, label):
        scores = F.log_softmax(logit, dim=-1)
        scores = scores.view(-1, scores.shape[-1])
        label = label.view(-1, label.shape[-1])

        mask = (label > 0).type(torch.cuda.FloatTensor)
        mask_sum = mask.sum(-1)
        m_scores = (scores * mask).sum(-1)
        
        m_scores = torch.masked_select(m_scores, mask_sum > 0)
        mask_sum = torch.masked_select(mask_sum, mask_sum > 0)
        m_scores = -m_scores / mask_sum
        loss = m_scores.mean()

        _, pred = torch.max(scores.detach(), -1)
        pred_label = torch.gather(label, -1, pred.unsqueeze(-1)).squeeze(-1)
        pred_seq_correct = (pred_label > 0).type(torch.cuda.FloatTensor).sum(-1)
        gt_seq = (mask_sum > 0).type(torch.cuda.FloatTensor).sum(-1)
        accuracy = pred_seq_correct / gt_seq
        accuracy = accuracy.mean() * 100.0

        #scores = F.softmax(logit, dim=-1).view(-1, logit.shape[-1])
        #rangelabel = torch.arange(0, label.shape[-1]).cuda().unsqueeze(0)
        #rangelabel = rangelabel.type(torch.cuda.FloatTensor)
        #rangelabel = label.view(-1, label.shape[-1]) * rangelabel
        #mask = (label.view(-1, label.shape[-1]) > 0).type(torch.cuda.FloatTensor)
        #mlabel = rangelabel * mask + -1.0 * (1 - mask)
        #mlabel, _ = torch.sort(mlabel, -1, descending=True)
        #mlabel = mlabel.type(torch.cuda.LongTensor)
        #loss = self.criterion(scores, mlabel)
        #loss = loss.view(logit.shape[0], -1)

        #mask = (label > 0).type(torch.cuda.FloatTensor)
        #mask_sum = mask.sum(-1)
        #loss = torch.masked_select(loss, mask_sum > 0)
        #loss = loss.mean()

        return loss, {'Multi-label Loss': loss.item(), 'Multi-label Loss-Acc': accuracy.item()}
