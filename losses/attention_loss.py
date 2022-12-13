import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()

    # logit: batch_size, layernum, head_num, seq_len, att_num
    # label: batch_size, seq_len, att_num
    def forward(self, logit, label):
        scores = F.log_softmax(logit, dim=-1)
        scores = scores[:,:,:,:,1:]

        label = label.view(-1, 1, 1, label.shape[-2], label.shape[-1])
        label = label[:,:,:,0:scores.shape[-2],:]
        label = label.expand_as(scores).contiguous()

        scores = scores.view(-1, scores.shape[-1])
        label = label.view(-1, label.shape[-1])

        mask = (label > 0).type(torch.cuda.FloatTensor)
        mask_sum = mask.sum(-1)
        m_scores = (scores * mask).sum(-1)
        
        m_scores = torch.masked_select(m_scores, mask_sum > 0)
        mask_sum = torch.masked_select(mask_sum, mask_sum > 0)
        m_scores = -m_scores / mask_sum
        loss = m_scores.mean()

        return loss, {'Attention Loss': loss.item()}

class ObjAttentionLoss(nn.Module):
    def __init__(self):
        super(ObjAttentionLoss, self).__init__()

    # logit: batch_size, seq_len, att_num
    # label: batch_size, seq_len, att_num
    def forward(self, logit, label):
        scores = F.log_softmax(logit, dim=-1)
        #scores = scores[:,:,:,:,1:]

        label = label.view(-1, label.shape[-2], label.shape[-1])
        label = label[:,0:scores.shape[-2],:]
        label = label.expand_as(scores).contiguous()

        scores = scores.view(-1, scores.shape[-1])
        label = label.view(-1, label.shape[-1])

        mask = (label > 0).type(torch.cuda.FloatTensor)
        mask_sum = mask.sum(-1)
        m_scores = (scores * mask).sum(-1)
        
        m_scores = torch.masked_select(m_scores, mask_sum > 0)
        mask_sum = torch.masked_select(mask_sum, mask_sum > 0)
        m_scores = -m_scores / mask_sum
        loss = m_scores.mean()

        return loss, {'Attention Loss': loss.item()}
