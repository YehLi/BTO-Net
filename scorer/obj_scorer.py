import os
import sys
import numpy as np
import torch

class ObjScorer(object):
    def __init__(self):
        super(ObjScorer, self).__init__()

    def __call__(self, pred, pos_labels, obj_att_mask):
        batch_size = pred.shape[0]
        max_len = max(pred.shape[-1], obj_att_mask.shape[-1])
        if pred.shape[-1] < max_len:
            tmp = torch.zeros(batch_size, max_len).cuda()
            tmp[:, 0:pred.shape[-1]] = pred
            pred = tmp
        if pos_labels.shape[1] < max_len:
            pos_labels_tmp = torch.zeros(batch_size, max_len, pos_labels.shape[-1]).cuda()
            obj_att_mask_tmp = torch.zeros(batch_size, max_len).cuda()
            pos_labels_tmp[:, 0:pos_labels.shape[1], :] = pos_labels
            obj_att_mask_tmp[:, 0:pos_labels.shape[1]] = obj_att_mask
            pos_labels = pos_labels_tmp
            obj_att_mask = obj_att_mask_tmp
        pred = pred.type(torch.cuda.LongTensor)

        pred_label = torch.gather(pos_labels, -1, pred.unsqueeze(-1)).squeeze(-1)
        pred_seq_correct = pred_label.sum(-1)
        gt_seq = obj_att_mask.sum(-1)
        accuracy = pred_seq_correct / gt_seq
        accuracy = accuracy.data.cpu().numpy()

        #pos_sum_labels = pos_labels.sum(1)
        #pos_sum_labels = (pos_sum_labels > 0).type(torch.cuda.FloatTensor)
        #pred_set = torch.gather(pos_sum_labels, -1, pred).type(torch.cuda.FloatTensor)
        #pred_set = (pred > 0).type(torch.cuda.FloatTensor) * pred_set
        #set_accuracy = pred_set.sum(-1) / (gt_seq - 1)
        #set_accuracy = set_accuracy.data.cpu().numpy()

        rewards = accuracy #+ 0.5 * set_accuracy

        rewards_info = {}
        rewards_info['accuracy'] = accuracy.mean()
        #rewards_info['set_accuracy'] = set_accuracy.mean()
        return rewards, rewards_info