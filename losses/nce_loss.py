import torch
import torch.nn as nn
import torch.nn.functional as F

class NCELoss(nn.Module):
    def __init__(self):
        super(NCELoss, self).__init__()
        #self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion = torch.nn.MultiLabelMarginLoss(reduction='none')
        
    def forward(self, scores, pos_labels, mlabels, att_mask, obj_att_mask):
        scores = scores.masked_fill(att_mask.unsqueeze(1) == 0, -1e9)
        scores = F.log_softmax(scores, dim=-1)
        logits = (scores * pos_labels).sum(-1)
        logits = torch.masked_select(logits, obj_att_mask > 0)
        loss = -logits.mean()
        #scores = F.softmax(scores, dim=-1)
        #loss = self.criterion(scores.view(-1, scores.shape[-1]), mlabels.view(-1, mlabels.shape[-1]))
        #loss = loss.view(scores.shape[0], -1)
        #loss = torch.masked_select(loss, obj_att_mask > 0)
        #loss = loss.mean()

        _, pred = torch.max(scores.detach(), -1)
        pred_label = torch.gather(pos_labels, -1, pred.unsqueeze(-1)).squeeze(-1)
        pred_seq_correct = pred_label.sum(-1)
        gt_seq = obj_att_mask.sum(-1)
        accuracy = pred_seq_correct / gt_seq
        accuracy = accuracy.mean() * 100.0

        pos_sum_labels = pos_labels.sum(1)
        pred_set = torch.gather(pos_sum_labels, -1, pred)
        pred_set = torch.masked_select(pred_set, pred > 0)
        pred_set_mask = (pred_set > 0).type(torch.cuda.FloatTensor)
        set_accuracy = pred_set_mask.sum() / pred_set_mask.shape[0] * 100.0

        count = 0
        pred_info = []
        for i in range(0, pred.shape[0], 5):
            count += 1
            if count >= 5:
                break
            pred_info.append(str(list(pred[i].cpu().data.numpy())) + '/' + str(list(pred_label[i].cpu().data.numpy())))
        pred_info = '\n'.join(pred_info)

        return loss, {'NCE Loss': loss.item(), 'accuracy': accuracy.item(), 'set_accuracy': set_accuracy.item(), 'nce pred info': pred_info}