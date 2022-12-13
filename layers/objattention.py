import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

import blocks
import lib.utils as utils

class ObjAttention(nn.Module):
    def __init__(self, embed_dim, att_hidden_drop):
        super(ObjAttention, self).__init__()
        self.Wah = nn.Linear(embed_dim, embed_dim, bias=False)
        self.alpha = nn.Linear(embed_dim, 1, bias=False)
        self.dropout = nn.Dropout(att_hidden_drop) if att_hidden_drop > 0 else None
        self.act = nn.Tanh()

    def forward(self, h, att_feats, p_att_feats, att_mask):
        Wah = self.Wah(h).unsqueeze(1)
        alpha = self.act(Wah + p_att_feats)
        if self.dropout is not None:
            alpha = self.dropout(alpha)

        alpha = self.alpha(alpha).squeeze(-1)
        alpha_logit = alpha.masked_fill(att_mask == 0, -1e9)
        alpha = F.softmax(alpha_logit, dim=-1)
        att = torch.bmm(alpha.unsqueeze(1), att_feats).squeeze(1)
        return att, alpha_logit