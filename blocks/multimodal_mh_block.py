import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.multihead_attention_block import BertAttention, BertOutput

class MBertOutput(nn.Module):
    def __init__(self, hidden_size, obj_hidden_size, drop=0., use_aoa=False, aoa_drop=0.):
        super(MBertOutput, self).__init__()
        self.use_aoa = use_aoa
        if self.use_aoa:
            self.aoa = nn.Sequential(nn.Linear(2 * hidden_size + obj_hidden_size, 2 * hidden_size), nn.GLU())
            self.aoa_drop = nn.Dropout(aoa_drop) if aoa_drop > 0 else None
        else:
            self.dense = nn.Linear(hidden_size + obj_hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(drop) if drop > 0 else None

    def forward(self, hidden_states, input_tensor):
        if self.use_aoa:
            hq = torch.cat([hidden_states, input_tensor], dim=-1)
            if self.aoa_drop is not None:
                hq = self.aoa_drop(hq)
            hidden_states = self.aoa(hq)
        else:
            hidden_states = self.dense(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

class MultimodalMHBlock(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        obj_hidden_size,
        num_attention_heads, 
        attention_dropout=0., 
        dropout=0.,
        use_aoa=False,
        aoa_drop=0.,
    ):
        super(MultimodalMHBlock, self).__init__()
        self.mh_att = BertAttention(
            hidden_size, 
            num_attention_heads, 
            attention_dropout
        )
        self.obj_mh_att = BertAttention(
            obj_hidden_size, 
            num_attention_heads, 
            attention_dropout,
            q_hidden_size=hidden_size
        )
        self.output = MBertOutput(hidden_size, obj_hidden_size, dropout, use_aoa, aoa_drop)

    def forward(self, x, key, value, obj_key, obj_value, attention_mask=None, obj_attention_mask=None):
        mh_output, mh_att_logit = self.mh_att(
            query=x,
            key=key,
            value=value,
            attention_mask=attention_mask
        )
        obj_mh_output, obj_mh_att_logit = self.obj_mh_att(
            query=x,
            key=obj_key,
            value=obj_value,
            attention_mask=obj_attention_mask
        )
        hidden_states = torch.cat([mh_output, obj_mh_output], dim=-1)
        attention_output = self.output(hidden_states, x)
        return attention_output, mh_att_logit, obj_mh_att_logit


class MultimodalSEBlock(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        obj_hidden_size,
        num_attention_heads, 
        attention_dropout=0., 
        dropout=0.,
        use_aoa=False,
        aoa_drop=0.,
    ):
        super(MultimodalSEBlock, self).__init__()
        self.mh_att = BertAttention(
            hidden_size, 
            num_attention_heads, 
            attention_dropout
        )
        self.obj_mh_att = BertAttention(
            obj_hidden_size, 
            num_attention_heads, 
            attention_dropout,
            q_hidden_size=hidden_size
        )

        self.se = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2, 1),
            nn.LayerNorm(hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size//2, hidden_size*2, 1)
        )

        self.output = BertOutput(hidden_size, dropout, use_aoa, aoa_drop)

    def forward(self, x, key, value, obj_key, obj_value, attention_mask=None, obj_attention_mask=None):
        mh_output, mh_att_logit = self.mh_att(
            query=x,
            key=key,
            value=value,
            attention_mask=attention_mask
        )
        obj_mh_output, obj_mh_att_logit = self.obj_mh_att(
            query=x,
            key=obj_key,
            value=obj_value,
            attention_mask=obj_attention_mask
        )

        hidden_states = torch.cat([mh_output.unsqueeze(-2), obj_mh_output.unsqueeze(-2)], dim=-2)
        output_sum = hidden_states.sum(-2)
        attn = self.se(output_sum)
        B, T, C = output_sum.size()
        attn = attn.view(B, T, 2, C)
        attn = F.softmax(attn, dim=2)
        hidden_states = (attn * hidden_states).sum(-2)
        attention_output = self.output(hidden_states, x)
        return attention_output, mh_att_logit, obj_mh_att_logit


class MultimodalSEBlock2(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        obj_hidden_size,
        num_attention_heads, 
        attention_dropout=0., 
        dropout=0.,
        use_aoa=False,
        aoa_drop=0.,
    ):
        super(MultimodalSEBlock2, self).__init__()
        self.mh_att = BertAttention(
            hidden_size, 
            num_attention_heads, 
            attention_dropout
        )
        self.obj_mh_att = BertAttention(
            obj_hidden_size, 
            num_attention_heads, 
            attention_dropout,
            q_hidden_size=hidden_size
        )

        self.output = BertOutput(hidden_size, dropout, use_aoa, aoa_drop)
        self.obj_output = BertOutput(obj_hidden_size, dropout, use_aoa, aoa_drop)
        
        scale = 2
        self.se = nn.Sequential(
            nn.Linear(hidden_size * scale, hidden_size//2, 1),
            nn.LayerNorm(hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size//2, hidden_size*2, 1)
        )

    def forward(self, x, key, value, obj_key, obj_value, attention_mask=None, obj_attention_mask=None):
        mh_output, mh_att_logit = self.mh_att(
            query=x,
            key=key,
            value=value,
            attention_mask=attention_mask
        )
        obj_mh_output, obj_mh_att_logit = self.obj_mh_att(
            query=x,
            key=obj_key,
            value=obj_value,
            attention_mask=obj_attention_mask
        )

        attention_output = self.output(mh_output, x)
        obj_attention_output = self.obj_output(obj_mh_output, x)

        hidden_states = torch.cat([attention_output.unsqueeze(-2), obj_attention_output.unsqueeze(-2)], dim=-2)
        B, T, S, C = hidden_states.size()
        
        #output_sum = hidden_states.sum(-2)
        output_sum = hidden_states.view(B, T, S*C)

        attn = self.se(output_sum)
        attn = attn.view(B, T, 2, C)
        attn = F.softmax(attn, dim=2)
        hidden_states = (attn * hidden_states).sum(-2)

        return hidden_states, mh_att_logit, obj_mh_att_logit