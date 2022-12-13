import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BertAttention(nn.Module):
    def __init__(
        self,
        hidden_size, 
        num_attention_heads, 
        dropout=0.,
        q_hidden_size=None
    ):
        super(BertAttention, self).__init__()
        assert hidden_size % num_attention_heads == 0

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, hidden_size) if q_hidden_size is None else nn.Linear(q_hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)  # B, T, C -> B, T, head_num, head_size
        shape_list = list(range(len(new_x_shape)))  # 0, 1, 2, 3
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2] # 0, 2, 1, 3
        return x.permute(shape_list)

    def forward(self, query, key, value, attention_mask, history_states=None):
        mixed_query_layer = self.query(query)

        if history_states is not None:
            mixed_key_layer = self.key(history_states)
            mixed_value_layer = self.value(history_states)
        else:
            mixed_key_layer = self.key(key)
            mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # B, head_num, T, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)      # B, head_num, T, head_size
        value_layer = self.transpose_for_scores(mixed_value_layer)  # B, head_num, T, head_size

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        
        
        #attention_scores = attention_scores + attention_mask
        attention_mask = attention_mask.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)


        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        shape_list = list(range(len(context_layer.shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        context_layer = context_layer.permute(shape_list).contiguous()   # B, T, head, head_size

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_scores

class BertOutput(nn.Module):
    def __init__(self, hidden_size, drop=0., use_aoa=False, aoa_drop=0.):
        super(BertOutput, self).__init__()
        self.use_aoa = use_aoa
        if self.use_aoa:
            self.aoa = nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size), nn.GLU())
            self.aoa_drop = nn.Dropout(aoa_drop) if aoa_drop > 0 else None
        else:
            self.dense = nn.Linear(hidden_size, hidden_size)
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

class MultiheadAttentionBlock(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        num_attention_heads, 
        attention_dropout=0., 
        dropout=0.,
        use_aoa=False,
        aoa_drop=0.,
    ):
        super(MultiheadAttentionBlock, self).__init__()
        self.mh_att = BertAttention(
            hidden_size, 
            num_attention_heads, 
            attention_dropout
        )
        self.output = BertOutput(hidden_size, dropout, use_aoa, aoa_drop)

    def forward(self, x, key=None, value=None, attention_mask=None, history_states=None):
        mh_output, att_logit = self.mh_att(
            query=x,
            key=key if key is not None else x,
            value=value if value is not None else x,
            attention_mask=attention_mask,
            history_states=history_states
        )
        attention_output = self.output(mh_output, x)
        return attention_output, att_logit