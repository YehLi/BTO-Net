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
from layers.objattention import ObjAttention

class ObjEncoder(nn.Module):
    def __init__(self, embed_dim, att_hidden_drop, dropout1, dropout2):
        super(ObjEncoder, self).__init__()
        self.num_layers = 2
        self.embed_dim = embed_dim

        self.bos = nn.Embedding(1, embed_dim)
        
        self.p_att_feats = nn.Linear(embed_dim, embed_dim)

        # First LSTM layer
        self.lstm1 = nn.LSTMCell(embed_dim * 3, embed_dim)

        # Second LSTM Layer
        self.lstm2 = nn.LSTMCell(2 * embed_dim, embed_dim)
        self.att = ObjAttention(embed_dim, att_hidden_drop)
        self.dropout1 = nn.Dropout(dropout1) if dropout1 > 0 else None
        self.dropout2 = nn.Dropout(dropout2) if dropout2 > 0 else None

        att_feat_embed = []
        obj_feat_embed = []
        last_dim = embed_dim
        for mlp_dim in cfg.LOSSES.NCE.MLP:
            if len(att_feat_embed) > 0:
                att_feat_embed.append(nn.ReLU(inplace=True))
            att_feat_embed.append(nn.Linear(last_dim, mlp_dim))

            if len(obj_feat_embed) > 0:
                obj_feat_embed.append(nn.ReLU(inplace=True))
            obj_feat_embed.append(nn.Linear(last_dim, mlp_dim))
            last_dim = mlp_dim
        if len(att_feat_embed) == 0:
            att_feat_embed.append(nn.Identity())
        if len(obj_feat_embed) == 0:
            obj_feat_embed.append(nn.Identity())

        self.att_feat_embed = nn.Sequential(*att_feat_embed)
        self.obj_feat_embed = nn.Sequential(*obj_feat_embed)
        self.eos = nn.Embedding(1, last_dim)

    def init_hidden(self, batch_size):
        return [Variable(torch.zeros(self.num_layers, batch_size, self.embed_dim).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size, self.embed_dim).cuda())]

    # encoder_out: gv_feat, feat1, feat2, feat3,...,featn
    def forward(self, encoder_out, obj_seq, att_mask):
        batch_size = encoder_out.size(0)
        state = self.init_hidden(batch_size)

        obj_seq = obj_seq.unsqueeze(-1).expand(batch_size, obj_seq.size(1), self.embed_dim)
        gv_feat = encoder_out[:,0,:]
        att_feats = encoder_out[:, 1:, :]
        obj_feat =  torch.gather(att_feats, 1, obj_seq)

        bos_index = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
        bos_embed = self.bos(bos_index)
        obj_feat = torch.cat([bos_embed, obj_feat], dim=1)
        p_att_feats = self.p_att_feats(att_feats)
        
        obj_len = obj_feat.size(1)
        outputs = Variable(torch.zeros(batch_size, obj_len, self.embed_dim).cuda())
        for t in range(obj_len):
            # lstm1
            h2_tm1 = state[0][-1]
            input1 = torch.cat([h2_tm1, gv_feat, obj_feat[:, t, :]], 1)
            h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
            att = self.att(h1_t, att_feats, p_att_feats, att_mask[:, 1:])

            # lstm2
            input2 = torch.cat([att, h1_t], 1)
            if self.dropout1 is not None:
                input2 = self.dropout1(input2)
            h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))

            state = (torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t]))
            outputs[:, t, :] = h2_t

        if self.dropout2 is not None:
            outputs = self.dropout2(outputs)

        nce_att_feats = att_feats.detach() if cfg.LOSSES.NCE.STOP_GRAD else att_feats
        att_feat_embed = self.att_feat_embed(nce_att_feats)
        eos_index = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
        eos_embed = self.eos(eos_index)
        att_feat_embed = torch.cat([eos_embed, att_feat_embed], dim=1)
        obj_feat_embed = self.obj_feat_embed(outputs)
        nce_logit = torch.matmul(obj_feat_embed, att_feat_embed.transpose(-1, -2))

        return outputs, nce_logit

    def decode(self, encoder_out, att_mask, **kwargs):
        batch_size = encoder_out.size(0)
        state = self.init_hidden(batch_size)
        gv_feat = encoder_out[:,0,:]
        att_feats = encoder_out[:, 1:, :]
        p_att_feats = self.p_att_feats(att_feats)

        bos_index = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
        bos_embed = self.bos(bos_index).squeeze(1)

        max_obj_len = cfg.MODEL.BTONET.MAX_LEN
        outputs = Variable(torch.zeros(batch_size, max_obj_len, self.embed_dim).cuda())
        obj_att_mask = Variable(torch.zeros(batch_size, max_obj_len)).cuda()
        cur_obj_feat = bos_embed

        att_feat_embed = self.att_feat_embed(att_feats)
        eos_index = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
        eos_embed = self.eos(eos_index)
        att_feat_embed = torch.cat([eos_embed, att_feat_embed], dim=1)

        unfinished = bos_index.eq(bos_index).squeeze(-1)
        obj_att_mask[:, 0] = unfinished.type(torch.cuda.FloatTensor)

        # t = 0              t = 1              t=2
        # wt=[1,0,1]         wt=[0,0,1]         wt=[0,0,0]
        # h_0                h_1                h_2
        # mask[:,1]=[1,0,1]  mask[:,2]=[0,0,1]  mask[:,3]=[0,0,0]

        for t in range(max_obj_len):
            # lstm1
            h2_tm1 = state[0][-1]
            input1 = torch.cat([h2_tm1, gv_feat, cur_obj_feat], 1)
            h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
            att = self.att(h1_t, att_feats, p_att_feats, att_mask[:, 1:])

            # lstm2
            input2 = torch.cat([att, h1_t], 1)
            if self.dropout1 is not None:
                input2 = self.dropout1(input2)
            h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))

            state = (torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t]))
            if self.dropout2 is not None:
               h2_t = self.dropout2(h2_t)
            outputs[:, t, :] = h2_t

            obj_feat_embed = self.obj_feat_embed(h2_t)
            scores = torch.matmul(obj_feat_embed.unsqueeze(1), att_feat_embed.transpose(-1, -2)).squeeze(1)
            scores = scores.masked_fill(att_mask == 0, -1e9)

            score, wt = torch.max(scores, 1)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)

            if t != max_obj_len - 1:
                obj_att_mask[:, t+1] = unfinished.type(torch.cuda.FloatTensor)

            if unfinished.sum() == 0:
                break

            wt = wt.view(-1, 1, 1).expand(wt.shape[0], 1, self.embed_dim)
            cur_obj_feat = torch.gather(encoder_out, 1, wt).squeeze(1)

        return outputs, obj_att_mask

    def select(self, t, candidate_logprob, batch_size, beam_size):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def decode_beam(self, encoder_out, att_mask, **kwargs):
        batch_size = encoder_out.size(0)
        beam_size = kwargs['BEAM_SIZE']
        max_obj_len = cfg.MODEL.BTONET.MAX_LEN

        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()
        bos_index = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
        bos_embed = self.bos(bos_index).squeeze(1)
        cur_obj_feat = bos_embed

        state = self.init_hidden(batch_size)
        gv_feat = encoder_out[:,0,:]
        att_feats = encoder_out[:, 1:, :]
        p_att_feats = self.p_att_feats(att_feats)
        att_feats_mask = att_mask

        att_feat_embed = self.att_feat_embed(att_feats)
        eos_index = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
        eos_embed = self.eos(eos_index)
        att_feat_embed = torch.cat([eos_embed, att_feat_embed], dim=1)
  
        outputs = []
        outputs_feat = []
        obj_att_mask = []
        obj_att_mask.append(torch.ones(batch_size, 1, 1).cuda())

        for t in range(max_obj_len):
            cur_beam_size = 1 if t == 0 else beam_size

            # lstm1
            h2_tm1 = state[0][-1]
            input1 = torch.cat([h2_tm1, gv_feat, cur_obj_feat], 1)
            h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
            att = self.att(h1_t, att_feats, p_att_feats, att_feats_mask[:, 1:])

            # lstm2
            input2 = torch.cat([att, h1_t], 1)
            if self.dropout1 is not None:
                input2 = self.dropout1(input2)
            h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))

            state = (torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t]))
            if self.dropout2 is not None:
               h2_t = self.dropout2(h2_t)

            obj_feat_embed = self.obj_feat_embed(h2_t)
            scores = torch.matmul(obj_feat_embed.unsqueeze(1), att_feat_embed.transpose(-1, -2)).squeeze(1)
            scores = scores.masked_fill(att_feats_mask == 0, -1e9)
            word_logprob = F.log_softmax(scores, dim=-1)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(t, candidate_logprob, batch_size, beam_size)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            #################################
            h_tmp = []
            c_tmp = []
            for i in range(self.num_layers):
                shape = [int(sh) for sh in state[0][i].shape]
                beam = selected_beam
                for _ in shape[1:]:
                    beam = beam.unsqueeze(-1)
                h_tmp.append(
                    torch.gather(state[0][i].view(*([batch_size, cur_beam_size] + shape[1:])), 1, 
                        beam.expand(*([batch_size, beam_size] + shape[1:]))
                    ).view(*([-1, ] + shape[1:]))
                )
                c_tmp.append(
                    torch.gather(state[1][i].view(*([batch_size, cur_beam_size] + shape[1:])), 1, 
                        beam.expand(*([batch_size, beam_size] + shape[1:]))
                    ).view(*([-1, ] + shape[1:]))
                )
            state = (torch.stack(h_tmp, 0), torch.stack(c_tmp, 0))
            #################################

            seq_logprob = selected_logprob.unsqueeze(-1)

            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            if t == 0:
                outputs_feat.append(h2_t.unsqueeze(1))
            else:
                outputs_feat.append(h2_t.view(batch_size, beam_size, -1))
            outputs_feat = list(torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, self.embed_dim)) for o in outputs_feat)

            obj_att_mask = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in obj_att_mask)
            if t != max_obj_len - 1:
                tmp_mask = (selected_words.view(batch_size, beam_size) != 0).float().unsqueeze(-1)
                tmp_seq_mask = seq_mask * tmp_mask
                obj_att_mask.append(tmp_seq_mask)


            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            ys = selected_words
            
            if t == 0:
                gv_feat = utils.expand_tensor_beam(gv_feat, beam_size)
                att_feats = utils.expand_tensor_beam(att_feats, beam_size)
                p_att_feats = utils.expand_tensor_beam(p_att_feats, beam_size)
                att_feats_mask = utils.expand_tensor_beam(att_feats_mask, beam_size)
                encoder_out = utils.expand_tensor_beam(encoder_out, beam_size)
                att_feat_embed = utils.expand_tensor_beam(att_feat_embed, beam_size)
            cur_obj_feat = torch.gather(encoder_out, 1, ys.unsqueeze(-1).expand(ys.shape[0], 1, self.embed_dim)).squeeze(1)

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, max_obj_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, max_obj_len))

        outputs_feat = [f.unsqueeze(2) for f in outputs_feat]
        outputs_feat = torch.cat(outputs_feat, 2)
        outputs_feat = torch.gather(outputs_feat, 1, sort_idxs.unsqueeze(-1).expand(batch_size, beam_size, max_obj_len, self.embed_dim))
        obj_att_mask = torch.cat(obj_att_mask, -1)
        obj_att_mask = torch.gather(obj_att_mask, 1, sort_idxs.expand(batch_size, beam_size, max_obj_len))

        out_size = 1
        outputs = outputs.contiguous()[:, :out_size].squeeze(1)
        outputs_feat = outputs_feat.contiguous()[:, :out_size].squeeze(1)
        obj_att_mask = obj_att_mask.contiguous()[:, :out_size].squeeze(1)

        return outputs_feat, obj_att_mask