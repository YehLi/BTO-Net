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
from layers.positional_encoding import PositionalEncoding

class ObjEncoderPtrw(nn.Module):
    def __init__(self, embed_dim, att_hidden_drop, dropout1, dropout2):
        super(ObjEncoderPtrw, self).__init__()
        self.num_layers = 2
        self.embed_dim = embed_dim

        self.bos = nn.Embedding(1, embed_dim)
        
        self.p_att_feats = nn.Linear(embed_dim, embed_dim)

        # First LSTM layer
        self.lstm1 = nn.LSTMCell(embed_dim * 4, embed_dim)

        # Second LSTM Layer
        self.lstm2 = nn.LSTMCell(2 * embed_dim, embed_dim)
        self.att = ObjAttention(embed_dim, att_hidden_drop)
        self.dropout1 = nn.Dropout(dropout1) if dropout1 > 0 else None
        #self.dropout2 = nn.Dropout(dropout2) if dropout2 > 0 else None

        self.embed_positions = PositionalEncoding(
            embed_dim, cfg.MODEL.TRANSFORMER.PE_MAX_LEN
        )

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

        self.obj_vocab = utils.load_lines(cfg.DATA_LOADER.OBJ_VOCAB_PATH)
        self.word_embed = nn.Embedding(len(self.obj_vocab)+1, embed_dim)
        self.word_logit = nn.Linear(embed_dim, len(self.obj_vocab)+1)


    def init_hidden(self, batch_size):
        return [Variable(torch.zeros(self.num_layers, batch_size, self.embed_dim).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size, self.embed_dim).cuda())]

    # encoder_out: gv_feat, feat1, feat2, feat3,...,featn
    def forward(self, encoder_out, obj_seq, att_mask, obj_seq_label=None):
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

        obj_index = torch.cat([bos_index, obj_seq_label], dim=1)
        obj_word_embed = self.word_embed(obj_index)

        nce_att_feats = att_feats.detach() if cfg.LOSSES.NCE.STOP_GRAD else att_feats
        att_feat_embed = self.att_feat_embed(nce_att_feats)
        eos_index = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
        eos_embed = self.eos(eos_index)
        att_feat_embed = torch.cat([eos_embed, att_feat_embed], dim=1)

        if cfg.LOSSES.NCE.NORM == True:
            att_feat_embed = F.normalize(att_feat_embed, p=2, dim=-1)
        
        obj_len = obj_feat.size(1)
        outputs_w = Variable(torch.zeros(batch_size, obj_len, len(self.obj_vocab)+1).cuda())
        outputs = Variable(torch.zeros(batch_size, obj_len, self.embed_dim).cuda())
        outputs_h = Variable(torch.zeros(batch_size, obj_len, self.embed_dim).cuda())
        nce_logit = Variable(torch.zeros(batch_size, obj_len, att_feat_embed.shape[1]).cuda())
        att_logits = Variable(torch.zeros(batch_size, obj_len, att_feats.shape[1]).cuda())
        obj_att_mask = Variable(torch.zeros(batch_size, obj_len)).cuda()
        obj_att_mask[:, 0] = 1
        outputs[:, 0, :] = gv_feat
        unfinished = bos_index.eq(bos_index).squeeze(-1)
        for t in range(obj_len):
            # lstm1
            h2_tm1 = state[0][-1]
            input1 = torch.cat([h2_tm1, gv_feat, obj_feat[:, t, :], obj_word_embed[:,t,:]], 1)
            h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
            att, att_logit = self.att(h1_t, att_feats, p_att_feats, att_mask[:, 1:])
            att_logits[:,t,:] = att_logit

            # lstm2
            input2 = torch.cat([att, h1_t], 1)
            if self.dropout1 is not None:
                input2 = self.dropout1(input2)
            h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))

            state = (torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t]))
            outputs_h[:, t, :] = h2_t

            w_logit = self.word_logit(h2_t)
            outputs_w[:, t, :] = w_logit
        
            obj_feat_embed = self.obj_feat_embed(h2_t)
            if cfg.LOSSES.NCE.NORM == True:
                obj_feat_embed = F.normalize(obj_feat_embed, p=2, dim=-1)

            logit = torch.matmul(obj_feat_embed.unsqueeze(1), att_feat_embed.transpose(-1, -2)).squeeze(1)
            if cfg.LOSSES.NCE.NORM == True:
                logit = logit / cfg.LOSSES.NCE.TEMP
            nce_logit[:, t] = logit

            mlogit = logit.masked_fill(att_mask == 0, -1e9)
            if cfg.MODEL.BTONET.TRAIN_WITH_TOPK > 1:
                score = nn.Softmax(dim=-1)(mlogit)
                score_sort, obj_seq_pred = torch.sort(score, -1, descending=True)
                unfinished = unfinished * (obj_seq_pred[:,0] > 0)
                obj_seq_pred = obj_seq_pred[:,0:cfg.MODEL.BTONET.TRAIN_WITH_TOPK]
                score_sort = score_sort[:, 0:cfg.MODEL.BTONET.TRAIN_WITH_TOPK]
                score_sort = score_sort / score_sort.sum(-1, keepdim=True)
                if cfg.MODEL.BTONET.TOPK_SCORE_GRAD == False:
                    score_sort = score_sort.detach()
                obj_seq_pred = obj_seq_pred.unsqueeze(-1).expand(obj_seq_pred.shape[0], cfg.MODEL.BTONET.TRAIN_WITH_TOPK, self.embed_dim)
                obj_feat_pred =  torch.gather(encoder_out, 1, obj_seq_pred)
                obj_feat_pred = obj_feat_pred * score_sort.unsqueeze(-1)
                obj_feat_pred = obj_feat_pred.sum(1)
            else:
                _, obj_seq_pred = torch.max(mlogit, -1)
                _, obj_word_seq_pred = torch.max(w_logit, -1)
                #unfinished = unfinished * (obj_seq_pred > 0)
                unfinished = unfinished * (obj_word_seq_pred > 0)
                obj_seq_pred = obj_seq_pred.unsqueeze(-1).expand(obj_seq_pred.shape[0], self.embed_dim)
                obj_feat_pred =  torch.gather(encoder_out, 1, obj_seq_pred.unsqueeze(1)).squeeze(1)

            if t != obj_len - 1:
                outputs[:, t+1, :] = obj_feat_pred
                obj_att_mask[:, t+1] = unfinished.type(torch.cuda.FloatTensor)

        positions = self.embed_positions(obj_len)

        if cfg.MODEL.BTONET.TRAIN_WITH_GT:
            outputs = torch.cat([gv_feat.unsqueeze(1), obj_feat[:, 1:, :]], dim=1)
            outputs = outputs + positions
        else:
            outputs = outputs + positions
        return outputs, outputs_h, nce_logit, obj_att_mask, att_logits, outputs_w

    def decode(self, encoder_out, att_mask, **kwargs):
        #beam_size = kwargs['BEAM_SIZE']
        #if beam_size > 1:
        #    return self.decode_beam(encoder_out, att_mask, **kwargs)

        batch_size = encoder_out.size(0)
        state = self.init_hidden(batch_size)
        gv_feat = encoder_out[:,0,:]
        att_feats = encoder_out[:, 1:, :]
        p_att_feats = self.p_att_feats(att_feats)

        bos_index = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
        bos_embed = self.bos(bos_index).squeeze(1)
        obj_word_embed = self.word_embed(bos_index)

        max_obj_len = cfg.MODEL.BTONET.MAX_LEN
        outputs = Variable(torch.zeros(batch_size, max_obj_len, self.embed_dim).cuda())
        obj_att_mask = Variable(torch.zeros(batch_size, max_obj_len)).cuda()
        cur_obj_feat = bos_embed

        att_feat_embed = self.att_feat_embed(att_feats)
        eos_index = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
        eos_embed = self.eos(eos_index)
        att_feat_embed = torch.cat([eos_embed, att_feat_embed], dim=1)
        if cfg.LOSSES.NCE.NORM == True:
            att_feat_embed = F.normalize(att_feat_embed, p=2, dim=-1)

        unfinished = bos_index.eq(bos_index).squeeze(-1)
        obj_att_mask[:, 0] = unfinished.type(torch.cuda.FloatTensor)

        nce_logit = Variable(torch.zeros(batch_size, max_obj_len, att_feat_embed.shape[1]).cuda())

        # t = 0              t = 1              t=2
        # wt=[1,0,1]         wt=[0,0,1]         wt=[0,0,0]
        # h_0                h_1                h_2
        # mask[:,1]=[1,0,1]  mask[:,2]=[0,0,1]  mask[:,3]=[0,0,0]

        for t in range(max_obj_len):
            # lstm1
            h2_tm1 = state[0][-1]
            input1 = torch.cat([h2_tm1, gv_feat, cur_obj_feat, obj_word_embed.squeeze(1)], 1)
            h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
            att, _ = self.att(h1_t, att_feats, p_att_feats, att_mask[:, 1:])

            # lstm2
            input2 = torch.cat([att, h1_t], 1)
            if self.dropout1 is not None:
                input2 = self.dropout1(input2)
            h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))

            state = (torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t]))
            #if self.dropout2 is not None:
            #   h2_t = self.dropout2(h2_t)
            #outputs[:, t, :] = h2_t

            w_logit = self.word_logit(h2_t)

            obj_feat_embed = self.obj_feat_embed(h2_t)
            if cfg.LOSSES.NCE.NORM == True:
                obj_feat_embed = F.normalize(obj_feat_embed, p=2, dim=-1)
            scores = torch.matmul(obj_feat_embed.unsqueeze(1), att_feat_embed.transpose(-1, -2)).squeeze(1)
            scores = scores.masked_fill(att_mask == 0, -1e9)
            nce_logit[:, t, :] = scores

            if cfg.MODEL.BTONET.TRAIN_WITH_TOPK > 1:
                scores = nn.Softmax(dim=-1)(scores)
                score_sort, wt = torch.sort(scores, -1, descending=True)
                unfinished = unfinished * (wt[:,0] > 0)
                wt = wt[:,0:cfg.MODEL.BTONET.TRAIN_WITH_TOPK]
                score_sort = score_sort[:, 0:cfg.MODEL.BTONET.TRAIN_WITH_TOPK]
                score_sort = score_sort / score_sort.sum(-1, keepdim=True)
                wt = wt.unsqueeze(-1).expand(wt.shape[0], cfg.MODEL.BTONET.TRAIN_WITH_TOPK, self.embed_dim)
                cur_obj_feat = torch.gather(encoder_out, 1, wt)
                out_cur_obj_feat = cur_obj_feat * score_sort.unsqueeze(-1)
                out_cur_obj_feat = out_cur_obj_feat.sum(1)
                cur_obj_feat = cur_obj_feat[:,0]
            else:
                _, wt = torch.max(scores, 1)
                wt = wt.view(-1).long()
                #unfinished = unfinished * (wt > 0)
                _, obj_word_seq_pred = torch.max(w_logit, -1)
                obj_word_seq_pred = obj_word_seq_pred.view(-1).long()
                unfinished = unfinished * (obj_word_seq_pred > 0)
                
                wt = wt.view(-1, 1, 1).expand(wt.shape[0], 1, self.embed_dim)
                cur_obj_feat = torch.gather(encoder_out, 1, wt).squeeze(1)
                out_cur_obj_feat = cur_obj_feat

                obj_word_embed = self.word_embed(obj_word_seq_pred.unsqueeze(-1))

            if t != max_obj_len - 1:
                obj_att_mask[:, t+1] = unfinished.type(torch.cuda.FloatTensor)

            if unfinished.sum() == 0:
                break
            ###############################################
            if t != max_obj_len - 1:
                outputs[:, t+1, :] = out_cur_obj_feat
        outputs[:, 0, :] = gv_feat

        positions = self.embed_positions(max_obj_len)
        outputs = outputs + positions
        return outputs, obj_att_mask, nce_logit

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

        _encoder_out = encoder_out
        _gv_feat = gv_feat

        att_feat_embed = self.att_feat_embed(att_feats)
        eos_index = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
        eos_embed = self.eos(eos_index)
        att_feat_embed = torch.cat([eos_embed, att_feat_embed], dim=1)
  
        outputs = []
        obj_att_mask = []
        obj_att_mask.append(torch.ones(batch_size, 1, 1).cuda())

        for t in range(max_obj_len):
            cur_beam_size = 1 if t == 0 else beam_size

            # lstm1
            h2_tm1 = state[0][-1]
            input1 = torch.cat([h2_tm1, gv_feat, cur_obj_feat], 1)
            h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
            att, _ = self.att(h1_t, att_feats, p_att_feats, att_feats_mask[:, 1:])

            # lstm2
            input2 = torch.cat([att, h1_t], 1)
            if self.dropout1 is not None:
                input2 = self.dropout1(input2)
            h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))

            state = (torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t]))

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

        obj_att_mask = torch.cat(obj_att_mask, -1)
        obj_att_mask = torch.gather(obj_att_mask, 1, sort_idxs.expand(batch_size, beam_size, max_obj_len))

        out_size = 1
        outputs = outputs.contiguous()[:, :out_size].squeeze(1)
        obj_att_mask = obj_att_mask.contiguous()[:, :out_size].squeeze(1)

        outputs = outputs.unsqueeze(-1).expand(batch_size, max_obj_len, self.embed_dim)
        outputs_feat = torch.gather(_encoder_out, 1, outputs).squeeze(1)
        outputs_feat = torch.cat([_gv_feat.unsqueeze(1), outputs_feat[:,1:,:]], dim=1)

        positions = self.embed_positions(max_obj_len)
        outputs_feat = outputs_feat + positions

        return outputs_feat, obj_att_mask