import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg
import losses
import blocks
import layers
import lib.utils as utils
from models.basic_model import BasicModel
from layers.positional_encoding import PositionalEncoding
from models.transformer import Transformer, Encoder
from layers.objencoder import ObjEncoder
#from layers.objencoderptr import ObjEncoderPtr
#from layers.objencoderptr_onelayer import ObjEncoderPtrOneLayer
#from layers.objencoderptrw import ObjEncoderPtrw
from layers.objattention import ObjAttention

class LSTMDecoder(nn.Module):
    def __init__(self):
        super(LSTMDecoder, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        # word embed
        self.word_embed = nn.Sequential(
            nn.Embedding(self.vocab_size, cfg.MODEL.TRANSFORMER.OBJ_DIM),
            utils.activation('RELU'),
            nn.Dropout(0.5)
        )
        self.dropout_lm  = nn.Dropout(cfg.MODEL.DROPOUT_LM) if cfg.MODEL.DROPOUT_LM > 0 else None
        self.logit = nn.Linear(cfg.MODEL.TRANSFORMER.OBJ_DIM, self.vocab_size)
        self.p_att_feats = nn.Linear(cfg.MODEL.TRANSFORMER.OBJ_DIM, cfg.MODEL.TRANSFORMER.OBJ_DIM)

        self.num_layers = 2

        # First LSTM layer
        self.lstm1 = nn.LSTMCell(cfg.MODEL.TRANSFORMER.OBJ_DIM*3, cfg.MODEL.TRANSFORMER.OBJ_DIM)
        # Second LSTM Layer
        self.lstm2 = nn.LSTMCell(cfg.MODEL.TRANSFORMER.OBJ_DIM*2, cfg.MODEL.TRANSFORMER.OBJ_DIM)
        self.att = ObjAttention(cfg.MODEL.TRANSFORMER.OBJ_DIM, 0.1)

    def init_hidden(self, batch_size):
        return [Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.TRANSFORMER.OBJ_DIM).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.TRANSFORMER.OBJ_DIM).cuda())]

    def forward(self, o_encoder_out, att_mask, **kwargs):
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        batch_size = seq.shape[0]
        gv_feat = o_encoder_out[:,0,:]
        att_feats = o_encoder_out[:, 1:, :]
        p_att_feats = self.p_att_feats(att_feats)
        state = self.init_hidden(batch_size)
        outputs = Variable(torch.zeros(batch_size, seq.size(1), self.vocab_size).cuda())

        for t in range(seq.size(1)):
            wt = seq[:,t]
            xt = self.word_embed(wt)

            # lstm1
            h2_tm1 = state[0][-1]
            input1 = torch.cat([h2_tm1, gv_feat, xt], 1)
            h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
            att, _ = self.att(h1_t, att_feats, p_att_feats, att_mask[:, 1:])

            # lstm2
            input2 = torch.cat([att, h1_t], 1)
            h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))
            state = (torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t]))

            if self.dropout_lm is not None:
                output = self.dropout_lm(h2_t)
            logit = self.logit(output)
            outputs[:, t] = logit
        return outputs


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Objformer(nn.Module):
    def __init__(self):
        super(Objformer, self).__init__()
        self.ss_prob = 0.0
        
        sequential = []
        sequential.append(nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.TRANSFORMER.OBJ_DIM))
        if cfg.MODEL.ATT_FEATS_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.TRANSFORMER.OBJ_DIM, eps=1e-12))
        if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
        self.o_att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        self.o_encoder = Encoder(
           embed_dim = cfg.MODEL.TRANSFORMER.OBJ_DIM,
           dropout = cfg.MODEL.TRANSFORMER.ENCODE_DROPOUT,
           num_heads = cfg.MODEL.TRANSFORMER.HEAD, 
           attention_dropout = cfg.MODEL.TRANSFORMER.ATTENTION_DROPOUT,
           ff_dropout = cfg.MODEL.TRANSFORMER.ENCODE_FF_DROPOUT, 
           layer_num = cfg.MODEL.BTONET.ENCODE_LAYERS,
           use_aoa = cfg.MODEL.OBJ_USE_AOA,
           aoa_drop = cfg.MODEL.OBJ_AOA_DROP,
           layer_drop = cfg.MODEL.BTONET.OBJ_DROP_ENC_LAYER
        )

        self.obj_encoder = layers.create(
            cfg.MODEL.OBJ_ENC,
            embed_dim = cfg.MODEL.TRANSFORMER.OBJ_DIM,
            att_hidden_drop = cfg.MODEL.BTONET.ATT_HIDDEN_DROP, 
            dropout1 = cfg.MODEL.BTONET.DROPOUT1, 
            dropout2 = cfg.MODEL.BTONET.DROPOUT2
        )

        if cfg.LOSSES.OBJ_REGION.LOSS_WEIGHT > 0:
            obj_vocab = utils.load_lines(cfg.DATA_LOADER.OBJ_VOCAB_PATH)
            region_cls_net = []
            last_dim = cfg.MODEL.TRANSFORMER.OBJ_DIM
            for mlp_dim in cfg.LOSSES.OBJ_REGION.MLP:
                region_cls_net.append(nn.Linear(last_dim, mlp_dim))
                region_cls_net.append(nn.ReLU(inplace=True))
                last_dim = mlp_dim
            region_cls_net.append(nn.Linear(last_dim, len(obj_vocab) + 1))
            self.o_region_cls_net = nn.Sequential(*region_cls_net)

        if cfg.LOSSES.OBJ_REGION_DEC.LOSS_WEIGHT > 0:
            obj_vocab = utils.load_lines(cfg.DATA_LOADER.OBJ_VOCAB_PATH)
            o_region_dec_cls_net = []
            last_dim = cfg.MODEL.TRANSFORMER.OBJ_DIM
            for mlp_dim in cfg.LOSSES.OBJ_REGION_DEC.MLP:
                o_region_dec_cls_net.append(nn.Linear(last_dim, mlp_dim))
                o_region_dec_cls_net.append(nn.ReLU(inplace=True))
                last_dim = mlp_dim
            o_region_dec_cls_net.append(nn.Linear(last_dim, len(obj_vocab) + 1))
            self.o_region_dec_cls_net = nn.Sequential(*o_region_dec_cls_net)

        if cfg.LOSSES.GV_SUP.LOSS_WEIGHT > 0:
            self.o_mil_net = nn.Linear(cfg.MODEL.TRANSFORMER.OBJ_DIM, 1000)

        if cfg.MODEL.BTONET.OBJ_USE_LANG_GUIDE:
            self.updown = LSTMDecoder()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_after(self, train_obj, train_cap, o_encoder_out, obj_encoder_out, obj_att_logits, obj_encoder_wlogits, att_mask, **kwargs):
        losses = {}
        if train_obj == False:
            return losses

        if cfg.LOSSES.OBJ_REGION.LOSS_WEIGHT > 0:
            o_region_cls = self.o_region_cls_net(o_encoder_out[:,1:,:])
            losses[cfg.LOSSES.OBJ_REGION.LOSS_NAME] = o_region_cls

        if cfg.LOSSES.OBJ_REGION_DEC.LOSS_WEIGHT > 0:
            if obj_encoder_wlogits == None:
                o_region_dec_cls = self.o_region_dec_cls_net(obj_encoder_out)
                losses[cfg.LOSSES.OBJ_REGION_DEC.LOSS_NAME] = o_region_dec_cls
            else:
                losses[cfg.LOSSES.OBJ_REGION_DEC.LOSS_NAME] = obj_encoder_wlogits

        if cfg.LOSSES.OBJ_DEC_ATT.LOSS_WEIGHT > 0:
            losses[cfg.LOSSES.OBJ_DEC_ATT.LOSS_NAME] = obj_att_logits

        if cfg.LOSSES.GV_SUP.LOSS_WEIGHT > 0:
            o_gvfeat = o_encoder_out[:,0,:]
            o_mil = self.o_mil_net(o_gvfeat)
            losses['o_' + cfg.LOSSES.GV_SUP.LOSS_NAME] = o_mil

        if cfg.MODEL.BTONET.OBJ_USE_LANG_GUIDE and (train_obj == True) and (train_cap == False) and self.training:
            out_logit = self.updown(o_encoder_out, att_mask.squeeze(1), **kwargs)
            losses[cfg.PARAM.OBJ_XE] = out_logit

        return losses

    def forward_rl(self, att_feats, att_mask, obj_seq, obj_att_mask, obj_seq_label, **kwargs):
        o_att_feats = self.o_att_embed(att_feats)
        o_encoder_out = self.o_encoder(o_att_feats, att_mask)
        final_att_mask = None
        obj_encoder_out, obj_att_mask_pred, nce_logit, obj_seq_pred = self.obj_encoder.decode(o_encoder_out, att_mask.squeeze(1), final_att_mask, **kwargs)
        return obj_encoder_out, obj_att_mask_pred, nce_logit, obj_seq_pred

    def forward(self, train_obj, train_cap, att_feats, att_mask, obj_seq, obj_att_mask, obj_seq_label, rl_stage=False, **kwargs):
        if rl_stage == True:
            return self.forward_rl(att_feats, att_mask, obj_seq, obj_att_mask, obj_seq_label, **kwargs)

        if train_obj:
            o_att_feats = self.o_att_embed(att_feats)
            o_encoder_out = self.o_encoder(o_att_feats, att_mask)

            ##########################################################################################
            if cfg.LOSSES.OBJ_REGION.LOSS_WEIGHT > 0 and cfg.MODEL.BTONET.TRAIN_FILTER_REG_TOPK > 0:
                with torch.no_grad():
                    o_region_cls = self.o_region_cls_net(o_encoder_out[:,1:,:].detach())
                    o_region_cls = F.softmax(o_region_cls, dim=-1)
                    s_att_mask = att_mask.squeeze(1)[:,1:]
                    region_cls_scores, _ = torch.max(o_region_cls[:,:,1:], -1)
                    #region_cls_scores = o_region_cls[:,:,1:].sum(-1)
                    region_cls_scores = region_cls_scores.masked_fill(s_att_mask == 0, -1e9)
                    _, region_cls_sel = torch.sort(region_cls_scores, -1, descending=True)
                    region_cls_sel = region_cls_sel[:, 0:cfg.MODEL.BTONET.TRAIN_FILTER_REG_TOPK]
                    final_att_mask = torch.zeros_like(s_att_mask)
                    copy_val = torch.zeros(region_cls_sel.shape[0], cfg.MODEL.BTONET.TRAIN_FILTER_REG_TOPK).cuda() + 1
                    final_att_mask.scatter_(1, region_cls_sel, copy_val)
                    final_att_mask = torch.cat([
                        torch.ones(region_cls_sel.shape[0], 1).cuda(),
                        final_att_mask
                    ], dim=1)
            else:
                final_att_mask = None
            ##########################################################################################

            obj_encoder_out, obj_encoder_out_h, nce_logits, obj_att_mask_pred, obj_att_logits, obj_encoder_wlogits = \
                self.obj_encoder(
                    o_encoder_out, 
                    obj_seq, 
                    att_mask.squeeze(1), 
                    obj_seq_label,
                    final_att_mask=final_att_mask,
                    ss_prob=self.ss_prob
                )
            if cfg.MODEL.BTONET.TRAIN_WITH_GT:
                obj_att_mask_pred = obj_att_mask
            o_region_cls = None
            obj_seq_pred = None

        else:
            with torch.no_grad():
                o_att_feats = self.o_att_embed(att_feats)
                o_encoder_out = self.o_encoder(o_att_feats, att_mask)

                if cfg.LOSSES.OBJ_REGION.LOSS_WEIGHT > 0:
                    o_region_cls = self.o_region_cls_net(o_encoder_out[:,1:,:])
                else:
                    o_region_cls = None

                if cfg.MODEL.BTONET.TEST_FILTER_REG_TOPK > 0 and o_region_cls is not None:
                    s_att_mask = att_mask.squeeze(1)[:,1:]
                    o_region_cls_softmax = F.softmax(o_region_cls, dim=-1)[:,:,1:]
                    o_region_cls_softmax_mask = torch.ones(o_region_cls_softmax.shape[0], o_region_cls_softmax.shape[1]).cuda()
                    o_region_cls_softmax_mask2 = torch.ones(o_region_cls_softmax.shape[0], o_region_cls_softmax.shape[-1]).cuda()
                    copy_val = torch.zeros(o_region_cls_softmax.shape[0], 1).cuda()
                    copy_val2 = torch.zeros(o_region_cls_softmax.shape[0], 1).cuda()

                    region_cls_sel_arr = []
                    for k in range(cfg.MODEL.BTONET.TEST_FILTER_REG_TOPK):
                        region_cls_scores, region_cls = torch.max(o_region_cls_softmax, -1)
                        #region_cls_scores = o_region_cls_softmax.sum(-1)
                        region_cls_scores = region_cls_scores.masked_fill(s_att_mask == 0, -1e9)
                        _, region_cls_sel = torch.max(region_cls_scores, -1)
                        region_cls_sel = region_cls_sel.unsqueeze(-1)
                        o_region_cls_softmax_mask.scatter_(1, region_cls_sel, copy_val)
                        o_region_cls_softmax = o_region_cls_softmax * o_region_cls_softmax_mask.unsqueeze(-1)
                        region_cls_sel_arr.append(region_cls_sel)
                        region_cls = torch.gather(region_cls, 1, region_cls_sel)
                        o_region_cls_softmax_mask2.scatter_(1, region_cls, copy_val2)
                        o_region_cls_softmax = o_region_cls_softmax * o_region_cls_softmax_mask2.unsqueeze(1)
                    region_cls_sel = torch.cat(region_cls_sel_arr, dim=-1)

                    #s_att_mask = att_mask.squeeze(1)[:,1:]
                    #o_region_cls_softmax = F.softmax(o_region_cls, dim=-1)
                    #region_cls_scores, _ = torch.max(o_region_cls_softmax[:,:,1:], -1)
                    ##region_cls_scores = o_region_cls_softmax[:,:,1:].sum(-1)
                    #region_cls_scores = region_cls_scores.masked_fill(s_att_mask == 0, -1e9)
                    #_, region_cls_sel = torch.sort(region_cls_scores, -1, descending=True)
                    #region_cls_sel = region_cls_sel[:, 0:cfg.MODEL.BTONET.TEST_FILTER_REG_TOPK]

                    final_att_mask = torch.zeros_like(s_att_mask)
                    copy_val = torch.zeros(region_cls_sel.shape[0], cfg.MODEL.BTONET.TEST_FILTER_REG_TOPK).cuda() + 1
                    final_att_mask.scatter_(1, region_cls_sel, copy_val)
                    final_att_mask = torch.cat([
                        torch.ones(region_cls_sel.shape[0], 1).cuda(),
                        final_att_mask
                    ], dim=1)
                else:
                    final_att_mask = None

                kwargs['GREEDY_DECODE'] = True
                obj_encoder_out, obj_att_mask_pred, nce_logits, obj_seq_pred = self.obj_encoder.decode(o_encoder_out, att_mask.squeeze(1), final_att_mask, **kwargs)
                #nce_logits = None
                obj_att_logits = None
                obj_encoder_out_h = None
                obj_encoder_wlogits = None

        losses = self.forward_after(train_obj, train_cap, o_encoder_out, obj_encoder_out, obj_att_logits, obj_encoder_wlogits, att_mask, **kwargs)

        if train_obj:
            losses[cfg.LOSSES.NCE.LOSS_NAME] = nce_logits

        return losses, obj_encoder_out, obj_att_mask_pred, (nce_logits, o_region_cls, obj_seq_pred)

class Btoformer(Transformer):
    def __init__(self):
        super(Btoformer, self).__init__()
        self.ss_prob = 0.0

        self.decoder = Decoder(
            vocab_size = self.vocab_size,
            embed_dim = cfg.MODEL.TRANSFORMER.DIM,
            obj_embed_dim = cfg.MODEL.TRANSFORMER.OBJ_DIM,
            dropout = cfg.MODEL.TRANSFORMER.DECODE_DROPOUT,
            num_heads = cfg.MODEL.TRANSFORMER.HEAD, 
            attention_dropout = cfg.MODEL.TRANSFORMER.ATTENTION_DROPOUT,
            ff_dropout = cfg.MODEL.TRANSFORMER.DECODE_FF_DROPOUT,
            layer_num = cfg.MODEL.TRANSFORMER.DECODE_LAYERS,
            use_aoa = cfg.MODEL.CAP_USE_AOA,
            aoa_drop = cfg.MODEL.CAP_AOA_DROP,
            layer_drop = cfg.MODEL.BTONET.CAP_DROP_DEC_LAYER
        )
        self.apply(self.init_weights)
        self.nce_criterion = losses.create('NCE').cuda()
        self.region_criterion = losses.create('MultiLabel').cuda()

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_after(self, train_cap, encoder_out, regatt_logit):
        losses = {}
        if train_cap == False:
            return losses

        if cfg.LOSSES.REGION.LOSS_WEIGHT > 0: 
            region_cls = self.region_cls_net(encoder_out[:,1:,:])
            losses[cfg.LOSSES.REGION.LOSS_NAME] = region_cls

        if cfg.LOSSES.REGION_ATT.LOSS_WEIGHT > 0:
            losses[cfg.LOSSES.REGION_ATT.LOSS_NAME] = regatt_logit

        if cfg.LOSSES.GV_SUP.LOSS_WEIGHT > 0:
            gvfeat = encoder_out[:,0,:]
            mil = self.mil_net(gvfeat)
            losses[cfg.LOSSES.GV_SUP.LOSS_NAME] = mil

        return losses

    def forward(self, train_obj, train_cap, obj_model, is_decode=False, **kwargs):
        if is_decode == True:
            seq_sample, logP_sample, nce_logit_sample, obj_seq_pred_sample = self.decode_rl(obj_model, **kwargs)
            return seq_sample, logP_sample, nce_logit_sample, obj_seq_pred_sample

        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        obj_seq = kwargs[cfg.PARAM.OBJ_SEQ]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_mask = att_mask.unsqueeze(1)
        obj_att_mask = kwargs[cfg.PARAM.OBJ_FEATS_MASK]
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        obj_seq = obj_seq.view(-1, obj_seq.shape[-1])
        #obj_att_mask = obj_att_mask.view(-1, obj_att_mask.shape[-1])

        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:,0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        ##############################################
        obj_reg_label = utils.expand_tensor(kwargs[cfg.PARAM.REGION_LABEL], cfg.DATA_LOADER.SEQ_PER_IMG)
        obj_seq_label = torch.gather(obj_reg_label, 1, obj_seq.unsqueeze(-1).expand(obj_seq.shape[0], obj_seq.shape[1], obj_reg_label.shape[-1]))
        obj_reg_label = torch.rand(obj_reg_label.shape).cuda() * 0.1 + obj_reg_label
        obj_reg_label[:,:,0] += 0.2
        _, obj_seq_label = torch.sort(obj_seq_label, -1, descending=True)
        obj_seq_label = obj_seq_label[:,:,0]
        ##############################################

        objlosses, obj_encoder_out, obj_att_mask_pred, nce_logits = \
            obj_model(train_obj, train_cap, att_feats, att_mask, obj_seq, obj_att_mask, obj_seq_label, **kwargs)

        if train_cap:
            att_feats = self.att_embed(att_feats)
            encoder_out = self.encoder(att_feats, att_mask)

            if self.training and self.ss_prob > 0:
                with torch.no_grad():
                    seq_copy = copy.copy(seq)
                    encoder_out_copy = copy.copy(encoder_out)
                    obj_encoder_out_copy = copy.copy(obj_encoder_out)
                    decoder_out, _, _ = self.decoder(seq_copy, encoder_out_copy, obj_encoder_out_copy, att_mask, obj_att_mask_pred, seq_mask)
                    B, T = decoder_out.shape[:2]
                    prob = torch.empty(B, T).cuda().uniform_(0, 1)
                    mask = prob < self.ss_prob
                    for t in range(0, T-1):
                        mask_t = mask[:, t]
                        if mask_t.sum() > 0:
                            ind = mask_t.nonzero().view(-1)
                            wt = seq[:, t+1].data.clone()
                            prob_prev = F.softmax(decoder_out[:, t].detach(), dim=-1) #prob_prev = torch.exp(decoder_out[:, t].detach())
                            wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
                            seq[:, t+1] = wt
                decoder_out, regatt_logit, obj_regatt_logit = self.decoder(seq, encoder_out, obj_encoder_out, att_mask, obj_att_mask_pred, seq_mask)
            else:
                decoder_out, regatt_logit, obj_regatt_logit = self.decoder(seq, encoder_out, obj_encoder_out, att_mask, obj_att_mask_pred, seq_mask)
        else:
            encoder_out = None
            decoder_out = None
            regatt_logit = None
            obj_regatt_logit = None

        losses = self.forward_after(train_cap, encoder_out, regatt_logit)
        if train_cap:
            losses[cfg.LOSSES.XE_TYPE] = decoder_out

        for key in objlosses:
            losses[key] = objlosses[key]
        
        return losses

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        history_states = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        obj_encoder_out = kwargs[cfg.PARAM.OBJ_FEATS]
        obj_att_mask = kwargs[cfg.PARAM.OBJ_FEATS_MASK]

        if history_states is None:
            seq_len = 1
        else:
            seq_len = history_states[0].size(1) + 1
        seq_mask = subsequent_mask(seq_len).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(1)
        decoder_out, history_states = self.decoder.decode(wt, encoder_out, obj_encoder_out, att_mask, obj_att_mask, seq_mask, history_states)
        decoder_out = decoder_out[:,-1,:]
        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, history_states

    def decode_rl(self, obj_model, **kwargs):
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = att_mask.unsqueeze(1)

        batch_size = att_feats.size(0)
        max_seq_length = cfg.MODEL.SEQ_LEN

        sents = Variable(torch.zeros((batch_size, max_seq_length), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, max_seq_length).cuda())
        bos_input = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = bos_input.eq(bos_input)

        obj_encoder_out, obj_att_mask, nce_logit, obj_seq_pred = obj_model(True, True, att_feats, att_mask, None, None, None, rl_stage=True, **kwargs)
        
        att_feats = self.att_embed(att_feats)
        encoder_out = self.encoder(att_feats, att_mask)

        history_states = None
        ys = bos_input.unsqueeze(1)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.OBJ_FEATS] = obj_encoder_out
        kwargs[cfg.PARAM.OBJ_FEATS_MASK] = obj_att_mask
        for t in range(0, max_seq_length):
            kwargs[cfg.PARAM.WT] = ys
            kwargs[cfg.PARAM.STATE] = history_states
            logprobs_t, history_states = self.get_logprobs_state(**kwargs)

            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)

            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break

            ys = wt.unsqueeze(1)
        
        return sents, logprobs, nce_logit, obj_seq_pred

    def decode(self, obj_model, **kwargs):
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = att_mask.unsqueeze(1)

        batch_size = att_feats.size(0)
        max_seq_length = cfg.MODEL.SEQ_LEN

        sents = Variable(torch.zeros((batch_size, max_seq_length), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, max_seq_length).cuda())
        bos_input = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = bos_input.eq(bos_input)

        _, obj_encoder_out, obj_att_mask, logits_arr = obj_model(False, False, att_feats, att_mask, None, None, None, **kwargs)
        nce_logits = logits_arr[0]
        o_region_cls = logits_arr[1]
        obj_pos_label = kwargs[cfg.PARAM.OBJ_POS_LABEL]
        obj_gt_mask = kwargs[cfg.PARAM.OBJ_FEATS_MASK]
        if nce_logits.shape[1] > obj_pos_label.shape[1]:
            nce_logits = nce_logits[:,0:obj_pos_label.shape[1], :]
        if nce_logits.shape[1] < obj_pos_label.shape[1]:
            obj_pos_label = obj_pos_label[:,0:nce_logits.shape[1], :]
            obj_gt_mask = obj_gt_mask[:,0:nce_logits.shape[1]]

        loss_info = {}
        _, nce_loss_info = self.nce_criterion(nce_logits[:, 0:obj_pos_label.shape[1], :], obj_pos_label, None, att_mask.squeeze(1), obj_gt_mask)
        obj_acc_cnt = nce_loss_info['accuracy'] * nce_logits.shape[0]
        loss_info['obj_acc_cnt'] = obj_acc_cnt
        obj_set_acc_cnt = nce_loss_info['set_accuracy'] * nce_logits.shape[0]
        loss_info['obj_set_acc_cnt'] = obj_set_acc_cnt

        if cfg.LOSSES.OBJ_REGION.LOSS_WEIGHT > 0:
            _, obj_reg_loss_info = self.region_criterion(o_region_cls, kwargs[cfg.PARAM.REGION_LABEL])
            obj_reg_acc_cnt = obj_reg_loss_info['Multi-label Loss-Acc'] * o_region_cls.shape[0]
            loss_info['obj_reg_acc_cnt'] = obj_reg_acc_cnt
        
        att_feats = self.att_embed(att_feats)
        encoder_out = self.encoder(att_feats, att_mask)

        history_states = None
        ys = bos_input.unsqueeze(1)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.OBJ_FEATS] = obj_encoder_out
        kwargs[cfg.PARAM.OBJ_FEATS_MASK] = obj_att_mask
        for t in range(0, max_seq_length):
            kwargs[cfg.PARAM.WT] = ys
            kwargs[cfg.PARAM.STATE] = history_states
            logprobs_t, history_states = self.get_logprobs_state(**kwargs)

            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)

            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break

            ys = wt.unsqueeze(1)
        
        return sents, logprobs, obj_acc_cnt, (logits_arr[0], logits_arr[2])

    def decode_beam(self, obj_model, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = att_mask.unsqueeze(1)

        batch_size = att_feats.size(0)
        beam_size = kwargs['BEAM_SIZE']
        max_seq_length = cfg.MODEL.SEQ_LEN

        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()
        bos_input = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())

        #o_att_feats = self.o_att_embed(att_feats)
        #o_encoder_out = self.o_encoder(o_att_feats, att_mask)
        #obj_encoder_out, obj_att_mask = self.obj_encoder.decode(o_encoder_out, att_mask.squeeze(1), **kwargs)
        #obj_encoder_out, obj_att_mask = self.obj_encoder.decode_beam(encoder_out, att_mask.squeeze(1), **kwargs)

        _, obj_encoder_out, obj_att_mask, logits_arr = obj_model(False, False, att_feats, att_mask, None, None, None, **kwargs)
        nce_logits = logits_arr[0]
        o_region_cls = logits_arr[1]
        obj_pos_label = kwargs[cfg.PARAM.OBJ_POS_LABEL]
        obj_gt_mask = kwargs[cfg.PARAM.OBJ_FEATS_MASK]
        if nce_logits.shape[1] > obj_pos_label.shape[1]:
            nce_logits = nce_logits[:,0:obj_pos_label.shape[1], :]
        if nce_logits.shape[1] < obj_pos_label.shape[1]:
            obj_pos_label = obj_pos_label[:,0:nce_logits.shape[1], :]
            obj_gt_mask = obj_gt_mask[:,0:nce_logits.shape[1]]
        
        loss_info = {}
        _, nce_loss_info = self.nce_criterion(nce_logits[:, 0:obj_pos_label.shape[1], :], obj_pos_label, None, att_mask.squeeze(1), obj_gt_mask)
        obj_acc_cnt = nce_loss_info['accuracy'] * nce_logits.shape[0]
        loss_info['obj_acc_cnt'] = obj_acc_cnt
        obj_set_acc_cnt = nce_loss_info['set_accuracy'] * nce_logits.shape[0]
        loss_info['obj_set_acc_cnt'] = obj_set_acc_cnt

        if cfg.LOSSES.OBJ_REGION.LOSS_WEIGHT > 0:
            _, obj_reg_loss_info = self.region_criterion(o_region_cls, kwargs[cfg.PARAM.REGION_LABEL])
            obj_reg_acc_cnt = obj_reg_loss_info['Multi-label Loss-Acc'] * o_region_cls.shape[0]
            loss_info['obj_reg_acc_cnt'] = obj_reg_acc_cnt


        att_feats = self.att_embed(att_feats)
        encoder_out = self.encoder(att_feats, att_mask)

        history_states = None
        ys = bos_input.unsqueeze(1)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.OBJ_FEATS] = obj_encoder_out
        kwargs[cfg.PARAM.OBJ_FEATS_MASK] = obj_att_mask

        outputs = []
        for t in range(0, max_seq_length):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = ys
            kwargs[cfg.PARAM.STATE] = history_states
            word_logprob, history_states = self.get_logprobs_state(**kwargs)
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
            for i in range(len(history_states)):
                shape = [int(sh) for sh in history_states[i].shape]
                beam = selected_beam
                for _ in shape[1:]:
                    beam = beam.unsqueeze(-1)
                history_states[i] = torch.gather(history_states[i].view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                    beam.expand(*([batch_size, beam_size] + shape[1:])))
                history_states[i] = history_states[i].view(*([-1, ] + shape[1:]))
            #################################

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            ys = selected_words

            if t == 0:
                att_mask = utils.expand_tensor_beam(att_mask, beam_size)
                obj_att_mask = utils.expand_tensor_beam(obj_att_mask, beam_size)
                encoder_out = utils.expand_tensor_beam(encoder_out, beam_size)
                obj_encoder_out = utils.expand_tensor_beam(obj_encoder_out, beam_size)
                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                kwargs[cfg.PARAM.OBJ_FEATS] = obj_encoder_out
                kwargs[cfg.PARAM.OBJ_FEATS_MASK] = obj_att_mask

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, max_seq_length))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, max_seq_length))

        out_size = 1
        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]

        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        return outputs, log_probs, loss_info


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size, 
        embed_dim,
        obj_embed_dim,
        dropout,
        num_heads, 
        attention_dropout,
        ff_dropout,
        layer_num,
        use_aoa=False,
        aoa_drop=0.,
        layer_drop = 0.
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim
        for _ in range(layer_num):
            layer = DecoderLayer(
                embed_dim = embed_dim,
                obj_embed_dim = obj_embed_dim,
                dropout = dropout,
                num_heads = num_heads, 
                attention_dropout = attention_dropout,
                ff_dropout = ff_dropout,
                use_aoa=use_aoa,
                aoa_drop=aoa_drop
            )
            self.layers.append(layer)

        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED)
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        #self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEncoding(
            embed_dim, cfg.MODEL.TRANSFORMER.PE_MAX_LEN
        )
        self.layer_norm = torch.nn.LayerNorm(embed_dim, eps=1e-12)

        self.generator = nn.Linear(embed_dim, vocab_size)
        self.dropout_lm  = nn.Dropout(cfg.MODEL.DROPOUT_LM)
        self.layer_drop = layer_drop

        if cfg.MODEL.BTONET.USE_HEAD_TRANSFORM:
            if cfg.MODEL.BTONET.USE_MULTILAYER:
                self.transform = nn.Sequential(
                    nn.Linear(embed_dim * layer_num, 2 * embed_dim),
                    nn.GLU(),
                    torch.nn.LayerNorm(embed_dim)
                )
            else:
                self.transform = nn.Sequential(
                    nn.Linear(embed_dim, 2 * embed_dim),
                    nn.GLU(),
                    torch.nn.LayerNorm(embed_dim)
                )

    def forward(self, tokens, encoder_out, obj_encoder_out, att_mask, obj_att_mask, seq_mask):
        # embed positions
        seq_len = tokens.size(1)
        positions = self.embed_positions(seq_len)

        # embed tokens and positions
        #x = self.embed_scale * self.embed_tokens(tokens)
        x = self.embed_tokens(tokens)

        x = x + positions
        x = self.layer_norm(x)
        if self.dropout is not None:
            x = self.dropout(x)

        obj_att_mask = obj_att_mask.unsqueeze(1)

        regatt_logit_arr = []
        obj_regatt_logit_arr = []
        x_arr = []

        if self.layer_drop > 0:
            for layer_id, layer in enumerate(self.layers):
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layer_drop) and (layer_id > 0):
                    x = x
                else:
                    x, regatt_logit, obj_regatt_logit = layer(x, encoder_out, obj_encoder_out, att_mask, obj_att_mask, seq_mask)
                
                if cfg.MODEL.BTONET.USE_MULTILAYER:
                    x_arr.append(x)
        else:
            for layer in self.layers:
                x, regatt_logit, obj_regatt_logit = layer(x, encoder_out, obj_encoder_out, att_mask, obj_att_mask, seq_mask)
                
                if cfg.MODEL.BTONET.USE_MULTILAYER:
                    x_arr.append(x)

        regatt_logit_arr.append(regatt_logit.unsqueeze(1))
        obj_regatt_logit_arr.append(obj_regatt_logit.unsqueeze(1))

        regatt_logit = torch.cat(regatt_logit_arr, dim=1)
        obj_regatt_logit = torch.cat(obj_regatt_logit_arr, dim=1)

        if cfg.MODEL.BTONET.USE_HEAD_TRANSFORM:
            if cfg.MODEL.BTONET.USE_MULTILAYER:
                x_arr = torch.cat(x_arr, dim=-1)
                x = self.transform(x_arr)
            else:
                x = self.transform(x)

        x = self.dropout_lm(x)
        out = self.generator(x)
        return out, regatt_logit, obj_regatt_logit

    def decode(self, wt, encoder_out, obj_encoder_out, att_mask, obj_att_mask, seq_mask, history_states=None):
        if history_states is None:
            seq_len = 1
            history_states = [None] * len(self.layers)
        else:
            seq_len = history_states[0].size(1) + 1
        
        # embed positions
        #seq_len = tokens.size(1)
        positions = self.embed_positions(seq_len)
        positions = positions[:,-1,:].unsqueeze(1)

        # embed tokens and positions
        #x = self.embed_scale * self.embed_tokens(tokens)
        x = self.embed_tokens(wt)

        x = x + positions
        x = self.layer_norm(x)
        if self.dropout is not None:
            x = self.dropout(x)

        x_arr = []
        obj_att_mask = obj_att_mask.unsqueeze(1)
        for i, layer in enumerate(self.layers):
            if history_states[i] is None:
                history_states[i] = x
            else:
                history_states[i] = torch.cat([history_states[i], x], dim=1)

            x, _, _ = layer(x, encoder_out, obj_encoder_out, att_mask, obj_att_mask, seq_mask, history_states[i])
            x_arr.append(x)

        if cfg.MODEL.BTONET.USE_HEAD_TRANSFORM:
            if cfg.MODEL.BTONET.USE_MULTILAYER:
                x_arr = torch.cat(x_arr, dim=-1)
                x = self.transform(x_arr)
            else:
                x = self.transform(x)

        x = self.dropout_lm(x)
        out = self.generator(x)
        return out, history_states

class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        obj_embed_dim,
        dropout,
        num_heads, 
        attention_dropout,
        ff_dropout,
        use_aoa,
        aoa_drop
    ):
        super(DecoderLayer, self).__init__()
        self.self_attn = blocks.create(
            'MultiheadAttention',
            hidden_size = embed_dim, 
            num_attention_heads = num_heads, 
            attention_dropout=attention_dropout, 
            dropout=dropout,
            use_aoa=use_aoa,
            aoa_drop=aoa_drop)

        self.x_attn = blocks.create(
            'MultimodalMH',
            hidden_size = embed_dim, 
            obj_hidden_size = obj_embed_dim, 
            num_attention_heads = num_heads, 
            attention_dropout=attention_dropout, 
            dropout=dropout,
            use_aoa=use_aoa,
            aoa_drop=aoa_drop)

        #self.x_attn = blocks.create(
        #    'MultimodalSE',
        #    hidden_size = embed_dim, 
        #    obj_hidden_size = obj_embed_dim, 
        #    num_attention_heads = num_heads, 
        #    attention_dropout=attention_dropout, 
        #    dropout=dropout,
        #    use_aoa=use_aoa,
        #    aoa_drop=aoa_drop)

        #self.x_attn = blocks.create(
        #    'MultimodalSE2',
        #    hidden_size = embed_dim, 
        #    obj_hidden_size = obj_embed_dim, 
        #    num_attention_heads = num_heads, 
        #    attention_dropout=attention_dropout, 
        #    dropout=dropout,
        #    use_aoa=use_aoa,
        #    aoa_drop=aoa_drop)

        self.ff_layer = blocks.create(
            'FeedForward',
            embed_dim = embed_dim, 
            ffn_embed_dim = embed_dim * 4, 
            relu_dropout = ff_dropout, 
            dropout = ff_dropout)

    def forward(self, x, encoder_out, obj_encoder_out, att_mask, obj_att_mask, seq_mask, history_states=None):
        x, _ = self.self_attn(x, key=None, value=None, attention_mask=seq_mask, history_states=history_states)
        x, regatt_logit, obj_regatt_logit = self.x_attn(
            x, 
            key=encoder_out, 
            value=encoder_out, 
            obj_key=obj_encoder_out, 
            obj_value=obj_encoder_out,
            attention_mask=att_mask,
            obj_attention_mask=obj_att_mask
        )
        x = self.ff_layer(x)
        return x, regatt_logit, obj_regatt_logit
