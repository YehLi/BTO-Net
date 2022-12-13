import copy
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

import blocks
import lib.utils as utils
from models.basic_model import BasicModel
from layers.positional_encoding import PositionalEncoding

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Transformer(BasicModel):
    def __init__(self):
        super(Transformer, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1

        # att_feats encoder
        sequential = []
        sequential.append(nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.TRANSFORMER.DIM))
        sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
        if cfg.MODEL.ATT_FEATS_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.TRANSFORMER.DIM, eps=1e-12))
        if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))    
        self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        self.encoder = Encoder(
            embed_dim = cfg.MODEL.TRANSFORMER.DIM,
            dropout = cfg.MODEL.TRANSFORMER.ENCODE_DROPOUT,
            num_heads = cfg.MODEL.TRANSFORMER.HEAD, 
            attention_dropout = cfg.MODEL.TRANSFORMER.ATTENTION_DROPOUT,
            ff_dropout = cfg.MODEL.TRANSFORMER.ENCODE_FF_DROPOUT,
            layer_num = cfg.MODEL.TRANSFORMER.ENCODE_LAYERS,
            layer_drop = cfg.MODEL.BTONET.CAP_DROP_ENC_LAYER
        )

        self.decoder = Decoder(
            vocab_size = self.vocab_size,
            embed_dim = cfg.MODEL.TRANSFORMER.DIM,
            dropout = cfg.MODEL.TRANSFORMER.DECODE_DROPOUT,
            num_heads = cfg.MODEL.TRANSFORMER.HEAD, 
            attention_dropout = cfg.MODEL.TRANSFORMER.ATTENTION_DROPOUT,
            ff_dropout = cfg.MODEL.TRANSFORMER.DECODE_FF_DROPOUT,
            layer_num = cfg.MODEL.TRANSFORMER.DECODE_LAYERS
        )

        if cfg.LOSSES.REGION.LOSS_WEIGHT > 0:
            obj_vocab = utils.load_lines(cfg.DATA_LOADER.OBJ_VOCAB_PATH)
            region_cls_net = []
            last_dim = cfg.MODEL.TRANSFORMER.DIM
            for mlp_dim in cfg.LOSSES.REGION.MLP:
                region_cls_net.append(nn.Linear(last_dim, mlp_dim))
                region_cls_net.append(nn.ReLU(inplace=True))
                last_dim = mlp_dim
            region_cls_net.append(nn.Linear(last_dim, len(obj_vocab) + 1))
            self.region_cls_net = nn.Sequential(*region_cls_net)

        if cfg.LOSSES.GV_SUP.LOSS_WEIGHT > 0:
            self.mil_net = nn.Linear(cfg.MODEL.TRANSFORMER.DIM, 1000)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_grad_obj(self, flag):
        pass

    def set_grad_cap(self, flag):
        pass

    def forward_after(self, encoder_out, decoder_out, regatt_logit):
        losses = {}
        if cfg.LOSSES.REGION.LOSS_WEIGHT > 0: 
            region_cls = self.region_cls_net(encoder_out[:,1:,:])
            losses[cfg.LOSSES.REGION.LOSS_NAME] = region_cls
        if cfg.LOSSES.REGION_ATT.LOSS_WEIGHT > 0:
            losses[cfg.LOSSES.REGION_ATT.LOSS_NAME] = regatt_logit
        return losses

    def forward(self, is_decode = False, **kwargs):
        if is_decode == True:
            return self.decode(**kwargs)

        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_mask = att_mask.unsqueeze(1)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:,0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        att_feats = self.att_embed(att_feats)
        encoder_out = self.encoder(att_feats, att_mask)
        decoder_out, regatt_logit = self.decoder(seq, encoder_out, att_mask, seq_mask)

        losses = self.forward_after(encoder_out, decoder_out, regatt_logit)
        losses[cfg.LOSSES.XE_TYPE] = decoder_out

        return losses

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        history_states = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        if history_states is None:
            seq_len = 1
        else:
            seq_len = history_states[0].size(1) + 1
        seq_mask = subsequent_mask(seq_len).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(1)
        decoder_out, history_states = self.decoder.decode(wt, encoder_out, att_mask, seq_mask, history_states)
        decoder_out = decoder_out[:,-1,:]
        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, history_states


    def decode(self, **kwargs):
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
        
        att_feats = self.att_embed(att_feats)
        encoder_out = self.encoder(att_feats, att_mask)
        
        history_states = None
        ys = bos_input.unsqueeze(1)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
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
        
        return sents, logprobs

    def select(self, t, candidate_logprob, batch_size, beam_size):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def decode_beam(self, **kwargs):
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

        att_feats = self.att_embed(att_feats)
        encoder_out = self.encoder(att_feats, att_mask)

        history_states = None
        ys = bos_input.unsqueeze(1)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

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
                encoder_out = utils.expand_tensor_beam(encoder_out, beam_size)
                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

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

        return outputs, log_probs    


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        dropout,
        num_heads, 
        attention_dropout,
        ff_dropout,
        layer_num,
        use_aoa=False,
        aoa_drop=0.,
        layer_drop=0.
    ):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([])  
        self.layer_drop = layer_drop
        for _ in range(layer_num):
            layer = EncoderLayer(
                embed_dim,
                dropout,
                num_heads, 
                attention_dropout,
                ff_dropout,
                use_aoa,
                aoa_drop
            )
            self.layers.append(layer)

    def forward(self, x, mask):
        if self.layer_drop > 0:
            for layer in self.layers:
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layer_drop):
                    x = x
                else:
                    x = layer(x, mask)
        else:
            for layer in self.layers:
                x = layer(x, mask)
        return x
        
class EncoderLayer(nn.Module):
    def __init__(
        self, 
        embed_dim,
        dropout,
        num_heads, 
        attention_dropout,
        ff_dropout,
        use_aoa,
        aoa_drop, 
    ):
        super(EncoderLayer, self).__init__()
        self.encoder_attn = blocks.create(
            'MultiheadAttention',
            hidden_size = embed_dim, 
            num_attention_heads = num_heads, 
            attention_dropout=attention_dropout, 
            dropout=dropout,
            use_aoa=use_aoa,
            aoa_drop=aoa_drop)

        self.ff_layer = blocks.create(
            'FeedForward',
            embed_dim = embed_dim, 
            ffn_embed_dim = embed_dim * 4, 
            relu_dropout = ff_dropout, 
            dropout = ff_dropout)

    def forward(self, x, mask):
        x, _ = self.encoder_attn(
            x, 
            key=None, 
            value=None, 
            attention_mask=mask, 
            history_states=None)
        x = self.ff_layer(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size, 
        embed_dim,
        dropout,
        num_heads, 
        attention_dropout,
        ff_dropout,
        layer_num,
        use_aoa=False,
        aoa_drop=0.
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim
        for _ in range(layer_num):
            layer = DecoderLayer(
                embed_dim = embed_dim,
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

    def forward(self, tokens, encoder_out, att_mask, seq_mask):
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

        regatt_logit_arr = []
        for layer in self.layers:
            x, regatt_logit = layer(x, encoder_out, att_mask, seq_mask)
            regatt_logit_arr.append(regatt_logit.unsqueeze(1))
        regatt_logit = torch.cat(regatt_logit_arr, dim=1)

        x = self.dropout_lm(x)
        out = self.generator(x)
        return out, regatt_logit

    def decode(self, wt, encoder_out, att_mask, seq_mask, history_states=None):
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

        for i, layer in enumerate(self.layers):
            if history_states[i] is None:
                history_states[i] = x
            else:
                history_states[i] = torch.cat([history_states[i], x], dim=1)

            x, _ = layer(x, encoder_out, att_mask, seq_mask, history_states[i])

        x = self.dropout_lm(x)
        out = self.generator(x)
        return out, history_states




class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
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
            'MultiheadAttention',
            hidden_size = embed_dim, 
            num_attention_heads = num_heads, 
            attention_dropout=attention_dropout, 
            dropout=dropout,
            use_aoa=use_aoa,
            aoa_drop=aoa_drop)

        self.ff_layer = blocks.create(
            'FeedForward',
            embed_dim = embed_dim, 
            ffn_embed_dim = embed_dim * 4, 
            relu_dropout = ff_dropout, 
            dropout = ff_dropout)

    def forward(self, x, encoder_out, att_mask, seq_mask, history_states=None):
        x, _ = self.self_attn(x, key=None, value=None, attention_mask=seq_mask, history_states=history_states)
        x, regatt_logit = self.x_attn(x, key=encoder_out, value=encoder_out, attention_mask=att_mask)
        x = self.ff_layer(x)
        return x, regatt_logit