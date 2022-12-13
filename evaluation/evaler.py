import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
import datasets.data_loader as data_loader
from lib.config import cfg

class Evaler(object):
    def __init__(
        self,
        eval_ids,
        gv_feat,
        att_feats,
        obj_vocab_path,
        region_info_path,
        token_info_path,
        eval_annfile
    ):
        super(Evaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        self.eval_ids = np.array(utils.load_ids(eval_ids))
        self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats, obj_vocab_path, region_info_path, token_info_path)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask, region_labels, obj_pos_labels, obj_masks):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        kwargs[cfg.PARAM.REGION_LABEL] = region_labels
        kwargs[cfg.PARAM.OBJ_POS_LABEL] = obj_pos_labels
        kwargs[cfg.PARAM.OBJ_FEATS_MASK] = obj_masks
        return kwargs
        
    def __call__(self, cap_model, obj_model, rname):
        cap_model.eval()
        obj_model.eval()
        
        results = []
        obj_acc_total = 0
        obj_set_acc_total = 0
        obj_reg_acc_total = 0
        with torch.no_grad():
            for _, (indices, gv_feat, att_feats, att_mask, region_labels, obj_pos_labels, obj_masks) in tqdm.tqdm(enumerate(self.eval_loader)):
                ids = self.eval_ids[indices]
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()
                region_labels = region_labels.cuda()
                obj_pos_labels = obj_pos_labels.cuda()
                obj_masks = obj_masks.cuda()
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask, region_labels, obj_pos_labels, obj_masks)
                if kwargs['BEAM_SIZE'] > 1:
                    seq, _, loss_info = cap_model.module.decode_beam(obj_model, **kwargs)
                else:
                    seq, _, loss_info, _ = cap_model.module.decode(obj_model, **kwargs)
                obj_acc_total += loss_info['obj_acc_cnt']
                obj_set_acc_total += loss_info['obj_set_acc_cnt']

                if 'obj_reg_acc_cnt' in loss_info:
                    obj_reg_acc_total += loss_info['obj_reg_acc_cnt']

                sents = utils.decode_sequence(self.vocab, seq.data)
                for sid, sent in enumerate(sents):
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                    results.append(result)

            obj_acc = obj_acc_total / len(results)
            obj_set_acc = obj_set_acc_total / len(results)
            obj_reg_acc = obj_reg_acc_total / len(results)
        eval_res = self.evaler.eval(results)

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

        cap_model.train()
        obj_model.train()
        return eval_res, obj_acc, obj_set_acc, obj_reg_acc