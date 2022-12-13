import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
from lib.config import cfg

class CocoObjDataset(data.Dataset):
    def __init__(
        self, 
        image_ids_path, 
        input_seq, 
        target_seq,
        gv_feat_path, 
        att_feats_folder, 
        obj_vocab_path,
        region_info_path,
        token_info_path,
        seq_per_img,
        max_feat_num
    ):
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        self.image_ids = utils.load_lines(image_ids_path)
        self.att_feats_folder = att_feats_folder if len(att_feats_folder) > 0 else None
        self.gv_feat = pickle.load(open(gv_feat_path, 'rb'), encoding='bytes') if len(gv_feat_path) > 0 else None

        if input_seq is not None and target_seq is not None:
            self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
            self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
            self.seq_len = len(self.input_seq[self.image_ids[0]][0,:])
        else:
            self.seq_len = -1
            self.input_seq = None
            self.target_seq = None

        self.obj_vocab = utils.load_lines(obj_vocab_path)
        self.region_info = pickle.load(open(region_info_path, 'rb'), encoding='bytes')
        self.token_info = pickle.load(open(token_info_path, 'rb'), encoding='bytes')

        if cfg.LOSSES.GV_SUP.LOSS_WEIGHT > 0:
            self.mil_info = pickle.load(open(cfg.DATA_LOADER.MIL_LABEL, 'rb'), encoding='bytes')
    
    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        indices = np.array([index]).astype('int')

        if self.gv_feat is not None:
            gv_feat = self.gv_feat[image_id]
            gv_feat = np.array(gv_feat).astype('float32')
        else:
            gv_feat = np.zeros((1,1))

        if self.att_feats_folder is not None:
            if cfg.DATA_LOADER.NUM_WORKERS == 0:
                att_feats = np.load(os.path.join(self.att_feats_folder, str(image_id) + '.npz'))['feat']
            else:
                content = np.load(os.path.join(self.att_feats_folder, str(image_id) + '.npz'))
                att_feats = content['features']
            att_feats = np.array(att_feats).astype('float32')
        else:
            att_feats = np.zeros((1,1))
        
        if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:
            att_feats = att_feats[:self.max_feat_num, :]

        if self.seq_len < 0:
            g_att_feats = np.sum(att_feats, axis=0) / att_feats.shape[0]
            att_feats = np.concatenate([np.expand_dims(g_att_feats, axis=0), att_feats], axis=0)
            return indices, gv_feat, att_feats

        ################################# regino_info #################################
        regino_info = self.region_info[image_id]['mobj_labels'] # full_labels, det_iou
        regino_label = np.zeros((att_feats.shape[0], len(self.obj_vocab)+1), dtype='int') - 1
        for i in range(att_feats.shape[0]):
            for rinfo in regino_info[i]:
                if rinfo != -1:
                    regino_label[i, rinfo] = 1
        ################################# regino_info #################################

        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')

        sample_list = []   
        n = len(self.input_seq[image_id])   
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)
        else:
            #sid = n
            #ixs = random.sample(range(n), self.seq_per_img - n)
            #input_seq[0:n, :] = self.input_seq[image_id]
            #target_seq[0:n, :] = self.target_seq[image_id]
            #sample_list.extend(list(range(n)))
            sid = n
            ixs = random.choices(list(range(n)), k=self.seq_per_img - n)
            input_seq[0:n, :] = self.input_seq[image_id]
            target_seq[0:n, :] = self.target_seq[image_id]
            sample_list.extend(list(range(n)))

        for i, ix in enumerate(ixs):
            input_seq[sid + i] = self.input_seq[image_id][ix,:]
            target_seq[sid + i] = self.target_seq[image_id][ix,:]
            sample_list.append(ix)

        ################################## obj seq ##################################
        #obj_seq_len = []
        #for ix in sample_list:
        #    obj_seq_len.append(len(self.token_info[image_id]['objects_pos'][ix]))
        #max_seq_len = np.max(obj_seq_len)
        obj_seq_len = [len(objs) for objs in self.token_info[image_id]['objects_pos']]
        max_seq_len = np.max(obj_seq_len)
        max_obj_ix = np.argmax(obj_seq_len)
        #############################################################################

        obj_seq = np.zeros((self.seq_per_img, max_seq_len), dtype='int')
        obj_seq_target = np.zeros((self.seq_per_img, max_seq_len+1, len(self.obj_vocab)+1), dtype='int') - 1
        obj_mask = np.zeros((self.seq_per_img, max_seq_len), dtype=np.float32)
        obj_pos_label = np.zeros((self.seq_per_img, max_seq_len+1, att_feats.shape[0]+1), dtype='int')
        obj_mlabel = np.zeros((self.seq_per_img, max_seq_len+1, att_feats.shape[0]+1), dtype='int') - 1
        tokens2rid = np.zeros((self.seq_per_img, self.seq_len, att_feats.shape[0]))

        for i, sam_ix in enumerate(sample_list):
            if cfg.DATA_LOADER.USE_LONG_OBJ == True:
                obj_ix = max_obj_ix
            else:
                obj_ix = sam_ix

            obj_seq_len = len(self.token_info[image_id]['objects_pos'][obj_ix])
            assert obj_seq_len == len(self.token_info[image_id]['objects_neg'][obj_ix])
            for j, obj_idx_set in enumerate(self.token_info[image_id]['objects_pos'][obj_ix]):
                obj_idx_set = np.array(obj_idx_set)
                r = random.sample(range(len(obj_idx_set)), 1)[0]
                obj_seq[i, j] = obj_idx_set[r]
                obj_mask[i,j] = 1

                obj_seq_target[i, j] = regino_label[obj_seq[i, j]]

                if 'objects_full_pos' in self.token_info[image_id]:
                    obj_idx_set_full = self.token_info[image_id]['objects_full_pos'][obj_ix][j]
                    obj_idx_set_full = np.array(obj_idx_set_full)
                    obj_pos_label[i, j, obj_idx_set_full+1] = 1
                else:
                    obj_pos_label[i, j, obj_idx_set+1] = 1

                top_list = list(obj_idx_set+1)
                for k in range(len(top_list)):
                    obj_mlabel[i, j, k] = top_list[k]
            obj_pos_label[i, obj_seq_len, 0] = 1
            obj_mlabel[i, obj_seq_len, 0] = 0
            obj_seq_target[i, obj_seq_len, 0] = 1

            #for j, obj_idx_set in enumerate(self.token_info[image_id]['objects_neg'][obj_ix]):
            #    obj_idx_set = np.array(obj_idx_set)
            #    if len(obj_idx_set) > 0:
            #        obj_neg_label[i, j, obj_idx_set+1] = 1
            #    obj_neg_label[i, j, 0] = 1
            #obj_neg_label[i, obj_seq_len, 1:] = 1

        for i, ix in enumerate(sample_list):
            rid_info = self.token_info[image_id]['tokens2rid'][ix]
            for j, rids in enumerate(rid_info):
                for rid in rids:
                    assert rid >= 0
                    tokens2rid[i, j, rid] = 1

        g_att_feats = np.sum(att_feats, axis=0) / att_feats.shape[0]
        att_feats = np.concatenate([np.expand_dims(g_att_feats, axis=0), att_feats], axis=0)

        if cfg.LOSSES.GV_SUP.LOSS_WEIGHT > 0:
            mil_labels = self.mil_info[int(image_id)]
        else:
            mil_labels = np.zeros((1,1))

        return indices, input_seq, target_seq, gv_feat, att_feats, regino_label, obj_seq, obj_pos_label, obj_mlabel, obj_mask, tokens2rid, mil_labels, obj_seq_target