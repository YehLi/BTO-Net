import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
from lib.config import cfg

class CocoDataset(data.Dataset):
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

        g_att_feats = np.sum(att_feats, axis=0) / att_feats.shape[0]
        att_feats = np.concatenate([np.expand_dims(g_att_feats, axis=0), att_feats], axis=0)

        if self.seq_len < 0:
            obj_seq_len = [len(objs) for objs in self.token_info[image_id]['objects_pos']]
            max_seq_len = np.max(obj_seq_len)
            max_obj_ix = np.argmax(obj_seq_len)
            obj_pos_label = np.zeros((max_seq_len+1, att_feats.shape[0]), dtype='int')
            obj_ix = max_obj_ix
            obj_seq_len = len(self.token_info[image_id]['objects_pos'][obj_ix])
            for j, obj_idx_set in enumerate(self.token_info[image_id]['objects_pos'][obj_ix]):
                obj_idx_set = np.array(obj_idx_set)
                obj_pos_label[j, obj_idx_set+1] = 1
            obj_pos_label[obj_seq_len, 0] = 1

            ################################# regino_info #################################
            regino_info = self.region_info[image_id]['mobj_labels'] # full_labels, det_iou
            regino_label = np.zeros((att_feats.shape[0] - 1, len(self.obj_vocab)+1), dtype='int') - 1
            for i in range(att_feats.shape[0] - 1):
                for rinfo in regino_info[i]:
                    if rinfo != -1:
                        regino_label[i, rinfo] = 1
            ################################# regino_info #################################

            return indices, gv_feat, att_feats, regino_label, obj_pos_label

        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
           
        n = len(self.input_seq[image_id])   
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)                
        else:
            sid = n
            ixs = random.sample(range(n), self.seq_per_img - n)
            input_seq[0:n, :] = self.input_seq[image_id]
            target_seq[0:n, :] = self.target_seq[image_id]
           
        for i, ix in enumerate(ixs):
            input_seq[sid + i] = self.input_seq[image_id][ix,:]
            target_seq[sid + i] = self.target_seq[image_id][ix,:]
        return indices, input_seq, target_seq, gv_feat, att_feats
