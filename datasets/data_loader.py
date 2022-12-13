import os
import torch
from torchvision import transforms
from lib.config import cfg
from datasets.coco_dataset import CocoDataset
from datasets.coco_obj_dataset import CocoObjDataset
import samplers.distributed
import numpy as np

def sample_collate(batch):
    indices, input_seq, target_seq, gv_feat, att_feats, region_label, obj_seq, obj_pos_label, obj_mlabel, obj_mask, tokens2rid, mil_labels, obj_seq_target = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    mil_labels = torch.stack([torch.from_numpy(b) for b in mil_labels], 0)

    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)

    max_att_num = max_att_num - 1
    batch_size = len(region_label)
    seq_per_img = obj_seq[0].shape[0]
    obj_seq_len = [x.shape[1] for x in obj_seq]
    max_obj_seq_len = np.max(obj_seq_len)


    region_labels = np.zeros((batch_size, max_att_num, region_label[0].shape[1]), dtype='int') - 1
    for i, label in enumerate(region_label):
        num = label.shape[0]
        region_labels[i, 0:num, :] = label

    obj_masks = np.zeros((batch_size, seq_per_img, max_obj_seq_len), dtype=np.float32)
    obj_seqs = np.zeros((batch_size, seq_per_img, max_obj_seq_len), dtype='int')
    for i, o in enumerate(obj_seq):
        num = o.shape[1]
        obj_seqs[i, :, 0:num] = o
        obj_masks[i, :, 0:num] = obj_mask[i]

    obj_masks_pad = np.ones((batch_size, seq_per_img, 1), dtype=np.float32)
    obj_masks = np.concatenate([obj_masks_pad, obj_masks], axis=2)

    obj_pos_labels = np.zeros((batch_size, seq_per_img, max_obj_seq_len+1, max_att_num+1), dtype='int')
    obj_mlabels = np.zeros((batch_size, seq_per_img, max_obj_seq_len+1, max_att_num+1), dtype='int') - 1
    obj_seq_targets = np.zeros((batch_size, seq_per_img, max_obj_seq_len+1, obj_seq_target[0].shape[-1]), dtype='int') - 1
    for i, label in enumerate(obj_pos_label):
        obj_num = label.shape[1]
        att_num = label.shape[2]
        obj_pos_labels[i, :, 0:obj_num, 0:att_num] = label
        obj_mlabels[i, :, 0:obj_num, 0:att_num] = obj_mlabel[i]
        obj_seq_targets[i, :, 0:obj_num, :] = obj_seq_target[i]

    tokens2rids = np.zeros((batch_size, seq_per_img, input_seq.shape[-1], max_att_num), dtype='int')
    for i, label in enumerate(tokens2rid):
        att_num = label.shape[-1]
        tokens2rids[i, :, :, 0:att_num] = label

    region_labels = torch.from_numpy(region_labels)
    obj_seqs = torch.from_numpy(obj_seqs)
    obj_pos_labels = torch.from_numpy(obj_pos_labels)
    obj_mlabels = torch.from_numpy(obj_mlabels)
    obj_masks = torch.from_numpy(obj_masks)
    tokens2rids = torch.from_numpy(tokens2rids)
    obj_seq_targets = torch.from_numpy(obj_seq_targets)

    return indices, input_seq, target_seq, gv_feat, att_feats, att_mask, region_labels, obj_seqs, obj_pos_labels, obj_mlabels, obj_masks, tokens2rids, mil_labels, obj_seq_targets

def sample_collate_val(batch):
    indices, gv_feat, att_feats, region_label, obj_pos_label = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)

    region_labels = np.zeros((len(obj_pos_label), max_att_num - 1, region_label[0].shape[1]), dtype='int') - 1
    for i, label in enumerate(region_label):
        num = label.shape[0]
        region_labels[i, 0:num, :] = label

    obj_seq_len = [x.shape[0] for x in obj_pos_label]
    max_obj_seq_len = np.max(obj_seq_len)
    obj_pos_labels = np.zeros((len(obj_pos_label), max_obj_seq_len, max_att_num), dtype='int')
    obj_masks = np.zeros((len(obj_pos_label), max_obj_seq_len), dtype=np.float32)
    for i, label in enumerate(obj_pos_label):
        obj_num = label.shape[0]
        att_num = label.shape[1]
        obj_pos_labels[i, 0:obj_num, 0:att_num] = label
        obj_masks[i,0:obj_num] = 1
    obj_pos_labels = torch.from_numpy(obj_pos_labels)
    obj_masks = torch.from_numpy(obj_masks)
    region_labels = torch.from_numpy(region_labels)
       
    return indices, gv_feat, att_feats, att_mask, region_labels, obj_pos_labels, obj_masks


def load_train(distributed, epoch, coco_set):
    sampler = samplers.distributed.DistributedSampler(coco_set, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    
    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TRAIN.BATCH_SIZE,
        shuffle = shuffle, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = cfg.DATA_LOADER.DROP_LAST, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        sampler = sampler, 
        collate_fn = sample_collate
    )
    return loader

def load_val(image_ids_path, gv_feat_path, att_feats_folder, obj_vocab_path, region_info_path, token_info_path):
    coco_set = CocoDataset(
        image_ids_path = image_ids_path, 
        input_seq = None, 
        target_seq = None, 
        gv_feat_path = gv_feat_path, 
        att_feats_folder = att_feats_folder,
        obj_vocab_path = obj_vocab_path,
        region_info_path = region_info_path,
        token_info_path = token_info_path,
        seq_per_img = 1, 
        max_feat_num = cfg.DATA_LOADER.MAX_FEAT
    )

    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TEST.BATCH_SIZE,
        shuffle = False, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = False, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, 
        collate_fn = sample_collate_val
    )
    return loader