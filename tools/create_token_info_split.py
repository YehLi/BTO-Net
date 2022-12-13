import os
import json
import pickle
import argparse
import string
import numpy as np
from tqdm import tqdm

max_bbox = 50
bg_thres = 0.1
max_length = 16

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]

def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))

def get_iou(pred_box, gt_box_list):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = np.maximum(pred_box[0], gt_box_list[:,0])
    ixmax = np.minimum(pred_box[2], gt_box_list[:,2])
    iymin = np.maximum(pred_box[1], gt_box_list[:,1])
    iymax = np.minimum(pred_box[3], gt_box_list[:,3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box_list[:,2] - gt_box_list[:,0] + 1.) * (gt_box_list[:,3] - gt_box_list[:,1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou

def split_rids(rids, bbox):
    if len(rids) == 1:
        return [[rids[0][0]]]
    rid_scores = np.array([r[1] for r in rids])
    rid_idx = np.array([r[0] for r in rids])
    order = rid_scores.argsort()[::-1]
    rid_idx = rid_idx[order]

    result = [[rid_idx[0]]]
    for i in range(1, len(rid_idx)):
        cur_bbox_id = rid_idx[i]
        cur_bbox = bbox[cur_bbox_id]

        is_find = False
        for j in range(len(result)):
            id_arr = np.array(result[j])
            bbox_arr = bbox[id_arr]
            iou_list = get_iou(cur_bbox, bbox_arr)
            iou_max = max(iou_list)
            if iou_max >= bg_thres:
                result[j].append(cur_bbox_id)
                is_find = True
                break
        if not is_find:
            result.append([cur_bbox_id])
    return result





def token_info(annos, phase):
    token_info = {}
    tmp_token_info = load_pickle(os.path.join(args.outfolder, 'sent', 'tmp' ,'tmp_tokens_' + phase + '_info.pkl'))
    avg_len = 0
    avg_count = 0
    for cocoid in tqdm(tmp_token_info):
        n = len(tmp_token_info[cocoid]['tokens2label'])
        detection_bbox = annos[cocoid]['detection'][:max_bbox]
        num_bbox = len(detection_bbox)
        detection_bbox = np.array(detection_bbox)

        tokens2rid_seq_arr = []
        tokens2label_seq_arr = []
        objects_pos_seq_arr = []
        objects_pos_seq_full_arr = []
        objects_neg_seq_arr = []

        for i in range(n):
            tokens2rid = tmp_token_info[cocoid]['tokens2rid']
            tokens2label = tmp_token_info[cocoid]['tokens2label']

            tokens2rid_seq = []
            tokens2label_seq = []
            objects_pos_seq = []
            objects_pos_full_seq = []
            objects_neg_seq = []
            for j, rids in enumerate(tokens2rid[i]):
                if tokens2label[i][j] == -1:
                    tokens2rid_seq.append(rids)
                    tokens2label_seq.append(tokens2label[i][j])
                else:
                    _rids = [r[0] for r in rids]
                    tokens2rid_seq.append(_rids)
                    tokens2label_seq.append(tokens2label[i][j])

                    rids_arr = split_rids(rids, detection_bbox)
                    for rids_ent in rids_arr:
                        if len(objects_pos_seq) == 0 or rids_ent != objects_pos_seq[len(objects_pos_seq) - 1]:
                            objects_pos_seq.append(list(set(rids_ent)))
                            objects_pos_full_seq.append(_rids)
                            rids_set = set(_rids)
                            objects_neg_seq.append([i for i in range(num_bbox) if i not in rids_set])
            
            tokens2rid_seq = tokens2rid_seq[:max_length+1]
            tokens2label_seq = tokens2label_seq[:max_length+1]
            while len(tokens2rid_seq) < max_length + 1:
                tokens2rid_seq.append([])
                tokens2label_seq.append(-1)

            tokens2rid_seq_arr.append(tokens2rid_seq)
            tokens2label_seq_arr.append(tokens2label_seq)
            objects_pos_seq_arr.append(objects_pos_seq)
            objects_pos_seq_full_arr.append(objects_pos_full_seq)
            objects_neg_seq_arr.append(objects_neg_seq)

            avg_len += len(objects_pos_seq)
            avg_count += 1
        
        token_info[cocoid] = {
            'tokens2rid': tokens2rid_seq_arr,
            'tokens2label': tokens2label_seq_arr,
            'objects_pos': objects_pos_seq_arr,
            'objects_full_pos': objects_pos_seq_full_arr,
            'objects_neg': objects_neg_seq_arr
        }

    save_pickle(token_info, os.path.join(args.outfolder, 'sent', 'tokens_' + phase + '_info_split.pkl'))

    print(avg_len * 1.0 / avg_count)

def main(args):
    annos = load_json(args.infile)
    token_info(annos, 'train')
    token_info(annos, 'val')
    token_info(annos, 'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default='/export1/task/caption/btonet_caption/data/VG_COCOEntities_ASG/anno_imagelevel_50box.json')
    parser.add_argument('--outfolder', default='/export1/dataset/mscoco_torch/btonet')

    args = parser.parse_args()
    main(args)
