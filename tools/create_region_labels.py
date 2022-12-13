import os
import json
import pickle
import argparse
import string
import numpy as np
from tqdm import tqdm
import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()

max_bbox = 50
fg_thres = 0.75
bg_thres = 0.5
MISC_ROOT = '/export1/task/caption/btonet_caption/data/misc'

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

def load_synonym_map():
    synonym_map2 = {}
    synonym_map3 = {}
    lines = read_lines(os.path.join(MISC_ROOT, 'hierarchy_class_name.txt'))
    for line in lines:
        w1, w2, w3 = line.split(',')
        synonym_map2[w1] = w2
        synonym_map3[w1] = w3
        synonym_map3[w2] = w3
    return synonym_map2, synonym_map3
synonym_map2, synonym_map3 = load_synonym_map()

def get_lemma(word):
    doc = nlp(word)
    lemmas = [token.lemma_ for token in doc]
    return lemmas[-1]


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

def get_obj_name(name, obj_vocab):
    if name in obj_vocab:
        return name
    name2 = synonym_map2.get(name, name)
    if name2 in obj_vocab:
        return name2

    if ' ' in name:
        n1, n2 = name.split(' ')
        n1 = get_lemma(n1)
        n2 = get_lemma(n2)
        name_lemma = n1 + ' ' + n2
    else:
        name_lemma = get_lemma(name)
    if name_lemma in obj_vocab:
        return name_lemma
    name_lemma2 = synonym_map2.get(name_lemma, name_lemma)
    if name_lemma2 in obj_vocab:
        return name_lemma2
    return None

def get_labels_by_iou(pred_bbox, gt_bbox, thres, obj_wtoi):
    gt_names = [box[0] for box in gt_bbox]
    gt_box_list = np.array([box[1] for box in gt_bbox])
    iou_list = get_iou(pred_bbox, gt_box_list)

    labels = []
    for i, iou in enumerate(iou_list):
        if iou >= thres:
            labels.append(obj_wtoi[gt_names[i]])

    if len(labels) == 0:
        labels.append(-1)
    return labels

def is_bg_bbox(pred_bbox, gt_bbox, thres):
    gt_box_list = np.array([box[1] for box in gt_bbox])
    iou_list = get_iou(pred_bbox, gt_box_list)
    for i, iou in enumerate(iou_list):
        if iou > thres:
            return False
    return True

def main(args):
    region_tmp = load_pickle(args.region_tmp_file)
    bbox_anno = load_json(args.anno_file)
    obj_vocab = read_lines(os.path.join(args.outfolder, 'txt' ,'obj_vocab.txt'))
    obj_itow = {i+1:w for i,w in enumerate(obj_vocab)}
    obj_wtoi = {w:i+1 for i,w in enumerate(obj_vocab)} # inverse table
    obj_vocab_set = set(obj_vocab)

    region_info = {}

    for cocoid in tqdm(region_tmp):

        det_mentioned_obj_labels = region_tmp[cocoid]['det_labels']
        det_full_obj_labels = [[-1] for _ in range(len(det_mentioned_obj_labels))]

        det_iou = region_tmp[cocoid]['det_iou']
        detection_bboxes = bbox_anno[cocoid]['detection']

        gt_bbox = []
        if 'vg' in bbox_anno[cocoid]:
            objects = bbox_anno[cocoid]['vg']['objects']
            for obj in objects:
                name, bbox = obj['name'], obj['bbox']
                name = get_obj_name(name, obj_vocab_set)
                if name is not None:
                    gt_bbox.append((name, tuple(bbox)))
        if 'coco_det' in bbox_anno[cocoid]:
            objects = bbox_anno[cocoid]['coco_det']['objects']
            for obj in objects:
                name, bbox = obj['name'], obj['bbox']
                name = get_obj_name(name, obj_vocab_set)
                if name is not None:
                    gt_bbox.append((name, tuple(bbox)))
        gt_bbox = list(set(gt_bbox))
        gt_bbox = [(bbox[0], list(bbox[1])) for bbox in gt_bbox]

        for i, pred_bbox in enumerate(detection_bboxes):
            if i >= max_bbox:
                continue
            if det_mentioned_obj_labels[i][0] != -1:
                det_full_obj_labels[i] = det_mentioned_obj_labels[i]
                continue

            if len(gt_bbox) > 0:
                det_full_obj_labels[i] = get_labels_by_iou(pred_bbox, gt_bbox, fg_thres, obj_wtoi)
                

        mentioned_bbox = []
        for i, label in enumerate(det_mentioned_obj_labels):
            if label[0] != -1:
                mentioned_bbox.append((obj_itow[label[0]], detection_bboxes[i]))

        gt_bbox.extend(mentioned_bbox)
        assert len(gt_bbox) > 0, str(cocoid)
        for i, pred_bbox in enumerate(detection_bboxes):
            if i >= max_bbox:
                continue
            if det_full_obj_labels[i][0] != -1:
                continue
            if len(gt_bbox) > 0 and is_bg_bbox(pred_bbox, gt_bbox, bg_thres):
                det_mentioned_obj_labels[i] = [0]
                det_full_obj_labels[i] = [0]

        region_info[cocoid] = { 'det_iou': det_iou, 'mobj_labels': det_mentioned_obj_labels, 'full_labels': det_full_obj_labels }
    
    save_pickle(region_info, os.path.join(args.outfolder, 'sent', 'regions_info.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--region_tmp_file', default='/export1/dataset/mscoco_torch/btonet/sent/tmp/tmp_regions_info.pkl')
    parser.add_argument('--anno_file', default='/export1/task/caption/btonet_caption/data/VG_COCOEntities_ASG/anno_align_bbox.json')
    parser.add_argument('--outfolder', default='/export1/dataset/mscoco_torch/btonet')

    args = parser.parse_args()
    main(args)