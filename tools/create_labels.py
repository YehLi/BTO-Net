import os
import json
import pickle
import argparse
import string
import numpy as np

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

def build_vocab(annos, args):
    count_thr = args.word_count_threshold

    # count up the number of words
    counts = {}
    for cocoid in annos:
        for sent in annos[cocoid]['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str,cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for cocoid in annos:
        for sent in annos[cocoid]['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len+1):
      print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
      # additional special UNK token we will use below to map infrequent words to
      print('inserting the special UNK token')
      vocab.append('UNK')
 
    for cocoid in annos:
        annos[cocoid]['final_captions'] = []
        for sent in annos[cocoid]['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
            assert len(caption) == len(txt)
            annos[cocoid]['final_captions'].append(caption)

    save_lines(vocab, os.path.join(args.outfolder, 'txt', 'coco_vocabulary.txt'))
    return vocab

def get_train_val_test(args, annos):
    train_id = read_lines(os.path.join(args.outfolder, '../txt/coco_train_image_id.txt'))
    val_id = read_lines(os.path.join(args.outfolder, '../txt/coco_val_image_id.txt'))
    test_id = read_lines(os.path.join(args.outfolder, '../txt/coco_test_image_id.txt'))

    train_id = [imgid for imgid in train_id if imgid in annos]
    val_id = [imgid for imgid in val_id if imgid in annos]
    test_id = [imgid for imgid in test_id if imgid in annos]

    save_lines(train_id, os.path.join(args.outfolder, './txt/coco_train_image_id.txt'))
    save_lines(val_id, os.path.join(args.outfolder, './txt/coco_val_image_id.txt'))
    save_lines(test_id, os.path.join(args.outfolder, './txt/coco_test_image_id.txt'))

    train_id = set(train_id)
    val_id = set(val_id)
    test_id = set(test_id)
    print("train num: " + str(len(train_id)))
    print("val num: " + str(len(val_id)))
    print("test num: " + str(len(test_id)))

    return train_id, val_id, test_id

def check_sent_num(args, annos, train_id):
    sent_num = 0
    for cocoid in annos:
        if cocoid in train_id:
            sent_num += len(annos[cocoid]['sentences'])
    print(sent_num)

    sent_num = 0
    anno2 = load_json("/export1/dataset/mscoco_torch/misc/dataset_coco_refine.json")['images']
    for img in anno2:
        cocoid = str(img['cocoid'])
        if cocoid in train_id:
            sent_num += len(img['sentences'])
    print(sent_num)

def save_regions_info(args, annos, obj_wtoi):
    regions_info = {}
    for cocoid in annos:
        assert len(annos[cocoid]['det_labels']) == len(annos[cocoid]['det_iou'])
        det_iou = np.array(annos[cocoid]['det_iou'])
        det_labels = []
        for wlist in annos[cocoid]['det_labels']:
            if len(wlist) == 0:
                det_labels.append([-1])
            else:
                wlist = [obj_wtoi.get(w) for w in wlist]
                det_labels.append(wlist)
        det_labels = np.array(det_labels)

        regions_info[cocoid] = { 'det_labels': det_labels, 'det_iou': det_iou }
    save_pickle(regions_info, os.path.join(args.outfolder, 'sent', 'tmp' ,'tmp_regions_info.pkl'))

def check_assert(args, annos):
    for cocoid in annos:
        n = len(annos[cocoid]['final_captions'])
        assert n == len(annos[cocoid]['tokens2label']) 
        assert n == len(annos[cocoid]['tokens2regionid'])
        for j,s in enumerate(annos[cocoid]['final_captions']):
            assert len(annos[cocoid]['tokens2label'][j]) == len(s)
            assert len(annos[cocoid]['tokens2regionid'][j]) == len(s)
    print('pass assert')

def save_seq_info(args, annos, wtoi, obj_wtoi, id_set, set_name):
    input_seq_map = {}
    target_seq_map = {}
    tokens_map = {}

    for cocoid in annos:
        if cocoid not in id_set:
            continue
        n = len(annos[cocoid]['final_captions'])
        input_seq = np.zeros((n, args.max_length + 1), dtype='uint32')
        target_seq = np.zeros((n, args.max_length + 1), dtype='int32') - 1
        tokens2rid = annos[cocoid]['tokens2regionid']
        tokens2label = annos[cocoid]['tokens2label']

        for j,s in enumerate(annos[cocoid]['final_captions']):
            tokens2label[j] = [obj_wtoi[label] if label is not None else -1 for label in tokens2label[j]]

            for k,w in enumerate(s):
                if k < args.max_length:
                    input_seq[j,k+1] = wtoi[w]
                    target_seq[j,k] = wtoi[w]

            seq_len = len(s)
            if seq_len <= args.max_length:
                target_seq[j,seq_len] = 0
            else:
                target_seq[j,args.max_length] = wtoi[s[args.max_length]]

        input_seq_map[cocoid] = input_seq
        target_seq_map[cocoid] = target_seq
        tokens_map[cocoid] = { 'tokens2rid': tokens2rid, 'tokens2label': tokens2label }
    
    save_pickle(tokens_map, os.path.join(args.outfolder, 'sent', 'tmp' ,'tmp_tokens_' + set_name + '_info.pkl'))
    save_pickle(input_seq_map, os.path.join(args.outfolder, 'sent', 'coco_' + set_name + '_input.pkl'))
    save_pickle(target_seq_map, os.path.join(args.outfolder, 'sent', 'coco_' + set_name + '_target.pkl'))

def main(args):
    annos = load_json(args.infile)
    train_id, val_id, test_id = get_train_val_test(args, annos)
    
    vocab = build_vocab(annos, args)
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    obj_vocab = read_lines(os.path.join(args.outfolder, 'txt' ,'obj_vocab.txt'))
    obj_wtoi = {w:i+1 for i,w in enumerate(obj_vocab)} # inverse table
    save_regions_info(args, annos, obj_wtoi)
    check_assert(args, annos)

    save_seq_info(args, annos, wtoi, obj_wtoi, train_id, 'train')
    save_seq_info(args, annos, wtoi, obj_wtoi, val_id, 'val')
    save_seq_info(args, annos, wtoi, obj_wtoi, test_id, 'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--infile', default='/export1/task/caption/btonet_caption/data/VG_COCOEntities_ASG/anno_imagelevel_50box.json')
    parser.add_argument('--outfolder', default='/export1/dataset/mscoco_torch/btonet')

    parser.add_argument('--max_length', default=16, type=int)
    parser.add_argument('--word_count_threshold', default=5, type=int)

    args = parser.parse_args()
    main(args)
