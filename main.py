import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file
import layers

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")

        self.rl_stage = False
        self.setup_logging()
        self.setup_dataset()
        self.setup_network()
        #self.val_evaler = Evaler(
        #    eval_ids = cfg.DATA_LOADER.VAL_ID,
        #    gv_feat = cfg.DATA_LOADER.VAL_GV_FEAT,
        #    att_feats = cfg.DATA_LOADER.VAL_ATT_FEATS,
        #    eval_annfile = cfg.INFERENCE.VAL_ANNFILE
        #)
        self.test_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            gv_feat = cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            obj_vocab_path = cfg.DATA_LOADER.OBJ_VOCAB_PATH,
            region_info_path = cfg.DATA_LOADER.REGION_INFO_PATH,
            token_info_path = cfg.DATA_LOADER.TOKEN_INFO_TEST_PATH,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE
        )
        self.scorer = Scorer()

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        cap_model = models.create(cfg.MODEL.TYPE)
        obj_model = models.create('Objformer')

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.cap_model = torch.nn.parallel.DistributedDataParallel(
                cap_model.to(self.device), 
                find_unused_parameters=True,
                device_ids = [self.args.local_rank], 
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
            self.obj_model = torch.nn.parallel.DistributedDataParallel(
                obj_model.to(self.device), 
                find_unused_parameters=True,
                device_ids = [self.args.local_rank], 
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
        else:
            self.cap_model = torch.nn.DataParallel(cap_model).cuda()
            self.obj_model = torch.nn.DataParallel(obj_model).cuda()

        if self.args.resume > 0:
            self.cap_model.load_state_dict(
                torch.load(self.snapshot_path("caption_model_cap", self.args.resume),
                    map_location=lambda storage, loc: storage)
            )
            self.obj_model.load_state_dict(
                torch.load(self.snapshot_path("caption_model_obj", self.args.resume),
                    map_location=lambda storage, loc: storage)
            )

        self.cap_optim = None
        self.obj_optim = None
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()

        if cfg.LOSSES.NCE.LOSS_WEIGHT > 0:
            self.nce_criterion = losses.create('NCE').cuda()
        if cfg.LOSSES.REGION.LOSS_WEIGHT > 0:
            self.region_criterion = losses.create('MultiLabel').cuda()
        if cfg.LOSSES.REGION_ATT.LOSS_WEIGHT > 0:
            self.region_att_criterion = losses.create('Attention').cuda()
        if cfg.LOSSES.OBJ_DEC_ATT.LOSS_WEIGHT > 0:
            self.obj_dec_att_criterion = losses.create('ObjAttention').cuda()
        if cfg.LOSSES.GV_SUP.LOSS_WEIGHT > 0:
            self.mil_criterion = losses.create('MIL').cuda()


    def setup_dataset(self):
        self.coco_set = datasets.coco_obj_dataset.CocoObjDataset(            
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID, 
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH, 
            target_seq = cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path = cfg.DATA_LOADER.TRAIN_GV_FEAT, 
            att_feats_folder = cfg.DATA_LOADER.TRAIN_ATT_FEATS, 
            obj_vocab_path = cfg.DATA_LOADER.OBJ_VOCAB_PATH,
            region_info_path = cfg.DATA_LOADER.REGION_INFO_PATH,
            token_info_path = cfg.DATA_LOADER.TOKEN_INFO_PATH,
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num = cfg.DATA_LOADER.MAX_FEAT,
        )

    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.coco_set)

    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None
            
        #val_res = self.val_evaler(self.model, 'val_' + str(epoch + 1))
        #self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
        #self.logger.info(str(val_res))

        test_res, obj_acc, obj_set_acc, obj_reg_acc = self.test_evaler(self.cap_model, self.obj_model, 'test_' + str(epoch + 1))
        self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
        self.logger.info(str(test_res))
        self.logger.info('obj acc: ' + str(obj_acc))
        self.logger.info('obj set acc: ' + str(obj_set_acc))
        self.logger.info('obj reg acc: ' + str(obj_reg_acc))

        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            #val -= val_res[score_type] * weight
            val -= test_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.cap_model.state_dict(), self.snapshot_path("caption_model_cap", epoch+1))
        torch.save(self.obj_model.state_dict(), self.snapshot_path("caption_model_obj", epoch+1))

    def make_kwargs(
        self, 
        indices, 
        input_seq, 
        target_seq, 
        gv_feat, 
        att_feats, 
        att_mask, 
        region_labels,
        obj_seqs,
        obj_pos_labels,
        obj_mlabels,
        obj_att_mask,
        tokens2rid,
        mil_labels,
        obj_seq_targets
    ):
        
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor)
        seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.ATT_FEATS_MASK: att_mask,
            cfg.PARAM.REGION_LABEL: region_labels,
            cfg.PARAM.OBJ_SEQ: obj_seqs,
            cfg.PARAM.OBJ_POS_LABEL: obj_pos_labels.view(-1, obj_pos_labels.shape[-2], obj_pos_labels.shape[-1]),
            cfg.PARAM.OBJ_MLABEL: obj_mlabels.view(-1, obj_mlabels.shape[-2], obj_mlabels.shape[-1]),
            cfg.PARAM.OBJ_FEATS_MASK: obj_att_mask.view(-1, obj_att_mask.shape[-1]),
            cfg.PARAM.TOKENS2RID_LABEL: tokens2rid.view(-1, tokens2rid.shape[-2], tokens2rid.shape[-1]),
            cfg.PARAM.MIL_LABEL: mil_labels,
            cfg.PARAM.OBJ_SEQ_TARGETS: obj_seq_targets
        }
        return kwargs

    def scheduled_cap_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.CAP_START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.CAP_START) // cfg.TRAIN.SCHEDULED_SAMPLING.CAP_INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.CAP_INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.CAP_MAX_PROB)
            self.cap_model.module.ss_prob = ss_prob

    def scheduled_obj_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.OBJ_START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.OBJ_START) // cfg.TRAIN.SCHEDULED_SAMPLING.OBJ_INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.OBJ_INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.OBJ_MAX_PROB)
            self.obj_model.module.ss_prob = ss_prob

    def display(self, iteration, data_time, batch_time, losses, loss_info):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        self.logger.info('Iteration ' + str(iteration) + 
            info_str + 
            ', cap lr = ' + str(self.cap_optim.get_lr()) + 
            ', obj lr = ' + str(self.obj_optim.get_lr())
        )
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()

    def calculate_loss(self, losses, **kwargs):
        loss_info = {}
        loss = 0

        ################### xe loss ######################
        if cfg.LOSSES.XE_TYPE in losses:
            xe_loss, xe_loss_info = self.xe_criterion(losses[cfg.LOSSES.XE_TYPE], kwargs[cfg.PARAM.TARGET_SENT])
            loss = loss + xe_loss
            for key in xe_loss_info:
                loss_info[key] = xe_loss_info[key]
        
        #################### region loss ####################
        if cfg.LOSSES.REGION.LOSS_NAME in losses:
            reg_loss, reg_loss_info = self.region_criterion(
                losses[cfg.LOSSES.REGION.LOSS_NAME], 
                utils.expand_tensor(kwargs[cfg.PARAM.REGION_LABEL], cfg.DATA_LOADER.SEQ_PER_IMG)
            )
            loss = loss + reg_loss * cfg.LOSSES.REGION.LOSS_WEIGHT
            for key in reg_loss_info:
                loss_info[key] = reg_loss_info[key]

        #################### obj region dec loss ####################
        if cfg.LOSSES.OBJ_REGION_DEC.LOSS_NAME in losses:
            #obj_reg_label = utils.expand_tensor(kwargs[cfg.PARAM.REGION_LABEL], cfg.DATA_LOADER.SEQ_PER_IMG)
            #obj_seq = kwargs[cfg.PARAM.OBJ_SEQ]
            #obj_seq = obj_seq.view(-1, obj_seq.shape[-1])
            obj_mask = kwargs[cfg.PARAM.OBJ_FEATS_MASK]
            obj_mask = obj_mask.view(-1, obj_mask.shape[-1])
            #obj_reg_label = torch.gather(obj_reg_label, 1, obj_seq.unsqueeze(-1).expand(obj_seq.shape[0], obj_seq.shape[1], obj_reg_label.shape[-1]))
            
            #obj_mask = obj_mask[:, 1:]
            #obj_reg_label = obj_reg_label.masked_fill(obj_mask.unsqueeze(-1) == 0, -1)

            #obj_reg_dec_logit = losses[cfg.LOSSES.OBJ_REGION_DEC.LOSS_NAME][:, :-1]

            obj_reg_label = kwargs[cfg.PARAM.OBJ_SEQ_TARGETS]
            obj_reg_dec_logit = losses[cfg.LOSSES.OBJ_REGION_DEC.LOSS_NAME]

            obj_reg_dec_loss, obj_reg_dec_loss_info = self.region_criterion(
                obj_reg_dec_logit,
                obj_reg_label
            )
            loss = loss + obj_reg_dec_loss * cfg.LOSSES.OBJ_REGION_DEC.LOSS_WEIGHT
            for key in obj_reg_dec_loss_info:
                loss_info['obj_dec_' + key] = obj_reg_dec_loss_info[key]

        #################### obj region loss ####################
        if cfg.LOSSES.OBJ_REGION.LOSS_NAME in losses:
            obj_reg_logit = losses[cfg.LOSSES.OBJ_REGION.LOSS_NAME]#[:, :-1]
            obj_reg_loss, obj_reg_loss_info = self.region_criterion(
                obj_reg_logit, 
                utils.expand_tensor(kwargs[cfg.PARAM.REGION_LABEL], cfg.DATA_LOADER.SEQ_PER_IMG)
            )
            loss = loss + obj_reg_loss * cfg.LOSSES.OBJ_REGION.LOSS_WEIGHT
            for key in obj_reg_loss_info:
                loss_info['obj_' + key] = obj_reg_loss_info[key]

        
        #################### nce loss ####################
        if cfg.LOSSES.NCE.LOSS_NAME in losses:
            nce_loss, nce_loss_info = self.nce_criterion(
                losses[cfg.LOSSES.NCE.LOSS_NAME],
                kwargs[cfg.PARAM.OBJ_POS_LABEL],
                kwargs[cfg.PARAM.OBJ_MLABEL],
                utils.expand_tensor(kwargs[cfg.PARAM.ATT_FEATS_MASK], cfg.DATA_LOADER.SEQ_PER_IMG),
                kwargs[cfg.PARAM.OBJ_FEATS_MASK],
            )
            loss = loss + nce_loss * cfg.LOSSES.NCE.LOSS_WEIGHT
            for key in nce_loss_info:
                loss_info[key] = nce_loss_info[key]

        #################### region attention loss ####################
        if cfg.LOSSES.REGION_ATT.LOSS_NAME in losses:
            reg_att_loss, reg_att_loss_info = self.region_att_criterion(
                losses[cfg.LOSSES.REGION_ATT.LOSS_NAME], # batch_size, layernum, head_num, seq_len, att_num
                kwargs[cfg.PARAM.TOKENS2RID_LABEL]
            )
            loss = loss + reg_att_loss * cfg.LOSSES.REGION_ATT.LOSS_WEIGHT
            for key in reg_att_loss_info:
                loss_info[key] = reg_att_loss_info[key]

        #################### obj decoder att loss ####################
        if cfg.LOSSES.OBJ_DEC_ATT.LOSS_NAME in losses:
            obj_dec_att_loss, obj_dec_att_loss_info = self.obj_dec_att_criterion(
                losses[cfg.LOSSES.OBJ_DEC_ATT.LOSS_NAME],
                kwargs[cfg.PARAM.OBJ_POS_LABEL][:, :, 1:]
            )
            loss = loss + obj_dec_att_loss * cfg.LOSSES.OBJ_DEC_ATT.LOSS_WEIGHT
            for key in obj_dec_att_loss_info:
                loss_info['obj_dec_' + key] = obj_dec_att_loss_info[key]

        ##################### mil ###################################
        if 'o_' + cfg.LOSSES.GV_SUP.LOSS_NAME in losses:
            #loss_sup = nn.MultiLabelMarginLoss()(
            #    losses['o_' + cfg.LOSSES.GV_SUP.LOSS_NAME], 
            #    utils.expand_tensor(kwargs[cfg.PARAM.MIL_LABEL], cfg.DATA_LOADER.SEQ_PER_IMG))
            #loss_sup = nn.MultiLabelSoftMarginLoss()(
            #    losses['o_' + cfg.LOSSES.GV_SUP.LOSS_NAME], 
            #    utils.expand_tensor(kwargs[cfg.PARAM.MIL_LABEL], cfg.DATA_LOADER.SEQ_PER_IMG), reduction='sum')
            
            o_mil_loss, o_mil_info = self.mil_criterion(
                losses['o_' + cfg.LOSSES.GV_SUP.LOSS_NAME], 
                utils.expand_tensor(kwargs[cfg.PARAM.MIL_LABEL], cfg.DATA_LOADER.SEQ_PER_IMG)
            )
            loss = loss + cfg.LOSSES.GV_SUP.LOSS_WEIGHT * o_mil_loss
            for key in o_mil_info:
                loss_info['obj_' + key] = o_mil_info[key]

        if cfg.LOSSES.GV_SUP.LOSS_NAME in losses:
            #loss_sup = nn.MultiLabelMarginLoss()(
            #    losses[cfg.LOSSES.GV_SUP.LOSS_NAME], 
            #    utils.expand_tensor(kwargs[cfg.PARAM.MIL_LABEL], cfg.DATA_LOADER.SEQ_PER_IMG))
            #loss_sup = nn.MultiLabelSoftMarginLoss()(
            #    losses[cfg.LOSSES.GV_SUP.LOSS_NAME], 
            #    utils.expand_tensor(kwargs[cfg.PARAM.MIL_LABEL], cfg.DATA_LOADER.SEQ_PER_IMG), reduction='sum')
            mil_loss, mil_info = self.mil_criterion(
                losses[cfg.LOSSES.GV_SUP.LOSS_NAME], 
                utils.expand_tensor(kwargs[cfg.PARAM.MIL_LABEL], cfg.DATA_LOADER.SEQ_PER_IMG)
            )
            loss = loss + cfg.LOSSES.GV_SUP.LOSS_WEIGHT * mil_loss
            for key in mil_info:
                loss_info[key] = mil_info[key]

        if cfg.PARAM.OBJ_XE in losses:
            obj_xe_loss, obj_xe_loss_info = self.xe_criterion(losses[cfg.PARAM.OBJ_XE], kwargs[cfg.PARAM.TARGET_SENT])
            loss = loss + obj_xe_loss
            for key in obj_xe_loss_info:
                loss_info['o_' + key] = obj_xe_loss_info[key]
            
        return loss, loss_info


    def forward(self, kwargs):
        if self.rl_stage == False:
            losses = self.cap_model(self.obj_model, **kwargs)
            loss, loss_info = self.calculate_loss(losses, **kwargs)
        else:
            ids = kwargs[cfg.PARAM.INDICES]
            gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

            # max
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = True
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            self.model.eval()
            with torch.no_grad():
                seq_max, logP_max = self.model.module.decode(**kwargs)
            self.model.train()
            rewards_max, rewards_info_max = self.scorer(ids, seq_max.data.cpu().numpy().tolist())
            rewards_max = utils.expand_numpy(rewards_max)

            ids = utils.expand_numpy(ids)
            gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)

            # sample
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = False
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            #seq_sample, logP_sample = self.model.module.decode(**kwargs)
            if cfg.MODEL.TYPE == 'Btoformer':
                seq_sample, logP_sample, obj_losses = self.model(is_decode=True, **kwargs)
                nce_loss, nce_loss_info = self.nce_criterion(
                    obj_losses[cfg.LOSSES.NCE.LOSS_NAME],
                    kwargs[cfg.PARAM.OBJ_POS_LABEL],
                    kwargs[cfg.PARAM.OBJ_MLABEL],
                    kwargs[cfg.PARAM.ATT_FEATS_MASK],
                    kwargs[cfg.PARAM.OBJ_FEATS_MASK],
                )
                
            else: 
                seq_sample, logP_sample = self.model(is_decode=True, **kwargs)
            rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist())

            rewards = rewards_sample - rewards_max
            rewards = torch.from_numpy(rewards).float().cuda()
            loss = self.rl_criterion(seq_sample, logP_sample, rewards)
            
            loss_info = {}
            for key in rewards_info_sample:
                loss_info[key + '_sample'] = rewards_info_sample[key]
            for key in rewards_info_max:
                loss_info[key + '_max'] = rewards_info_max[key]

            if cfg.MODEL.TYPE == 'Btoformer':
                for key in nce_loss_info:
                    loss_info[key] = nce_loss_info[key]
                loss += nce_loss * cfg.LOSSES.NCE.LOSS_WEIGHT

        return loss, loss_info

    def visualize(self, input_seq, target_seq, gv_feat, att_feats, att_mask, 
        region_labels, obj_seqs, obj_pos_labels, obj_mlabels, obj_att_mask, tokens2rid):
        vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)
        obj_vocab = utils.load_lines(cfg.DATA_LOADER.OBJ_VOCAB_PATH)

        obj_seqs = obj_seqs.view(-1, obj_seqs.shape[-1])
        obj_att_mask = obj_att_mask.view(-1, obj_att_mask.shape[-1])[:, 1:]
        region_labels = region_labels.unsqueeze(1).expand(region_labels.shape[0], 5, region_labels.shape[-2], region_labels.shape[-1])
        region_labels = region_labels.reshape(-1, region_labels.shape[-2], region_labels.shape[-1])
        obj_pos_labels = obj_pos_labels.view(-1, obj_pos_labels.shape[-2], obj_pos_labels.shape[-1])[:, :-1, 1:]

        batch_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        obj_seq_len = obj_seqs.shape[1]

        for b in range(batch_size):
            ########## print sent #############
            words = []
            for j in range(seq_len):
                if target_seq[b, j] == 0:
                    break
                if j > 0:
                    assert input_seq[b, j] == target_seq[b, j-1] 
                words.append(vocab[target_seq[b, j]])
            sent = ' '.join(words)
            print(sent)

            ######### print obj_seqs #############
            obj_seq = []
            for j in range(obj_seq_len):
                rid = obj_seqs[b, j]
                if obj_att_mask[b, j] == 0:
                    break
                assert obj_pos_labels[b, j, rid] == 1
                rid_info = region_labels[b, rid]
                assert rid_info[0] == -1
                obj_labels = []
                for k in range(len(rid_info)):
                    if rid_info[k] == 1:
                        obj_labels.append(obj_vocab[k-1])
                obj_labels = ','.join(obj_labels)
                obj_seq.append(obj_labels)
            obj_seq = ' '.join(obj_seq)
            print(obj_seq)

    def train(self):
        self.cap_model.train()
        self.obj_model.train()

        iteration = 0
        train_obj = False
        train_cap = False

        for epoch in  range(cfg.SOLVER.MAX_EPOCH + cfg.SOLVER.OBJ_PRETRAIN_EPOCH + cfg.SOLVER.CAP_PRETRAIN_EPOCH + cfg.SOLVER.CONST_EPOCH):
            if epoch == cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            self.setup_loader(epoch)
            step_per_epoch = len(self.training_loader)

            if epoch < cfg.SOLVER.OBJ_PRETRAIN_EPOCH:
                train_obj = True
                train_cap = False
                self.obj_model.train()
                self.cap_model.train()
                
            elif epoch >= cfg.SOLVER.OBJ_PRETRAIN_EPOCH and epoch < (cfg.SOLVER.CAP_PRETRAIN_EPOCH + cfg.SOLVER.OBJ_PRETRAIN_EPOCH):
                train_obj = False
                train_cap = True
                self.obj_model.train()
                self.cap_model.train()
                     
            elif epoch >= cfg.SOLVER.CAP_PRETRAIN_EPOCH + cfg.SOLVER.OBJ_PRETRAIN_EPOCH:
                train_obj = True
                train_cap = True
                self.obj_model.train()
                self.cap_model.train()
                
            if self.cap_optim is None:
                self.cap_optim = Optimizer(
                    self.cap_model,
                    base_lr=cfg.SOLVER.CAP_BASE_LR, 
                    min_lr=cfg.SOLVER.MIN_LR,
                    step_per_epoch=step_per_epoch, 
                    warmup_epoch=cfg.SOLVER.CAP_LR_POLICY.WARMUP_EPOCH, 
                    max_epoch=cfg.SOLVER.CAP_PRETRAIN_EPOCH + cfg.SOLVER.MAX_EPOCH,
                    lr_policy=cfg.SOLVER.CAP_LR_POLICY
                )
            if self.obj_optim is None:
                self.obj_optim = Optimizer(
                    self.obj_model,
                    base_lr=cfg.SOLVER.OBJ_BASE_LR, 
                    min_lr=cfg.SOLVER.MIN_LR,
                    step_per_epoch=step_per_epoch, 
                    warmup_epoch=cfg.SOLVER.OBJ_LR_POLICY.WARMUP_EPOCH, 
                    max_epoch=cfg.SOLVER.OBJ_PRETRAIN_EPOCH + cfg.SOLVER.MAX_EPOCH,
                    lr_policy=cfg.SOLVER.OBJ_LR_POLICY
                )
            
            self.cap_optim.zero_grad()
            self.obj_optim.zero_grad()

            start = time.time()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            losses = AverageMeter()
            for _, (indices, input_seq, target_seq, gv_feat, att_feats, att_mask, 
                region_labels, obj_seqs, obj_pos_labels, obj_mlabels, obj_att_mask, 
                tokens2rid, mil_labels, obj_seq_targets) in enumerate(self.training_loader):

                #self.visualize(input_seq, target_seq, gv_feat, att_feats, att_mask, 
                #   region_labels, obj_seqs, obj_pos_labels, obj_mlabels, obj_att_mask, tokens2rid)

                data_time.update(time.time() - start)

                input_seq = input_seq.cuda()
                target_seq = target_seq.cuda()
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()
                region_labels = region_labels.cuda()
                obj_seqs = obj_seqs.cuda()
                obj_pos_labels = obj_pos_labels.cuda()
                obj_mlabels = obj_mlabels.cuda()
                obj_att_mask = obj_att_mask.cuda()
                tokens2rid = tokens2rid.cuda()
                mil_labels = mil_labels.cuda()
                mil_labels = mil_labels.type(torch.cuda.LongTensor)
                obj_seq_targets = obj_seq_targets.cuda().view(-1, obj_seq_targets.shape[-2], obj_seq_targets.shape[-1])

                kwargs = self.make_kwargs(
                    indices, 
                    input_seq, 
                    target_seq, 
                    gv_feat, 
                    att_feats, 
                    att_mask,
                    region_labels,
                    obj_seqs,
                    obj_pos_labels,
                    obj_mlabels,
                    obj_att_mask,
                    tokens2rid,
                    mil_labels,
                    obj_seq_targets
                )
                #loss, loss_info = self.forward(kwargs)
                losses_dict = self.cap_model(train_obj, train_cap, self.obj_model, **kwargs)
                loss, loss_info = self.calculate_loss(losses_dict, **kwargs)
                loss.backward()

                if train_cap:
                    utils.clip_gradient(self.cap_optim.optimizer, self.cap_model,
                        cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                    self.cap_optim.step()
                    self.cap_optim.zero_grad()
                    self.cap_optim.scheduler_step('Iter')

                if train_obj:
                    utils.clip_gradient(self.obj_optim.optimizer, self.obj_model,
                        cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                    self.obj_optim.step()
                    self.obj_optim.zero_grad()
                    self.obj_optim.scheduler_step('Iter')

                batch_time.update(time.time() - start)
                start = time.time()
                losses.update(loss.item())
                self.display(iteration, data_time, batch_time, losses, loss_info)
                iteration += 1

                if self.distributed:
                    dist.barrier()

                #if (iteration+1) % 30 == 0:
                #   break
                
            if epoch >= cfg.SOLVER.CAP_PRETRAIN_EPOCH + cfg.SOLVER.OBJ_PRETRAIN_EPOCH:
                self.save_model(epoch)

            val = self.eval(epoch)
                
            if train_cap:
                self.cap_optim.scheduler_step('Epoch', val)
                self.scheduled_cap_sampling(epoch - cfg.SOLVER.OBJ_PRETRAIN_EPOCH)
            if train_obj:
                self.obj_optim.scheduler_step('Epoch', val)
                if epoch >= cfg.SOLVER.CAP_PRETRAIN_EPOCH + cfg.SOLVER.OBJ_PRETRAIN_EPOCH:
                   self.scheduled_obj_sampling(epoch - cfg.SOLVER.CAP_PRETRAIN_EPOCH)
                else:
                   self.scheduled_obj_sampling(epoch)
               
            if self.distributed:
                dist.barrier()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    trainer = Trainer(args)
    trainer.train()
