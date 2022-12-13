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

class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        self.device = torch.device("cuda")

        self.setup_logging()
        self.setup_network()
        self.evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            gv_feat = cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            obj_vocab_path = cfg.DATA_LOADER.OBJ_VOCAB_PATH,
            region_info_path = cfg.DATA_LOADER.REGION_INFO_PATH,
            token_info_path = cfg.DATA_LOADER.TOKEN_INFO_TEST_PATH,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE
        )

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        
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

    def setup_network(self):
        cap_model = models.create(cfg.MODEL.TYPE)
        obj_model = models.create('Objformer')
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
        
    def eval(self, epoch):
        res = self.evaler(self.cap_model, self.obj_model, 'test_' + str(epoch))
        self.logger.info('######## Epoch ' + str(epoch) + ' ########')
        self.logger.info(str(res))

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', default=None, type=str)
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

    tester = Tester(args)
    tester.eval(args.resume)
