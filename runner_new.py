#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Farid Yagubbayli <faridyagubbayli@gmail.com>
# based on implementation provided by Haozhe Xie <cshzxie@gmail.com>


from utils.data_loaders import ShapeNetDataModule
from models.model import Model
from core.test import test_net
from core.train import train_net
from config import cfg
from pprint import pprint
from datetime import datetime as dt
from argparse import ArgumentParser
import logging
import matplotlib
import multiprocessing as mp
import numpy as np
import os
import sys
import pytorch_lightning as pl
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test', dest='test',
                        help='Test neural networks', action='store_true')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches',
                        default=cfg.TRAIN.NUM_EPOCHES, type=int)
    parser.add_argument('--weights', dest='weights',
                        help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path',
                        help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        pl.seed_everything(cfg.CONST.RNG_SEED)
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHES = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True

    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    if not args.test:
        model = Model(cfg)
        data_module = ShapeNetDataModule(cfg)

        logger = pl.loggers.TensorBoardLogger("tb_logs", name="my_model")

        trainer = pl.Trainer(gpus=1, automatic_optimization=False, max_epochs=cfg.TRAIN.NUM_EPOCHES,
                             log_every_n_steps=1, logger=logger)
        trainer.fit(model, data_module)
    else:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            test_net(cfg)
        else:
            print('[FATAL] %s Please specify the file path of checkpoint.' %
                  (dt.now()))
            sys.exit(2)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception(
            "Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    main()
