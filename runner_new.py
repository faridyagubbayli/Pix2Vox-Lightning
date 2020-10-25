#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Farid Yagubbayli <faridyagubbayli@gmail.com>
# based on implementation provided by Haozhe Xie <cshzxie@gmail.com>


from utils.data_loaders import ShapeNetDataModule
from models.model import Model
from core.test import test_net
from datetime import datetime as dt
import logging
import matplotlib
import multiprocessing as mp
import os
import sys
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')


def save_cfg(cfg, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, 'config.yaml')
    OmegaConf.save(cfg, path)


def main():
    cfg = OmegaConf.load('conf/config.yaml')

    if cfg.seed != -1:
        pl.seed_everything(cfg.seed)

    # Start train/test process
    if not cfg.is_test:
        model = Model(cfg.network, cfg.tester)
        data_module = ShapeNetDataModule(cfg.data)

        logger = pl.loggers.TensorBoardLogger("tb_logs", name="pix2vox")
        save_cfg(cfg, logger.log_dir)

        trainer = pl.Trainer(automatic_optimization=False, log_every_n_steps=1, logger=logger, **cfg.trainer)
        trainer.fit(model, data_module)
        trainer.save_checkpoint('saved_model.ckpt')
    else:
        pass


if __name__ == '__main__':
    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    main()
