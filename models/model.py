import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

import torch.utils.data

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time
import json


class Model(pl.LightningModule):


    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Set up networks
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.refiner = Refiner(cfg)
        self.merger = Merger(cfg)
        print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), utils.network_utils.count_parameters(self.encoder)))
        print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(self.decoder)))
        print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(self.refiner)))
        print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(self.merger)))
        
        
        # Initialize weights of networks
        self.encoder.apply(utils.network_utils.init_weights)
        self.decoder.apply(utils.network_utils.init_weights)
        self.refiner.apply(utils.network_utils.init_weights)
        self.merger.apply(utils.network_utils.init_weights)
        
        self.bce_loss = nn.BCELoss()
        
        
    def configure_optimizers(self):
        cfg = self.cfg
        # Set up solver
        if cfg.TRAIN.POLICY == 'adam':
            encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                            lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                            betas=cfg.TRAIN.BETAS)
            decoder_solver = torch.optim.Adam(self.decoder.parameters(),
                                            lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                            betas=cfg.TRAIN.BETAS)
            refiner_solver = torch.optim.Adam(self.refiner.parameters(),
                                            lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                            betas=cfg.TRAIN.BETAS)
            merger_solver = torch.optim.Adam(self.merger.parameters(), 
                                             lr=cfg.TRAIN.MERGER_LEARNING_RATE, 
                                             betas=cfg.TRAIN.BETAS)
        elif cfg.TRAIN.POLICY == 'sgd':
            encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                            lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                            momentum=cfg.TRAIN.MOMENTUM)
            decoder_solver = torch.optim.SGD(self.decoder.parameters(),
                                            lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                            momentum=cfg.TRAIN.MOMENTUM)
            refiner_solver = torch.optim.SGD(self.refiner.parameters(),
                                            lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                            momentum=cfg.TRAIN.MOMENTUM)
            merger_solver = torch.optim.SGD(self.merger.parameters(),
                                            lr=cfg.TRAIN.MERGER_LEARNING_RATE,
                                            momentum=cfg.TRAIN.MOMENTUM)
        else:
            raise Exception('[FATAL] %s Unknown optimizer %s.' %
                            (dt.now(), cfg.TRAIN.POLICY))
            
            # Set up learning rate scheduler to decay learning rates dynamically
        encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                    milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                    gamma=cfg.TRAIN.GAMMA)
        decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                    milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                    gamma=cfg.TRAIN.GAMMA)
        refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver,
                                                                    milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                    gamma=cfg.TRAIN.GAMMA)
        merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(merger_solver,
                                                                milestones=cfg.TRAIN.MERGER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
        
        return [encoder_solver, decoder_solver, refiner_solver, merger_solver], [encoder_lr_scheduler, decoder_lr_scheduler, refiner_lr_scheduler, merger_lr_scheduler]
    
    def fwd(self, batch):
        cfg = self.cfg
        taxonomy_names, sample_names, rendering_images, ground_truth_volumes = batch

        image_features = self.encoder(rendering_images)
        raw_features, generated_volumes = self.decoder(image_features)

        if cfg.NETWORK.USE_MERGER and self.current_epoch >= cfg.TRAIN.EPOCH_START_USE_MERGER:
            generated_volumes = self.merger(raw_features, generated_volumes)
        else:
            generated_volumes = torch.mean(generated_volumes, dim=1)
        encoder_loss = self.bce_loss(generated_volumes, ground_truth_volumes) * 10
        
        if cfg.NETWORK.USE_REFINER and self.current_epoch >= cfg.TRAIN.EPOCH_START_USE_REFINER:
            generated_volumes = self.refiner(generated_volumes)
            refiner_loss = self.bce_loss(generated_volumes, ground_truth_volumes) * 10
        else:
            refiner_loss = encoder_loss
        
        return generated_volumes, encoder_loss, refiner_loss
    
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        cfg = self.cfg
        (opt_enc, opt_dec, opt_ref, opt_merg) = self.optimizers()
        
        generated_volumes, encoder_loss, refiner_loss = self.fwd(batch)
        
        self.log('loss/EncoderDecoder', encoder_loss, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('loss/Refiner', refiner_loss, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        if cfg.NETWORK.USE_REFINER and self.current_epoch >= cfg.TRAIN.EPOCH_START_USE_REFINER:
            self.manual_backward(encoder_loss, opt_enc, retain_graph=True)
            self.manual_backward(refiner_loss, opt_ref)
        else:
            self.manual_backward(encoder_loss, opt_enc)
            
        for opt in self.optimizers():
            opt.step()
            opt.zero_grad()

    def _eval_step(self, batch, batch_idx):
        # SUPPORTS ONLY BATCH_SIZE=1
        cfg = self.cfg
        taxonomy_names, sample_names, rendering_images, ground_truth_volumes = batch
        taxonomy_id = taxonomy_names[0]
        sample_name = sample_names[0]

        generated_volumes, encoder_loss, refiner_loss = self.fwd(batch)

        self.log('val_loss/EncoderDecoder', encoder_loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)

        self.log('val_loss/Refiner', refiner_loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)

        # IoU per sample
        sample_iou = []
        for th in cfg.TEST.VOXEL_THRESH:
            _volume = torch.ge(generated_volumes, th).float()
            intersection = torch.sum(_volume.mul(ground_truth_volumes)).float()
            union = torch.sum(
                torch.ge(_volume.add(ground_truth_volumes), 1)).float()
            sample_iou.append((intersection / union).item())

        # Print sample loss and IoU
        n_samples = -1
        print('\n[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
              (dt.now(), batch_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
               refiner_loss.item(), ['%.4f' % si for si in sample_iou]))

        return {
            'taxonomy_id': taxonomy_id,
            'sample_name': sample_name,
            'sample_iou': sample_iou
        }
        
    def _eval_epoch_end(self, outputs):
        cfg = self.cfg

        # Load taxonomies of dataset
        taxonomies = []
        with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            taxonomies = json.loads(file.read())
        taxonomies = {t['taxonomy_id']: t for t in taxonomies}

        test_iou = {}
        for output in outputs:
            taxonomy_id, sample_name, sample_iou = output[
                'taxonomy_id'], output['sample_name'], output['sample_iou']
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

        mean_iou = []
        for taxonomy_id in test_iou:
            test_iou[taxonomy_id]['iou'] = torch.mean(
                torch.tensor(test_iou[taxonomy_id]['iou']), dim=0)
            mean_iou.append(test_iou[taxonomy_id]['iou']
                            * test_iou[taxonomy_id]['n_samples'])
        n_samples = len(outputs)
        mean_iou = torch.stack(mean_iou)
        mean_iou = torch.sum(mean_iou, dim=0) / n_samples

        # Print header
        print('============================ TEST RESULTS ============================')
        print('Taxonomy', end='\t')
        print('#Sample', end='\t')
        print(' Baseline', end='\t')
        for th in cfg.TEST.VOXEL_THRESH:
            print('t=%.2f' % th, end='\t')
        print()
        # Print body
        for taxonomy_id in test_iou:
            print('%s' % taxonomies[taxonomy_id]
                  ['taxonomy_name'].ljust(8), end='\t')
            print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
            if 'baseline' in taxonomies[taxonomy_id]:
                print('%.4f' % taxonomies[taxonomy_id]['baseline']
                      ['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
            else:
                print('N/a', end='\t\t')

            for ti in test_iou[taxonomy_id]['iou']:
                print('%.4f' % ti, end='\t')
            print()
        # Print mean IoU for each threshold
        print('Overall ', end='\t\t\t\t')
        for mi in mean_iou:
            print('%.4f' % mi, end='\t')
        print('\n')

        max_iou = torch.max(mean_iou)
        self.log('Refiner/IoU', max_iou, prog_bar=True, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)
        
    def validation_epoch_end(self, outputs):
        self._eval_epoch_end(outputs)
        
    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)
        
    def test_epoch_end(self, outputs):
        self._eval_epoch_end(outputs)

    def get_progress_bar_dict(self):
        # don't show the loss as it's None
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
