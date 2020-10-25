from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

import torch.utils.data
from omegaconf import DictConfig

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

from datetime import datetime as dt
import json


class Model(pl.LightningModule):

    def __init__(self, cfg_network: DictConfig, cfg_tester: DictConfig):
        super().__init__()
        self.cfg_network = cfg_network
        self.cfg_tester = cfg_tester

        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Set up networks
        self.encoder = Encoder(cfg_network)
        self.decoder = Decoder(cfg_network)
        self.refiner = Refiner(cfg_network)
        self.merger = Merger(cfg_network)
        
        # Initialize weights of networks
        self.encoder.apply(utils.network_utils.init_weights)
        self.decoder.apply(utils.network_utils.init_weights)
        self.refiner.apply(utils.network_utils.init_weights)
        self.merger.apply(utils.network_utils.init_weights)
        
        self.bce_loss = nn.BCELoss()

    def configure_optimizers(self):
        params = self.cfg_network.optimization
        # Set up solver
        if params.policy == 'adam':
            encoder_solver = optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                            lr=params.encoder_lr,
                                            betas=params.betas)
            decoder_solver = optim.Adam(self.decoder.parameters(),
                                            lr=params.decoder_lr,
                                            betas=params.betas)
            refiner_solver = optim.Adam(self.refiner.parameters(),
                                            lr=params.refiner_lr,
                                            betas=params.betas)
            merger_solver = optim.Adam(self.merger.parameters(),
                                             lr=params.merger_lr,
                                             betas=params.betas)
        elif params.policy == 'sgd':
            encoder_solver = optim.SGD(filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                            lr=params.encoder_lr,
                                            momentum=params.momentum)
            decoder_solver = optim.SGD(self.decoder.parameters(),
                                            lr=params.decoder_lr,
                                            momentum=params.momentum)
            refiner_solver = optim.SGD(self.refiner.parameters(),
                                            lr=params.refiner_lr,
                                            momentum=params.momentum)
            merger_solver = optim.SGD(self.merger.parameters(),
                                            lr=params.merger_lr,
                                            momentum=params.momentum)
        else:
            raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), params.policy))
            
            # Set up learning rate scheduler to decay learning rates dynamically
        encoder_lr_scheduler = optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                    milestones=params.encoder_lr_milestones,
                                                                    gamma=params.gamma)
        decoder_lr_scheduler = optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                    milestones=params.decoder_lr_milestones,
                                                                    gamma=params.gamma)
        refiner_lr_scheduler = optim.lr_scheduler.MultiStepLR(refiner_solver,
                                                                    milestones=params.refiner_lr_milestones,
                                                                    gamma=params.gamma)
        merger_lr_scheduler = optim.lr_scheduler.MultiStepLR(merger_solver,
                                                                milestones=params.merger_lr_milestones,
                                                                gamma=params.gamma)
        
        return [encoder_solver, decoder_solver, refiner_solver, merger_solver], \
               [encoder_lr_scheduler, decoder_lr_scheduler, refiner_lr_scheduler, merger_lr_scheduler]
    
    def _fwd(self, batch):
        taxonomy_names, sample_names, rendering_images, ground_truth_volumes = batch

        image_features = self.encoder(rendering_images)
        raw_features, generated_volumes = self.decoder(image_features)

        if self.cfg_network.use_merger and self.current_epoch >= self.cfg_network.optimization.epoch_start_use_merger:
            generated_volumes = self.merger(raw_features, generated_volumes)
        else:
            generated_volumes = torch.mean(generated_volumes, dim=1)
        encoder_loss = self.bce_loss(generated_volumes, ground_truth_volumes) * 10
        
        if self.cfg_network.use_refiner and self.current_epoch >= self.cfg_network.optimization.epoch_start_use_refiner:
            generated_volumes = self.refiner(generated_volumes)
            refiner_loss = self.bce_loss(generated_volumes, ground_truth_volumes) * 10
        else:
            refiner_loss = encoder_loss
        
        return generated_volumes, encoder_loss, refiner_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        (opt_enc, opt_dec, opt_ref, opt_merg) = self.optimizers()
        
        generated_volumes, encoder_loss, refiner_loss = self._fwd(batch)
        
        self.log('loss/EncoderDecoder', encoder_loss, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('loss/Refiner', refiner_loss, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.cfg_network.use_refiner and self.current_epoch >= self.cfg_network.optimization.epoch_start_use_refiner:
            self.manual_backward(encoder_loss, opt_enc, retain_graph=True)
            self.manual_backward(refiner_loss, opt_ref)
        else:
            self.manual_backward(encoder_loss, opt_enc)
            
        for opt in self.optimizers():
            opt.step()
            opt.zero_grad()

    def training_epoch_end(self, outputs) -> None:
        # Update Rendering Views
        if self.cfg_network.update_n_views_rendering:
            n_views_rendering = self.trainer.datamodule.update_n_views_rendering()
            print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' %
                  (dt.now(), self.current_epoch + 2, self.trainer.max_epochs, n_views_rendering))

    def _eval_step(self, batch, batch_idx):
        # SUPPORTS ONLY BATCH_SIZE=1
        taxonomy_names, sample_names, rendering_images, ground_truth_volumes = batch
        taxonomy_id = taxonomy_names[0]
        sample_name = sample_names[0]

        generated_volumes, encoder_loss, refiner_loss = self._fwd(batch)

        self.log('val_loss/EncoderDecoder', encoder_loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)

        self.log('val_loss/Refiner', refiner_loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)

        # IoU per sample
        sample_iou = []
        for th in self.cfg_tester.voxel_thresh:
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
        # Load taxonomies of dataset
        taxonomies = []
        taxonomy_path = self.trainer.datamodule.get_test_taxonomy_file_path()
        with open(taxonomy_path, encoding='utf-8') as file:
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
        for th in self.cfg_tester.voxel_thresh:
            print('t=%.2f' % th, end='\t')
        print()
        # Print body
        for taxonomy_id in test_iou:
            print('%s' % taxonomies[taxonomy_id]
                  ['taxonomy_name'].ljust(8), end='\t')
            print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
            if 'baseline' in taxonomies[taxonomy_id]:
                n_views_rendering = self.trainer.datamodule.get_n_views_rendering()
                print('%.4f' % taxonomies[taxonomy_id]['baseline']
                      ['%d-view' % n_views_rendering], end='\t\t')
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
