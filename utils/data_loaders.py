# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
from argparse import ArgumentParser, Namespace

import cv2
import json
import numpy as np
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset

from datetime import datetime as dt
from enum import Enum, unique

from omegaconf import DictConfig

import utils.binvox_rw

import pytorch_lightning as pl


@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2
    

class ShapeNetDataModule(pl.LightningDataModule):
    
    def __init__(self, cfg_data: DictConfig):
        super().__init__()
        self.cfg_data = cfg_data
        self.train_data_loader = None

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        const = self.cfg_data.constants
        c_dt = self.cfg_data.transforms
        c_dl = self.cfg_data.loader

        # Set up data augmentation
        IMG_SIZE = const.img_h, const.img_w
        CROP_SIZE = const.crop_img_h, const.crop_img_w
        train_transforms = utils.data_transforms.Compose([
            utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(c_dt.train_rand_bg_color_range),
            utils.data_transforms.ColorJitter(c_dt.brightness, c_dt.contrast, c_dt.saturation),
            utils.data_transforms.RandomNoise(c_dt.noise_std),
            utils.data_transforms.Normalize(c_dt.mean, std=c_dt.std),
            utils.data_transforms.RandomFlip(),
            utils.data_transforms.RandomPermuteRGB(),
            utils.data_transforms.ToTensor(),
        ])

        ds_config = self.cfg_data.dataset[c_dl.train_dataset]
        train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[c_dl.train_dataset](ds_config)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset_loader.get_dataset(
                utils.data_loaders.DatasetType.TRAIN, const.n_views_rendering, train_transforms),
                batch_size=c_dl.batch_size,
                num_workers=c_dl.num_workers,
                pin_memory=True,
                shuffle=True,
                drop_last=True)
        self.train_data_loader = train_data_loader
        return train_data_loader
    
    def val_dataloader(self):
        const = self.cfg_data.constants
        c_dt = self.cfg_data.transforms
        c_dl = self.cfg_data.loader

        # Set up data augmentation
        IMG_SIZE = const.img_h, const.img_w
        CROP_SIZE = const.crop_img_h, const.crop_img_w

        ds_config = self.cfg_data.dataset[c_dl.test_dataset]
        val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[c_dl.test_dataset](ds_config)

        val_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(c_dt.test_rand_bg_color_range),
            utils.data_transforms.Normalize(mean=c_dt.mean, std=c_dt.std),
            utils.data_transforms.ToTensor(),
        ])
        
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.VAL, const.n_views_rendering, val_transforms),
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False)
        
        return val_data_loader

    def test_dataloader(self):
        const = self.cfg_data.constants
        c_dt = self.cfg_data.transforms
        c_dl = self.cfg_data.loader

        # Set up data augmentation
        IMG_SIZE = const.img_h, const.img_w
        CROP_SIZE = const.crop_img_h, const.crop_img_w

        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(
                c_dt.test_rand_bg_color_range),
            utils.data_transforms.Normalize(
                mean=c_dt.mean, std=c_dt.std),
            utils.data_transforms.ToTensor(),
        ])

        ds_config = self.cfg_data.dataset[c_dl.test_dataset]
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[c_dl.test_dataset](ds_config)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, const.n_views_rendering, test_transforms),
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False)
        
        return test_data_loader

    def get_test_taxonomy_file_path(self):
        ds_name = self.cfg_data.loader.test_dataset
        return self.cfg_data.dataset[ds_name].taxonomy_path

    def get_n_views_rendering(self):
        return self.cfg_data.constants.n_views_rendering

    def update_n_views_rendering(self):
        n_views_rendering = random.randint(1, self.get_n_views_rendering())
        self.train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
        return n_views_rendering
# //////////////////////////////// = End of DatasetType Class Definition = ///////////////////////////////// #


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, n_views_rendering, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxonomy_name, sample_name, rendering_images, volume

    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        if self.dataset_type == DatasetType.TRAIN:
            selected_rendering_image_paths = [
                rendering_image_paths[i]
                for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering)
            ]
        else:
            selected_rendering_image_paths = [rendering_image_paths[i] for i in range(self.n_views_rendering)]

        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                      (dt.now(), image_path))
                sys.exit(2)

            rendering_images.append(rendering_image)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)

        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

        return taxonomy_name, sample_name, np.asarray(rendering_images), volume


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class ShapeNetDataLoader:
    def __init__(self, cfg: DictConfig):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = cfg.rendering_path
        self.volume_path_template = cfg.voxel_path

        # Load all taxonomies of the dataset
        with open(cfg.taxonomy_path, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_folder_name = taxonomy['taxonomy_id']
            print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' %
                  (dt.now(), taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['val']

            files.extend(self.get_files_of_taxonomy(taxonomy_folder_name, samples))

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return ShapeNetDataset(dataset_type, files, n_views_rendering, transforms)

    def get_files_of_taxonomy(self, taxonomy_folder_name, samples):
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(volume_file_path):
                print('[WARN] %s Ignore sample %s/%s since volume file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get file list of rendering images
            img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
            img_folder = os.path.dirname(img_file_path)
            total_views = len(os.listdir(img_folder))
            rendering_image_indexes = range(total_views)
            rendering_images_file_path = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue

                rendering_images_file_path.append(img_file_path)

            if len(rendering_images_file_path) == 0:
                print('[WARN] %s Ignore sample %s/%s since image files not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_images_file_path,
                'volume': volume_file_path,
            })

            # Report the progress of reading dataset
            # if sample_idx % 500 == 499 or sample_idx == n_samples - 1:
            #     print('[INFO] %s Collecting %d of %d' % (dt.now(), sample_idx + 1, n_samples))

        return files_of_taxonomy


# /////////////////////////////// = End of ShapeNetDataLoader Class Definition = /////////////////////////////// #


class Pascal3dDataset(torch.utils.data.dataset.Dataset):
    """Pascal3D class used for PyTorch DataLoader"""
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume, bounding_box = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images, bounding_box)

        return taxonomy_name, sample_name, rendering_images, volume

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_path = self.file_list[idx]['rendering_image']
        bounding_box = self.file_list[idx]['bounding_box']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        rendering_image = cv2.imread(rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if len(rendering_image.shape) < 3:
            print('[WARN] %s It seems the image file %s is grayscale.' % (dt.now(), rendering_image_path))
            rendering_image = np.stack((rendering_image, ) * 3, -1)

        # Get data of volume
        with open(volume_path, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            volume = volume.data.astype(np.float32)

        return taxonomy_name, sample_name, np.asarray([rendering_image]), volume, bounding_box


# //////////////////////////////// = End of Pascal3dDataset Class Definition = ///////////////////////////////// #


class Pascal3dDataLoader:
    def __init__(self, cfg: DictConfig):
        self.dataset_taxonomy = None
        self.volume_path_template = cfg.voxel_path
        self.annotation_path_template = cfg.annotation_path
        self.rendering_image_path_template = cfg.rendering_path

        # Load all taxonomies of the dataset
        with open(cfg.taxonomy_path, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_name = taxonomy['taxonomy_name']
            print('[INFO] %s Collecting files of Taxonomy[Name=%s]' % (dt.now(), taxonomy_name))

            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['test']

            files.extend(self.get_files_of_taxonomy(taxonomy_name, samples))

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return Pascal3dDataset(files, transforms)

    def get_files_of_taxonomy(self, taxonomy_name, samples):
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file list of rendering images
            rendering_image_file_path = self.rendering_image_path_template % (taxonomy_name, sample_name)
            # if not os.path.exists(rendering_image_file_path):
            #     continue

            # Get image annotations
            annotations_file_path = self.annotation_path_template % (taxonomy_name, sample_name)
            annotations_mat = scipy.io.loadmat(annotations_file_path, squeeze_me=True, struct_as_record=False)
            img_width, img_height, _ = annotations_mat['record'].imgsize
            annotations = annotations_mat['record'].objects

            cad_index = -1
            bbox = None
            if (type(annotations) == np.ndarray):
                max_bbox_aera = -1

                for i in range(len(annotations)):
                    _cad_index = annotations[i].cad_index
                    _bbox = annotations[i].__dict__['bbox']

                    bbox_xmin = _bbox[0]
                    bbox_ymin = _bbox[1]
                    bbox_xmax = _bbox[2]
                    bbox_ymax = _bbox[3]
                    _bbox_area = (bbox_xmax - bbox_xmin) * (bbox_ymax - bbox_ymin)

                    if _bbox_area > max_bbox_aera:
                        bbox = _bbox
                        cad_index = _cad_index
                        max_bbox_aera = _bbox_area
            else:
                cad_index = annotations.cad_index
                bbox = annotations.bbox

            # Convert the coordinates of bounding boxes to percentages
            bbox = [bbox[0] / img_width, bbox[1] / img_height, bbox[2] / img_width, bbox[3] / img_height]
            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_name, cad_index)
            if not os.path.exists(volume_file_path):
                print('[WARN] %s Ignore sample %s/%s since volume file not exists.' %
                      (dt.now(), taxonomy_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_name,
                'sample_name': sample_name,
                'rendering_image': rendering_image_file_path,
                'bounding_box': bbox,
                'volume': volume_file_path,
            })

        return files_of_taxonomy


# /////////////////////////////// = End of Pascal3dDataLoader Class Definition = /////////////////////////////// #


class Pix3dDataset(torch.utils.data.dataset.Dataset):
    """Pix3D class used for PyTorch DataLoader"""
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume, bounding_box = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images, bounding_box)

        return taxonomy_name, sample_name, rendering_images, volume

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_path = self.file_list[idx]['rendering_image']
        bounding_box = self.file_list[idx]['bounding_box']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        rendering_image = cv2.imread(rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if len(rendering_image.shape) < 3:
            print('[WARN] %s It seems the image file %s is grayscale.' % (dt.now(), rendering_image_path))
            rendering_image = np.stack((rendering_image, ) * 3, -1)

        # Get data of volume
        with open(volume_path, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            volume = volume.data.astype(np.float32)

        return taxonomy_name, sample_name, np.asarray([rendering_image]), volume, bounding_box


# //////////////////////////////// = End of Pascal3dDataset Class Definition = ///////////////////////////////// #


class Pix3dDataLoader:
    def __init__(self, cfg: DictConfig):
        self.dataset_taxonomy = None
        self.annotations = dict()
        self.volume_path_template = cfg.voxel_path
        self.rendering_image_path_template = cfg.rendering_path

        # Load all taxonomies of the dataset
        with open(cfg.taxonomy_path, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

        # Load all annotations of the dataset
        _annotations = None
        with open(cfg.annotation_path, encoding='utf-8') as file:
            _annotations = json.loads(file.read())

        for anno in _annotations:
            filename, _ = os.path.splitext(anno['img'])
            anno_key = filename[4:]
            self.annotations[anno_key] = anno

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_name = taxonomy['taxonomy_name']
            print('[INFO] %s Collecting files of Taxonomy[Name=%s]' % (dt.now(), taxonomy_name))

            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['test']

            files.extend(self.get_files_of_taxonomy(taxonomy_name, samples))

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return Pix3dDataset(files, transforms)

    def get_files_of_taxonomy(self, taxonomy_name, samples):
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # Get image annotations
            anno_key = '%s/%s' % (taxonomy_name, sample_name)
            annotations = self.annotations[anno_key]

            # Get file list of rendering images
            _, img_file_suffix = os.path.splitext(annotations['img'])
            rendering_image_file_path = self.rendering_image_path_template % (taxonomy_name, sample_name,
                                                                              img_file_suffix[1:])

            # Get the bounding box of the image
            img_width, img_height = annotations['img_size']
            bbox = [
                annotations['bbox'][0] / img_width,
                annotations['bbox'][1] / img_height,
                annotations['bbox'][2] / img_width,
                annotations['bbox'][3] / img_height
            ]  # yapf: disable
            model_name_parts = annotations['voxel'].split('/')
            model_name = model_name_parts[2]
            volume_file_name = model_name_parts[3][:-4].replace('voxel', 'model')

            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_name, model_name, volume_file_name)
            if not os.path.exists(volume_file_path):
                print('[WARN] %s Ignore sample %s/%s since volume file not exists.' %
                      (dt.now(), taxonomy_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_name,
                'sample_name': sample_name,
                'rendering_image': rendering_image_file_path,
                'bounding_box': bbox,
                'volume': volume_file_path,
            })

        return files_of_taxonomy


# /////////////////////////////// = End of Pascal3dDataLoader Class Definition = /////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
    'Pascal3D': Pascal3dDataLoader,
    'Pix3D': Pix3dDataLoader
}  # yapf: disable
