from __future__ import print_function, division
import os

import cv2
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange

def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return trnsfrms_val


class Whole_Slide_Bag(Dataset):
    def __init__(self,
                 file_path,
                 pretrained=False,
                 custom_transforms=None,
                 target_patch_size=-1,
                 ):
        """
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
        self.pretrained = pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        self.roi_transforms = eval_transforms(pretrained=True)

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]

        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
                 file_path,
                 wsi,
                 pretrained=False,
                 custom_transforms=None,
                 custom_downsample=1,
                 target_patch_size=-1
                 ):
        """
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
        self.pretrained = pretrained
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)  # 1、toTensor 2、归一化mean/std
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords'][:]
            self.coords = dset
            self.length = len(dset)
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size,) * 2
        elif custom_downsample > 1:
            self.target_patch_size = (self.patch_size // custom_downsample,) * 2
        else:
            self.target_patch_size = None

    # self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        dset = self.coords
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = self.wsi.read_region(coord, self.patch_level, (256, 256)).convert('RGB')
        # img = np.array(img)
        # (r, g, b) = cv2.split(img)  # 通道分解
        # bH = cv2.equalizeHist(b)
        # gH = cv2.equalizeHist(g)
        # rH = cv2.equalizeHist(r)
        # img = cv2.merge((bH, gH, rH), )  # 通道合成
        # img = Image.fromarray(img).convert('RGB')
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


class Dataset_All_Bags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # self.df = os.listdir(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]
        # return self.df[idx]




