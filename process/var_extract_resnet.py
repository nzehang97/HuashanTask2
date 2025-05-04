import argparse
import os
import cv2
import h5py
import openslide
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.transforms import transforms
from skimage.feature import graycomatrix, graycoprops

def is_artifact(features, features1, features2, features3):
    # 这里可以训练SVM等模型
    return features[0][0] > 0.3 or features1[0][0] > 0.3 or features2[0][0] > 0.3 or features3[0][0] > 0.3


def is_weiying(features, features1, features2, features3):
    # 这里可以训练SVM等模型
    threshold = 150
    a = int(features[0][0] < threshold)
    b = int(features1[0][0] < threshold)
    c = int(features2[0][0] < threshold)
    d = int(features3[0][0] < threshold)
    if a + b + c + d >= 1:
        move = True
    else:
        move = False
    return move


def get_patch_var1(coords, src_path, patch_level=0, patch_size=512):
    index = []
    remove_num = 0
    wsi = openslide.open_slide(src_path)
    for i in range(len(coords)):
        if i % 10 == 0:
            print(f'\rvaring: {i}/{len(coords)}', end='')
        try:
            img = wsi.read_region(coords[i], patch_level, (patch_size, patch_size)).convert('RGB')
        except:
            return 0, True
        img = img.resize([224, 224])
        img = np.array(img)
        gray = img[:, :, 1]
        glcm = graycomatrix(gray, [3], [0])
        glcm1 = graycomatrix(gray, [3], [45])
        glcm2 = graycomatrix(gray, [3], [90])
        glcm3 = graycomatrix(gray, [3], [135])
        # features = graycoprops(glcm, 'homogeneity')
        # features1 = graycoprops(glcm1, 'homogeneity')
        # features2 = graycoprops(glcm2, 'homogeneity')
        # features3 = graycoprops(glcm3, 'homogeneity')

        contrast = graycoprops(glcm, 'contrast')
        contrast1 = graycoprops(glcm1, 'contrast')
        contrast2 = graycoprops(glcm2, 'contrast')
        contrast3 = graycoprops(glcm3, 'contrast')

        vis0 = img[:, :, 0]
        vis1 = img[:, :, 1]
        vis2 = img[:, :, 2]

        mse1 = np.mean((vis0 - vis1) ** 2)
        mse2 = np.mean((vis0 - vis2) ** 2)
        mse3 = np.mean((vis1 - vis2) ** 2)
        mse = (mse1 + mse2 + mse3) / 3

        mean_vis0 = np.mean(vis0)
        mean_vis1 = np.mean(vis1)
        mean_vis2 = np.mean(vis2)

        if is_weiying(contrast, contrast1, contrast2, contrast3) or mse <= 40 or (
                mean_vis1 > mean_vis2 or mean_vis1 > mean_vis0):
            remove_num += 1
        else:
            index.append(i)
            # img.convert('RGB').save(f'../vis/{slide_dir}/{coords[i][0]}_{coords[i][1]}.jpeg')
    print(f'\nremove {remove_num} patches')
    return index


def var_index(args):

    rootpath = args.patches_h5_dir
    source_dir = args.source
    os.makedirs(args.patch_index_dir, exist_ok=True)
    existed = os.listdir(args.patch_index_dir)
    existed = [e.split('.h5')[0] for e in existed]
    count = len(existed)

    for file in sorted(os.listdir(rootpath)):
        total_Slide = len(os.listdir(rootpath))
        patch_h5 = os.path.join(rootpath, file)

        if os.path.isfile(patch_h5):
            slide_dir = file.split('.h5')[0]

            src_path = os.path.join(source_dir, slide_dir + args.slide_ext)
            if os.path.exists(src_path) and (slide_dir not in existed):
                count += 1
                print(f'{slide_dir}  {count}/{total_Slide}')
                with h5py.File(patch_h5, 'r') as hdf5_file:
                    coords = hdf5_file['coords'][:]
                    patch_level = hdf5_file['coords'].attrs['patch_level']
                    patch_size = hdf5_file['coords'].attrs['patch_size']
                index = get_patch_var1(coords, src_path, patch_level, patch_size)
                with h5py.File(f'{args.patch_index_dir}/{slide_dir}.h5', 'w') as f:
                    f[slide_dir] = index

