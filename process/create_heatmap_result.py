# _*_ coding:utf-8 _*_
from __future__ import print_function
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from utils.vis_utils.heatmap_utils import initialize_wsi, drawHeatmap
from models.model_clam import Attn_Net_Gated, initialize_weights
import torch.nn.functional as F


class CLAM_SB(nn.Module):
    def __init__(self, dropout, drop_att):
        super(CLAM_SB, self).__init__()
        size = [768, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=drop_att, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers_1 = nn.Linear(size[1], 3)
        initialize_weights(self)

    def forward(self, h0):
        h0 = h0.to(torch.float32)
        A, h = self.attention_net(h0[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A_raw = torch.transpose(A, 2, 1)  # KxN A(8, 1, 100)
        A = F.softmax(A_raw, dim=2)  # softmax over N
        M = torch.squeeze(torch.bmm(A, h), axis=1)

        logits = self.classifiers_1(M)

        return logits, A_raw


def create_result(args):
    files = np.sort(os.listdir(args.patches_h5_dir))
    os.makedirs(args.result_dir, exist_ok=True)
    for slide_id in files:
        slide_id = slide_id.split('.h5')[0]

        slide_path = f'{args.source}/{slide_id+args.slide_ext}'

        seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': [],
                      'exclude_ids': []}
        filter_params = {'a_t': 1, 'a_h': 1, 'max_n_holes': 2}

        scale = 17
        wsi_object = initialize_wsi(slide_path, seg_params=seg_params, filter_params=filter_params, scale=scale)

        with h5py.File(f'{args.patches_h5_dir}/' + slide_id + '.h5', 'r') as hdf5_file:
            coords = hdf5_file['coords'][:]
            patch_level = hdf5_file['coords'].attrs['patch_level']
            patch_size = hdf5_file['coords'].attrs['patch_size']
            patch_size = patch_size if patch_level == 0 else patch_size*2*patch_level

        with h5py.File(f'{args.feat_dir}/{slide_id}.h5', 'r') as hdf5_file:
            feature = hdf5_file['features'][:]

        with h5py.File(f'{args.patch_index_dir}/{slide_id}.h5', 'r') as hdf5_file:
            var_index = hdf5_file[f'{slide_id}'][:]
            mask = np.zeros(coords.shape[0], dtype=bool)
            for i in var_index:
                mask[i] = True
            coords = coords[mask]
            feature = torch.from_numpy(feature[mask]).unsqueeze(0)
        with h5py.File(f'{args.cluster_dir}/features_100/{slide_id}.h5', 'r') as hdf5_file:
            feature_49 = torch.from_numpy(hdf5_file['features'][:]).unsqueeze(0)

        model = CLAM_SB(dropout=0, drop_att=0).to(args.device)
        weight_path = 'weight_fold_0.pth'
        assert os.path.exists(weight_path), "file: '{}' dose not exist.".format(weight_path)
        model.load_state_dict(torch.load(weight_path, map_location=args.device))

        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            out, _ = model(feature_49.to(args.device))
            out = softmax(out).flatten().cpu()

        with torch.no_grad():
            _, scores = model(feature.to(args.device))
        scores = scores.flatten().cpu()
        # 热图展示
        # cmap: https://matplotlib.org/stable/gallery/color/colormap_reference.html
        heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap='jet',
                              alpha=0.5, use_holes=True, binarize=False, vis_level=-1, scale=scale,
                              blank_canvas=False, thresh=-1, patch_size=patch_size*2, convert_to_percentiles=True)

        # draw = ImageDraw.Draw(heatmap)
        print(f'probability-----SF1:{out[0]}--Pit1:{out[1]}--Tpit:{out[2]}')
        heatmap.save(os.path.join(args.result_dir, f'{slide_id}_SF1:{out[0]:.5f}-Pit1:{out[1]:.5f}-Tpit:{out[2]:.5f}.png'))
        del heatmap

