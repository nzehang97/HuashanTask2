import argparse
import os
import h5py
import time
import numpy as np
from sklearn.cluster import KMeans


def get_feature(fea_path, var_index):
    if os.path.isfile(fea_path):
        with h5py.File(fea_path, 'r') as hdf5_file:
            feature = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]

        mask = np.zeros(feature.shape[0], dtype=bool)
        for i in var_index:
            mask[i] = True
        feature = feature[mask]
        coords = coords[mask]
        return feature, coords


def chuster(feas_path, CLASS_NUM, slide_id, args, var_index):
    print('\ncluster--', slide_id)
    fea_path = os.path.join(feas_path, slide_id + '.h5')
    feature_raw, coords_raw= get_feature(fea_path, var_index)

    print(f'Kmeans--{CLASS_NUM}类--start')
    start_time = time.time()
    kmeans = KMeans(n_clusters=CLASS_NUM, random_state=0).fit(feature_raw)
    print(f'spend-{time.time() - start_time}')
    print(f'Kmeans--{CLASS_NUM}类--end')
    label = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    with h5py.File(f'{args.cluster_dir}/kmeans_{CLASS_NUM}_index/{slide_id}.h5', 'w') as hh:
        hh[f'{slide_id}'] = label
    feature_49 = []
    coord_49 = []
    for j in range(CLASS_NUM):
        clu_cen_j = cluster_centers[j, :].reshape(-1, 1)
        class_j_idx = np.array([idx for idx, l in enumerate(label) if l == j])

        temp_path_perm = np.tile(clu_cen_j, len(class_j_idx))
        sam_j = feature_raw[class_j_idx, :].T
        sum_data = np.sum((sam_j - temp_path_perm) ** 2, axis=0)
        a = np.argmin(sum_data)

        feature_49.append(feature_raw[class_j_idx[a]])
        coord_49.append(coords_raw[class_j_idx[a]])
    assert len(coord_49) == CLASS_NUM
    with h5py.File(f'{args.cluster_dir}/features_{CLASS_NUM}/{slide_id}.h5', 'w') as ff:
        ff[f'features'] = feature_49
        ff[f'coords'] = coord_49


def cluster(args):
    CLASS_NUM = 100
    feas_path = args.feat_dir

    total_Slide = len(os.listdir(feas_path))
    count = 0
    os.makedirs(f'{args.cluster_dir}/features_{CLASS_NUM}', exist_ok=True)
    os.makedirs(f'{args.cluster_dir}/kmeans_{CLASS_NUM}_index', exist_ok=True)

    for file in os.listdir(feas_path):
        slide_id = file.split('.h5')[0]
        if os.path.exists(f'{args.cluster_dir}/features_{CLASS_NUM}/{file}'):
            continue

        with h5py.File(f'{args.patch_index_dir}/{slide_id}.h5', 'r') as hdf5_file:
            var_index = hdf5_file[f'{slide_id}'][:]
        if len(var_index) < CLASS_NUM:
            continue

        count += 1
        print(f'{slide_id}  {count}/{total_Slide}')

        chuster(feas_path, CLASS_NUM, slide_id, args, var_index)

