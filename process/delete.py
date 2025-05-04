import os
import shutil


def delete(args):
    if os.path.exists(args.data_h5_dir):
        shutil.rmtree(args.data_h5_dir)

    if os.path.exists(args.feat_dir):
        shutil.rmtree(args.feat_dir)

    if os.path.exists(args.patch_index_dir):
        shutil.rmtree(args.patch_index_dir)

    if os.path.exists(args.cluster_dir):
        shutil.rmtree(args.cluster_dir)
