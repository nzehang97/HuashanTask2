import argparse
from process.create_patches_fp import create_patches
from process.extract_features_fp import feature_extract
from process.var_extract_resnet import var_index
from process.cluster_100 import cluster
from process.create_heatmap_result import create_result
from process.delete import delete

parser = argparse.ArgumentParser(description='CAMIL_process')
parser.add_argument('--source', type=str, default='DATA', help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type=int, default=256, help='step_size')
parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
parser.add_argument('--patch', default=True, action='store_true')
parser.add_argument('--seg', default=True, action='store_true')
parser.add_argument('--stitch', default=True, action='store_true')
parser.add_argument('--no_auto_skip', default=False, action='store_false')
parser.add_argument('--preset', default=None, type=str)

parser.add_argument('--patch_level', type=int, default=0)
parser.add_argument('--process_list', type=str, default=None)
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--data_h5_dir', type=str, default='0_RESULTS_DIRECTORY')
parser.add_argument('--slide_ext', type=str, default='.svs')  # 需要命令输入
parser.add_argument('--csv_path', type=str, default='0_RESULTS_DIRECTORY/process_list.csv')

parser.add_argument('--feat_dir', type=str, default='0_Extracted_feature')
parser.add_argument('--patches_h5_dir', type=str, default='0_RESULTS_DIRECTORY/patches')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=256)

parser.add_argument('--patch_index_dir', type=str, default='0_patch_index')
parser.add_argument('--cluster_dir', type=str, default='0_cluster_data')
parser.add_argument('--result_dir', type=str, default='result_data')
args = parser.parse_args()

if __name__ == '__main__':

    print('\ncreating patches......')
    create_patches(args)
    print('\nfeature extracting......')
    feature_extract(args)
    print('\nremoving background......')
    var_index(args)
    print('\nclustering......')
    cluster(args)
    print('\ncreating result......')
    create_result(args)
    print('\ndeleting......')
    delete(args)
    print('\nFinished~')
