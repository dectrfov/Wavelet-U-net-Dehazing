from utils import str2bool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ori_data_path', type=str, default='ori',  help='Origin image path')
parser.add_argument('--haze_data_path', type=str, default='haze',  help='Haze image path')
parser.add_argument('--val_ori_data_path', type=str, default='val_ori',  help='Validation origin image path')
parser.add_argument('--val_haze_data_path', type=str, default='val_haze',  help='Validation haze image path')
parser.add_argument('--sample_output_folder', type=str, default='samples/',  help='Validation haze image path')
parser.add_argument('--use_gpu', type=str2bool, default=True, help='Use GPU')
parser.add_argument('--gpu', type=int, default=-1, help='GPU id')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--num_workers', type=int, default=4, help='Number of threads for data loader, for window set to 0')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--val_batch_size', type=int, default=16, help='Validation batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
parser.add_argument('--model_dir', type=str, default='./model/')
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--ckpt', type=str, default='dehaze_chromatic_100.pkl')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
