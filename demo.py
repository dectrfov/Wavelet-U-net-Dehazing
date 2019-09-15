import os
import csv
import glob
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import numpy as np
from PIL import Image
from utils import logger
from config import get_config
from model import ACT

@logger
def make_test_data(cfg, img_path_list, device):
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([480, 640]),
        torchvision.transforms.ToTensor()
    ])
    imgs = []
    for img_path in img_path_list:
        x = data_transform(Image.open(str(img_path))).unsqueeze(0)
        x = x.to(device)
        imgs.append(x)
    return imgs


@logger
def load_pretrain_network(cfg, device):
    net = ACT().to(device)
    net.load_state_dict(torch.load(os.path.join(cfg.model_dir,  cfg.ckpt)))
    return net


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------

    network = load_pretrain_network(cfg, device)
    print('Start eval')
    network.eval()

    # load data
    path = cfg.sample_output_folder
    name= os.listdir(path)

    for i in name:
        test_file_path =path+i
        test_file_path = glob.glob(test_file_path)
        test_images = make_test_data(cfg, test_file_path, device)
        dehaze_image = network(test_images[0])
        torchvision.utils.save_image(torch.cat((test_images[0], dehaze_image), 0), "result/" + i)

if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
    #python demo.py  --net_name aod --sample_output_folder samples/  --use_gpu true --gpu 0 --model_dir model/ --ckpt dehaze_chromatic_100.pkl