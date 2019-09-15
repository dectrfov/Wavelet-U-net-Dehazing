import os
import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from utils import logger, weight_init
from config import get_config
from model import ACT
from data import HazeDataset
import torchvision.models as models
import math
import numpy as np


@logger
def load_data(cfg):
    data_transform = transforms.Compose([
        transforms.Resize([480, 640]),
        transforms.ToTensor()
    ])
    train_haze_dataset = HazeDataset(cfg.ori_data_path, cfg.haze_data_path, data_transform)
    train_loader = torch.utils.data.DataLoader(train_haze_dataset, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    val_haze_dataset = HazeDataset(cfg.val_ori_data_path, cfg.val_haze_data_path, data_transform)
    val_loader = torch.utils.data.DataLoader(val_haze_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                                             num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    return train_loader, len(train_loader), val_loader, len(val_loader)

@logger
def save_model(epoch, path, net, optimizer, net_name):
    if not os.path.exists(os.path.join(path, net_name)):
        os.mkdir(os.path.join(path, net_name))
    torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
               f=os.path.join(path, net_name, '{}_{}.pkl'.format('', epoch)))

@logger
def load_optimizer(net, cfg):
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer

@logger
def load_network(device):
    net = ACT().to(device)
    net.apply(weight_init)
    return net

@logger
def loss_func(device):
    criterion = torch.nn.MSELoss().to(device)
    return criterion


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------
    # load summaries
    #summary = load_summaries(cfg)
    # -------------------------------------------------------------------
    # load data
    train_loader, train_number, val_loader, val_number = load_data(cfg)
    # -------------------------------------------------------------------
    # load loss
    criterion = loss_func(device)
    # -------------------------------------------------------------------
    # load network
    #network = load_network(device)
    network=ACT().to(device)
    # -------------------------------------------------------------------
    # load optimizer
    optimizer = load_optimizer(network, cfg)
    # -------------------------------------------------------------------
    # start train
    
    print('Start train')
    network.train()
    for epoch in range(cfg.epochs):
        Loss=0
        for step, (ori_image, haze_image) in enumerate(train_loader):
            count = epoch * train_number + (step + 1)
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            dehaze_image = network(haze_image)
            loss = criterion(dehaze_image, ori_image)
            Loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        
        print('Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.8f}'
                      .format(epoch , cfg.epochs, step + 1, train_number,
                              optimizer.param_groups[0]['lr'], Loss))
        # -------------------------------------------------------------------
        # start validation

        network.eval()
        Loss=0
        for step, (ori_image, haze_image) in enumerate(val_loader):
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            dehaze_image = network(haze_image)
            loss = criterion(dehaze_image, ori_image)
            
        test_loss.append(Loss)
        print('VAL Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.4f}|PNSR: {: .4f}'
                  .format(epoch + 1, cfg.epochs, step + 1, train_number,
                          optimizer.param_groups[0]['lr'], loss.item(),10*math.log10(1.0/loss.item())))
        
        torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat((haze_image, dehaze_image, ori_image), 0),nrow=ori_image.shape[0]),os.path.join(cfg.sample_output_folder, 'w{}_{}.jpg'.format(epoch , step)))
        
        network.train()
        # -------------------------------------------------------------------
        # save per epochs model
        save_model(epoch+1, cfg.model_dir, network, optimizer, cfg.net_name)
    # -------------------------------------------------------------------
    # train finish

if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
