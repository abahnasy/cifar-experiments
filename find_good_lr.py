"""
Estimating an Optimal Learning Rate For a Deep Neural Network
Idea borrowed from this paper: Cyclical Learning Rates for Training Neural Networks
Ref: https://arxiv.org/abs/1506.01186
"""

import os
import argparse
import logging
import math


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
from models.registry import MODELS_REGISTRY


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'





def find_lr(
    model,
    criterion, 
    data_loader, 
    optimizer,
    lr_scheduler=None,
    init_value = 1e-8, 
    final_value=10., 
    beta = 0.98, 
    logger=None, 
    TBLogger=None):
    """
    """

    num = len(data_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses_list = []
    log_lrs = []


    # max_epochs = 1
    # max_iter = max_epochs*len(data_loader) # TODO: rename to train_data_loader
    # _epoch = 0
    # _iter = 0
    # total_steps = 1 * len(data_loader)
    
    
    
    
    model.train()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        batch_num += 1
        optimizer.zero_grad()
        # torch.cuda.reset_max_memory_allocated()
        logger.info("processing training batch [{}/{}]".format(batch_idx, len(data_loader)))

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        TBLogger.add_scalar("lr_finder/avg_loss", avg_loss, batch_num)
        TBLogger.add_scalar("lr_finder/smoothed_loss", smoothed_loss, batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses_list
        # Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        TBLogger.add_scalar('lr_finder/best_loss', best_loss, batch_num)
        losses_list.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        TBLogger.add_scalar('lr_finder/log_lrs', log_lrs[-1], batch_num) # get last saved one in the list
        TBLogger.add_scalar('lr_finder/lr', lr, batch_num) # actual lr value

        loss.backward()
        optimizer.step()
        # _iter+=1
        
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        

    return log_lrs, losses_list




def prepare_data_loaders(cfg, logger):
    """
    """
    logger.info("Preparing transformations")

    DATA_DIR = to_absolute_path(cfg.data_dir)    
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # download and prepare data !
    trainset = torchvision.datasets.CIFAR10(
    root= DATA_DIR, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    # valset = torchvision.datasets.CIFAR10(
    #     root= DATA_DIR, train=False, download=True, transform=transform_test)
    # valloader = torch.utils.data.DataLoader(
    #     valset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    return trainloader



@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    """
    """

    # prepare loggers and read configurations
    print(OmegaConf.to_yaml(cfg))
    logger = logging.getLogger(__name__)
    
    WRITER = SummaryWriter(log_dir="./find_good_lr")
    
    # prepare network
    net_class = MODELS_REGISTRY.get(cfg.model)
    net = net_class()
    net = net.to(DEVICE)
    # prepare optimizers and losses
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr,
                            momentum=0.9, weight_decay=5e-4)
    # prepare dataloaders
    data_loader = prepare_data_loaders(cfg, logger)

    log_lrs, losses_list = find_lr(net, criterion, data_loader, optimizer, lr_scheduler = None, logger= logger, TBLogger=WRITER)
    # TODO: plot Loss Change in TBLogger


if __name__ == "__main__":
    main()

    

