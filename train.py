'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import hydra

import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import logging

from models.registry import MODELS_REGISTRY


import hydra

from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'






@hydra.main(config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    
    print(OmegaConf.to_yaml(cfg))


    logger = logging.getLogger(__name__)
    
    TRAIN_WRITER = SummaryWriter(log_dir="./train")
    VAL_WRITER = SummaryWriter(log_dir="./val")

    logger.info("processs arguments")

    DATA_DIR = to_absolute_path(cfg.data_dir)
    EPOCHS = cfg.epochs
   
    
    logger.info("Preparing transformations")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # download and prepare data !
    trainset = torchvision.datasets.CIFAR10(
    root= DATA_DIR, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    valset = torchvision.datasets.CIFAR10(
        root= DATA_DIR, train=False, download=True, transform=transform_test)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    logger.info("Building the moded now")
    
    net_class = MODELS_REGISTRY.get(cfg.model)
    net = net_class()
    net = net.to(DEVICE)

    if cfg.checkpoint:
        # Load checkpoint.
        CKPT = to_absolute_path(cfg.checkpoint)
        logger.info('==> Resuming from checkpoint..')
        assert os.path.exists(CKPT), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(CKPT)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr,
                        momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # training and validation loops
    best_acc = 0
    for epoch in range(EPOCHS):

        logger.info("Epoch: {}".format(epoch))

        global_step = epoch*len(trainloader)
  
        net.train()
        train_loss = 0
        correct= 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # calculate accuracy and loss over the current batch only !
            TRAIN_WRITER.add_scalar("Loss/Batch",loss.item() , global_step+batch_idx)
            TRAIN_WRITER.add_scalar("Accuracy/Batch",100.* predicted.eq(targets).sum().item()/targets.size(0) , global_step + batch_idx)

            print("Epoch: {}/{} | Batch: {}/{} | ".format(epoch, EPOCHS, batch_idx, len(trainloader)), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            # run a validation batch every certain number of trainig batches
            
            if(batch_idx %cfg.val_interval == 0):
                local_step = global_step + batch_idx
                logger.info("evalutating mini batch")
                # net.eval()
                val_loss = 0
                val_correct= 0
                val_total = 0
                for val_batch_idx, (inputs, targets) in enumerate(valloader):
                    
                    if val_batch_idx == cfg.val_batches:
                        break
                    
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                  
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

                
                VAL_WRITER.add_scalar("Loss/Batch",val_loss/(cfg.val_batches) , local_step)
                VAL_WRITER.add_scalar("Accuracy/Batch",100.* val_correct/val_total , local_step)
        
        # # Train metrics over Epoch
        # TRAIN_WRITER.add_scalar("Loss/Epoch",train_loss/(len(trainloader)) , epoch)
        # TRAIN_WRITER.add_scalar("Accuracy/Epoch",100.* correct/total , epoch)
        
        # # run full validation cycle
        # net.eval()
        # eval_loss = 0
        # eval_correct= 0
        # eval_total = 0

        # for batch_idx, (inputs, targets) in enumerate(valloader):
        #     inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
        #     outputs = net(inputs)
        #     loss = criterion(outputs, targets)
           
            

        #     eval_loss += loss.item()
        #     _, predicted = outputs.max(1)
        #     eval_total += targets.size(0)
        #     eval_correct += predicted.eq(targets).sum().item()
        
        # # val metrics over epoch
        # VAL_WRITER.add_scalar("Loss/Epoch",eval_loss/(len(valloader)) , epoch)
        # VAL_WRITER.add_scalar("Accuracy/Epoch",100.* eval_correct/eval_total , epoch)

        #  # Save checkpoint.
        # acc = 100.*eval_correct/eval_total
        # if acc > best_acc:
        #     logger.info('Saving..')
        #     state = {
        #         'net': net.state_dict(),
        #         'acc': acc,
        #         'epoch': epoch,
        #     }
        #     if not os.path.isdir('checkpoint'):
        #         os.mkdir('checkpoint')
        #     torch.save(state, './checkpoint/ckpt.pth')
        #     best_acc = acc

        # scheduler.step()


    
if __name__ == "__main__":
    my_app()