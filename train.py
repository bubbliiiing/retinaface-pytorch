from __future__ import print_function
import os
import time
import math
import torch
import numpy as np
import datetime
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from nets.retinaface import RetinaFace
from nets.retinaface_training import DataGenerator, MultiBoxLoss, detection_collate
from utils.anchors import Anchors
from utils.config import cfg_re50, cfg_mnet


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,criterion,epoch,epoch_size,gen,Epoch,anchors,cfg,cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_landmark_loss = 0

    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            # forward
            out = net(images)
            r_loss, c_loss, landm_loss = criterion(out, anchors, targets)
            loss = cfg['loc_weight'] * r_loss + c_loss + landm_loss

            loss.backward()
            optimizer.step()
            
            total_c_loss += c_loss.item()
            total_r_loss += r_loss.item()
            total_landmark_loss += landm_loss.item()
            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'Conf Loss'         : total_c_loss / (iteration + 1), 
                                'Regression Loss'   : total_r_loss / (iteration + 1), 
                                'LandMark Loss'     : total_landmark_loss / (iteration + 1), 
                                'lr'                : get_lr(optimizer),
                                's/step'            : waste_time})
            pbar.update(1)
            start_time = time.time()

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f.pth'%((epoch+1),(total_c_loss + total_r_loss + total_landmark_loss)/(epoch_size+1)))
    return (total_c_loss + total_r_loss + total_landmark_loss)/(epoch_size+1)

if __name__ == "__main__":
    num_classes = 2
    #-------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet或者resnet50
    #-------------------------------#
    backbone = "mobilenet"
    training_dataset_path = './data/widerface/train/label.txt'
    #-------------------------------#
    #   是否使用主干特征提取网络
    #   的预训练权重
    #-------------------------------#
    pretrained = False

    if backbone == "mobilenet":
        cfg = cfg_mnet
    elif backbone == "resnet50":  
        cfg = cfg_re50
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))
    
    img_dim = cfg['image_size']

    #-------------------------------#
    #   Cuda的使用
    #-------------------------------#
    Cuda = True
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    #-------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    #-------------------------------------------#
    model = RetinaFace(cfg=cfg, pretrained = pretrained).train()
    model_path = "model_data/Retinaface_mobilenet0.25.pth"
    # 加快模型训练的效率
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    anchors = Anchors(cfg, image_size=(img_dim, img_dim)).get_anchors()
    if Cuda:
        anchors = anchors.cuda()

    criterion = MultiBoxLoss(num_classes, 0.35, 7, Cuda)
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr = 1e-3
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 25
        
        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_dataset = DataGenerator(training_dataset_path,img_dim)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)

        epoch_size = train_dataset.get_len()//Batch_size
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.body.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            loss = fit_one_epoch(net,criterion,epoch,epoch_size,gen,Freeze_Epoch,anchors,cfg,Cuda)
            lr_scheduler.step(loss)

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr = 1e-4
        Batch_size = 4
        Freeze_Epoch = 25
        Unfreeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
        train_dataset = DataGenerator(training_dataset_path,img_dim)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        epoch_size = train_dataset.get_len()//Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.body.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            loss = fit_one_epoch(net,criterion,epoch,epoch_size,gen,Unfreeze_Epoch,anchors,cfg,Cuda)
            lr_scheduler.step(loss)