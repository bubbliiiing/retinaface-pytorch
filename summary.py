#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.retinaface import RetinaFace
from utils.config import cfg_mnet

if __name__ == '__main__':
    #--------------------------------------------#
    #   需要使用device来指定网络在GPU还是CPU运行
    #--------------------------------------------#
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = RetinaFace(cfg_mnet).to(device)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    summary(model, input_size=(3, 1280, 1280))
