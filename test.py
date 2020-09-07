import torch
from nets.retinaface import RetinaFace
from utils.config import cfg_mnet
if __name__ == '__main__':
    # Test EfficientNet
    model = RetinaFace(cfg_mnet)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
