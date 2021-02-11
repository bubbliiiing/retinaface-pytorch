import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
from torch.autograd import Variable


def conv_bn(inp, oup, stride = 1, leaky = 0.1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
    
def conv_dw(inp, oup, stride = 1, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 640,640,3 -> 320,320,8
            conv_bn(3, 8, 2, leaky = 0.1),
            # 320,320,8 -> 320,320,16
            conv_dw(8, 16, 1),

            # 320,320,16 -> 160,160,32
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),

            # 160,160,32 -> 80,80,64
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
        )
        # 80,80,64 -> 40,40,128
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
        )
        # 40,40,128 -> 20,20,256
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), 
            conv_dw(256, 256, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
