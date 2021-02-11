from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
import torchvision.models.detection.backbone_utils as backbone_utils
from torchvision import models

from nets.layers import FPN, SSH
from nets.mobilenet025 import MobileNetV1


#---------------------------------------------------#
#   种类预测（是否包含人脸）
#---------------------------------------------------#
class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

#---------------------------------------------------#
#   预测框预测
#---------------------------------------------------#
class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

#---------------------------------------------------#
#   人脸关键点预测
#---------------------------------------------------#
class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, pretrained = False, mode = 'train'):
        super(RetinaFace,self).__init__()
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if pretrained:
                checkpoint = torch.load("./model_data/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=pretrained)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        # 获得每个初步有效特征层的通道数
        in_channels_list = [cfg['in_channel'] * 2, cfg['in_channel'] * 4, cfg['in_channel'] * 8]
        self.fpn = FPN(in_channels_list, cfg['out_channel'])
        # 利用ssh模块提高模型感受野
        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

        self.mode = mode

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是C3  80, 80, 64
        #         C4  40, 40, 128
        #         C5  20, 20, 256
        #-------------------------------------------#
        out = self.body.forward(inputs)

        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是output1  80, 80, 64
        #         output2  40, 40, 64
        #         output3  20, 20, 64
        #-------------------------------------------#
        fpn = self.fpn.forward(out)

        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        #-------------------------------------------#
        #   将所有结果进行堆叠
        #-------------------------------------------#
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.mode == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
