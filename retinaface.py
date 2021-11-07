import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from nets.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import letterbox_image, preprocess_input
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)


#------------------------------------#
#   请注意主干网络与预训练权重的对应
#   即注意修改model_path和backbone
#------------------------------------#
class Retinaface(object):
    _defaults = {
        #---------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择损失较低的即可。
        #---------------------------------------------------------------------#
        "model_path"        : 'model_data/Retinaface_mobilenet0.25.pth',
        #---------------------------------------------------------------------#
        #   所使用的的主干网络：mobilenet、resnet50
        #---------------------------------------------------------------------#
        "backbone"          : 'mobilenet',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.45,
        #---------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        #---------------------------------------------------------------------#
        "input_shape"       : [1280, 1280, 3],
        #---------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #--------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #--------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Retinaface
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   不同主干网络的config信息
        #---------------------------------------------------#
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        #---------------------------------------------------#
        #   先验框的生成
        #---------------------------------------------------#
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()
        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.net    = RetinaFace(cfg=self.cfg, mode='eval').eval()

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_image = image.copy()
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image = np.array(image,np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        with torch.no_grad():
            #-----------------------------------------------------------#
            #   图片预处理，归一化。
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image        = image.cuda()

            #---------------------------------------------------------#
            #   传入网络进行预测
            #---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            
            #-----------------------------------------------------------#
            #   对预测框进行解码
            #-----------------------------------------------------------#
            boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            #-----------------------------------------------------------#
            #   获得预测结果的置信度
            #-----------------------------------------------------------#
            conf    = conf.data.squeeze(0)[:, 1:2]
            #-----------------------------------------------------------#
            #   对人脸关键点进行解码
            #-----------------------------------------------------------#
            landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

            #-----------------------------------------------------------#
            #   对人脸识别结果进行堆叠
            #-----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return old_image

            #---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            #---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
            
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            #---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            #---------------------------------------------------#
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            print(b[0], b[1], b[2], b[3], b[4])
            #---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            #---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
        return old_image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image = np.array(image,np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            #-----------------------------------------------------------#
            #   图片预处理，归一化。
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image        = image.cuda()

            #---------------------------------------------------------#
            #   传入网络进行预测
            #---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            #-----------------------------------------------------------#
            #   对预测框进行解码
            #-----------------------------------------------------------#
            boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            #-----------------------------------------------------------#
            #   获得预测结果的置信度
            #-----------------------------------------------------------#
            conf    = conf.data.squeeze(0)[:, 1:2]
            #-----------------------------------------------------------#
            #   对人脸关键点进行解码
            #-----------------------------------------------------------#
            landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

            #-----------------------------------------------------------#
            #   对人脸识别结果进行堆叠
            #-----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   传入网络进行预测
                #---------------------------------------------------------#
                loc, conf, landms = self.net(image)
                #-----------------------------------------------------------#
                #   对预测框进行解码
                #-----------------------------------------------------------#
                boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                #-----------------------------------------------------------#
                #   获得预测结果的置信度
                #-----------------------------------------------------------#
                conf    = conf.data.squeeze(0)[:, 1:2]
                #-----------------------------------------------------------#
                #   对人脸关键点进行解码
                #-----------------------------------------------------------#
                landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

                #-----------------------------------------------------------#
                #   对人脸识别结果进行堆叠
                #-----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def get_map_txt(self, image):
        #---------------------------------------------------#
        #   把图像转换成numpy的形式
        #---------------------------------------------------#
        image = np.array(image,np.float32)
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        #---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        #---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        with torch.no_grad():
            #-----------------------------------------------------------#
            #   图片预处理，归一化。
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image        = image.cuda()

            #---------------------------------------------------------#
            #   传入网络进行预测
            #---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            #-----------------------------------------------------------#
            #   对预测框进行解码
            #-----------------------------------------------------------#
            boxes   = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            #-----------------------------------------------------------#
            #   获得预测结果的置信度
            #-----------------------------------------------------------#
            conf    = conf.data.squeeze(0)[:, 1:2]
            #-----------------------------------------------------------#
            #   对人脸关键点进行解码
            #-----------------------------------------------------------#
            landms  = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

            #-----------------------------------------------------------#
            #   对人脸识别结果进行堆叠
            #-----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return np.array([])

            #---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            #---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
            
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        return boxes_conf_landms
