import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat
from torch.autograd import Variable

from retinaface import Retinaface
from utils.anchors import Anchors
from utils.box_utils import (decode, decode_landm, letterbox_image,
                             non_max_suppression, retinaface_correct_boxes)
from utils.config import cfg_mnet, cfg_re50


def preprocess_input(image):
    image -= np.array((104, 117, 123),np.float32)
    return image

class FPS_Retinaface(Retinaface):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def get_FPS(self, image, test_interval):
        image = np.array(image,np.float32)
        im_height, im_width, _ = np.shape(image)
        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]]
        if self.letterbox_image:
            image = np.array(letterbox_image(image,[self.input_shape[1], self.input_shape[0]]), np.float32)
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0)
            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()
            loc, conf, landms = self.net(image)
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes = boxes.cpu().numpy()

            conf = conf.data.squeeze(0)[:,1:2].cpu().numpy()

            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            landms = landms.cpu().numpy()

            boxes_conf_landms = np.concatenate([boxes, conf, landms],-1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            if len(boxes_conf_landms)>0:
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
                    
                boxes_conf_landms[:,:4] = boxes_conf_landms[:,:4]*scale
                boxes_conf_landms[:,5:] = boxes_conf_landms[:,5:]*scale_for_landmarks

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                loc, conf, landms = self.net(image)
                boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                boxes = boxes.cpu().numpy()

                conf = conf.data.squeeze(0)[:,1:2].cpu().numpy()

                landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
                landms = landms.cpu().numpy()

                boxes_conf_landms = np.concatenate([boxes, conf, landms],-1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
                if len(boxes_conf_landms)>0:
                    if self.letterbox_image:
                        boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
                        
                    boxes_conf_landms[:,:4] = boxes_conf_landms[:,:4]*scale
                    boxes_conf_landms[:,5:] = boxes_conf_landms[:,5:]*scale_for_landmarks
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

if __name__ == '__main__':
    retinaface = FPS_Retinaface()
    test_interval = 100
    img = Image.open('img/street.jpg')
    tact_time = retinaface.get_FPS(img, test_interval)
    print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
