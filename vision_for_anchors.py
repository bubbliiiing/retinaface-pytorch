from itertools import product as product
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.config import cfg_mnet


#-----------------------------#
#   中心解码，宽高解码
#-----------------------------#
def decode(loc, priors, variances):
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

#-----------------------------#
#   关键点解码
#-----------------------------#
def decode_landm(pre, priors, variances):
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms

class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        self.min_sizes  = cfg['min_sizes']
        self.steps      = cfg['steps']
        self.clip       = cfg['clip']
        #---------------------------#
        #   图片的尺寸
        #---------------------------#
        self.image_size = image_size
        #---------------------------#
        #   三个有效特征层高和宽
        #---------------------------#
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            #-----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            #-----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        output  = np.zeros_like(anchors[:, :4])
        output[:,0] = anchors[:,0] - anchors[:,2] / 2
        output[:,1] = anchors[:,1] - anchors[:,3] / 2
        output[:,2] = anchors[:,0] + anchors[:,2] / 2
        output[:,3] = anchors[:,1] + anchors[:,3] / 2

        if self.clip:
            output = np.clip(output, 0, 1)
        return output

if __name__ == "__main__":
    cfg_mnet['image_size'] = 640
    #--------------------------------#
    #   先验框的生成
    #--------------------------------#
    cfg     = cfg_mnet
    anchors = Anchors(cfg, image_size = (cfg_mnet['image_size'], cfg_mnet['image_size'])).get_anchors()
    anchors = anchors[-800:] * cfg_mnet['image_size']

    #--------------------------------#
    #   先验框中心绘制
    #--------------------------------#
    center_x = (anchors[:, 0] + anchors[:, 2]) / 2
    center_y = (anchors[:, 1] + anchors[:, 3]) / 2

    fig = plt.figure()
    ax  = fig.add_subplot(121)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    ax.invert_yaxis()
    plt.scatter(center_x,center_y)

    #--------------------------------#
    #   先验框宽高绘制
    #--------------------------------#
    box_widths  = anchors[0:2,2] - anchors[0:2,0]
    box_heights = anchors[0:2,3] - anchors[0:2,1]
    for i in [0,1]:
        rect = plt.Rectangle([anchors[i, 0], anchors[i, 1]], box_widths[i], box_heights[i], color="r", fill=False)
        ax.add_patch(rect)

    #--------------------------------#
    #   先验框中心绘制
    #--------------------------------#
    ax = fig.add_subplot(122)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    ax.invert_yaxis()  #y轴反向
    plt.scatter(center_x,center_y)

    #--------------------------------#
    #   对先验框调整获得预测框
    #--------------------------------#
    mbox_loc = np.random.randn(800, 4)
    mbox_ldm = np.random.randn(800, 10)

    anchors[:, :2] = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchors[:, 2:] = (anchors[:, 2:] - anchors[:, :2]) * 2

    mbox_loc                = torch.Tensor(mbox_loc)
    anchors                 = torch.Tensor(anchors)
    cfg_mnet['variance']    = torch.Tensor(cfg_mnet['variance'])
    decode_bbox             = decode(mbox_loc, anchors, cfg_mnet['variance'])

    box_widths  = decode_bbox[0: 2, 2] - decode_bbox[0: 2, 0]
    box_heights = decode_bbox[0: 2, 3] - decode_bbox[0: 2, 1]

    for i in [0,1]:
        rect = plt.Rectangle([decode_bbox[i, 0], decode_bbox[i, 1]], box_widths[i], box_heights[i], color="r", fill=False)
        plt.scatter((decode_bbox[i, 2] + decode_bbox[i, 0]) / 2, (decode_bbox[i,3] + decode_bbox[i,1]) / 2, color="b")
        ax.add_patch(rect)

    plt.show()
