import torch
import numpy as np


def point_form(boxes):
    # 转换形式，转换成左上角右下角的形式
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    # 转换成中心宽高的形式
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    # 计算所有真实框和先验框的交面积
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    # 计算所有真实框和先验框的交并比
    # 行为真实框，列为先验框
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    # 计算交并比
    overlaps = jaccard(
        truths,
        point_form(priors)
    )

    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # 计算每个先验框最对应的真实框
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    # 找到与真实框重合程度最好的先验框，用于保证每个真实框都要有对应的一个先验框
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 对best_truth_idx内容进行设置
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    matches = truths[best_truth_idx]            
    # Shape: [num_priors] 此处为每一个anchor对应的label取出来
    conf = labels[best_truth_idx]        
           
    conf[best_truth_overlap < threshold] = 0    
    loc = encode(matches, priors, variances)

    matches_landm = landms[best_truth_idx]
    landm = encode_landm(matches_landm, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    landm_t[idx] = landm


def encode(matched, priors, variances):
    # 进行编码的操作
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # 中心编码
    g_cxcy /= (variances[0] * priors[:, 2:])
    
    # 宽高编码
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def encode_landm(matched, priors, variances):

    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)

    # 减去中心后除上宽高
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    return g_cxcy

def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

def decode(loc, priors, variances):
    # 中心解码，宽高解码
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    # 关键点解码
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def non_max_suppression(boxes, conf_thres=0.5, nms_thres=0.3):
    detection = boxes
    # 1、找出该图片中得分大于门限函数的框。在进行重合框筛选前就进行得分的筛选可以大幅度减少框的数量。
    mask = detection[:,4] >= conf_thres
    detection = detection[mask]
    if not np.shape(detection)[0]:
        return []

    best_box = []
    scores = detection[:,4]
    # 2、根据得分对框进行从大到小排序。
    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]

    while np.shape(detection)[0]>0:
        # 3、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
        best_box.append(detection[0])
        if len(detection) == 1:
            break
        ious = iou(best_box[-1],detection[1:])
        detection = detection[1:][ious<nms_thres]

    return np.array(best_box)

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
    return iou
