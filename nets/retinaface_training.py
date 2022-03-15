import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------#
#   获得框的左上角和右下角
#------------------------------#
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,
                     boxes[:, :2] + boxes[:, 2:]/2), 1)

#------------------------------#
#   获得框的中心和宽高
#------------------------------#
def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,
                     boxes[:, 2:] - boxes[:, :2], 1)

#----------------------------------#
#   计算所有真实框和先验框的交面积
#----------------------------------#
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    #------------------------------#
    #   获得交矩形的左上角
    #------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    #------------------------------#
    #   获得交矩形的右下角
    #------------------------------#
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    #-------------------------------------#
    #   计算先验框和所有真实框的重合面积
    #-------------------------------------#
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    #-------------------------------------#
    #   返回的inter的shape为[A,B]
    #   代表每一个真实框和先验框的交矩形
    #-------------------------------------#
    inter = intersect(box_a, box_b)
    #-------------------------------------#
    #   计算先验框和真实框各自的面积
    #-------------------------------------#
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter
    #-------------------------------------#
    #   每一个真实框和先验框的交并比[A,B]
    #-------------------------------------#
    return inter / union  # [A,B]

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

def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    #----------------------------------------------#
    #   计算所有的先验框和真实框的重合程度
    #----------------------------------------------#
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    #----------------------------------------------#
    #   所有真实框和先验框的最好重合程度
    #   best_prior_overlap [truth_box,1]
    #   best_prior_idx [truth_box,1]
    #----------------------------------------------#
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    #----------------------------------------------#
    #   所有先验框和真实框的最好重合程度
    #   best_truth_overlap [1,prior]
    #   best_truth_idx [1,prior]
    #----------------------------------------------#
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    #----------------------------------------------#
    #   用于保证每个真实框都至少有对应的一个先验框
    #----------------------------------------------#
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 对best_truth_idx内容进行设置
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    #----------------------------------------------#
    #   获取每一个先验框对应的真实框[num_priors,4]
    #----------------------------------------------#
    matches = truths[best_truth_idx]            
    # Shape: [num_priors] 此处为每一个anchor对应的label取出来
    conf = labels[best_truth_idx]        
    matches_landm = landms[best_truth_idx]
           
    #----------------------------------------------#
    #   如果重合程度小于threhold则认为是背景
    #----------------------------------------------#
    conf[best_truth_overlap < threshold] = 0    
    #----------------------------------------------#
    #   利用真实框和先验框进行编码
    #   编码后的结果就是网络应该有的预测结果
    #----------------------------------------------#
    loc = encode(matches, priors, variances)
    landm = encode_landm(matches_landm, priors, variances)

    #----------------------------------------------#
    #   [num_priors, 4]
    #----------------------------------------------#
    loc_t[idx] = loc
    #----------------------------------------------#
    #   [num_priors]
    #----------------------------------------------#
    conf_t[idx] = conf
    #----------------------------------------------#
    #   [num_priors, 10]
    #----------------------------------------------#
    landm_t[idx] = landm


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_pos, variance, cuda=True):
        super(MultiBoxLoss, self).__init__()
        #----------------------------------------------#
        #   对于retinaface而言num_classes等于2
        #----------------------------------------------#
        self.num_classes    = num_classes
        #----------------------------------------------#
        #   重合程度在多少以上认为该先验框可以用来预测
        #----------------------------------------------#
        self.threshold      = overlap_thresh
        #----------------------------------------------#
        #   正负样本的比率
        #----------------------------------------------#
        self.negpos_ratio   = neg_pos
        self.variance       = variance
        self.cuda           = cuda

    def forward(self, predictions, priors, targets):
        #--------------------------------------------------------------------#
        #   取出预测结果的三个值：框的回归信息，置信度，人脸关键点的回归信息
        #--------------------------------------------------------------------#
        loc_data, conf_data, landm_data = predictions
        #--------------------------------------------------#
        #   计算出batch_size和先验框的数量
        #--------------------------------------------------#
        num         = loc_data.size(0)
        num_priors  = (priors.size(0))

        #--------------------------------------------------#
        #   创建一个tensor进行处理
        #--------------------------------------------------#
        loc_t   = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t  = torch.LongTensor(num, num_priors)

        for idx in range(num):
            # 获得真实框与标签
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data

            # 获得先验框
            defaults = priors.data
            #--------------------------------------------------#
            #   利用真实框和先验框进行匹配。
            #   如果真实框和先验框的重合度较高，则认为匹配上了。
            #   该先验框用于负责检测出该真实框。
            #--------------------------------------------------#
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
            
        #--------------------------------------------------#
        #   转化成Variable
        #   loc_t   (num, num_priors, 4)
        #   conf_t  (num, num_priors)
        #   landm_t (num, num_priors, 10)
        #--------------------------------------------------#
        zeros = torch.tensor(0)
        if self.cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            zeros = zeros.cuda()

        #------------------------------------------------------------------------#
        #   有人脸关键点的人脸真实框的标签为1，没有人脸关键点的人脸真实框标签为-1
        #   所以计算人脸关键点loss的时候pos1 = conf_t > zeros
        #   计算人脸框的loss的时候pos = conf_t != zeros
        #------------------------------------------------------------------------#  
        pos1 = conf_t > zeros
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        
        pos = conf_t != zeros
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        #--------------------------------------------------#
        #   batch_conf  (num * num_priors, 2)
        #   loss_c      (num, num_priors)
        #--------------------------------------------------#
        conf_t[pos] = 1
        batch_conf = conf_data.view(-1, self.num_classes)
        # 这个地方是在寻找难分类的先验框
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # 难分类的先验框不把正样本考虑进去，只考虑难分类的负样本
        loss_c[pos.view(-1, 1)] = 0
        loss_c = loss_c.view(num, -1)
        #--------------------------------------------------#
        #   loss_idx    (num, num_priors)
        #   idx_rank    (num, num_priors)
        #--------------------------------------------------#
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        #--------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   num_pos     (num, )
        #   neg         (num, num_priors)
        #--------------------------------------------------#
        num_pos = pos.long().sum(1, keepdim=True)
        # 限制负样本数量
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        #--------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   pos_idx   (num, num_priors, num_classes)
        #   neg_idx   (num, num_priors, num_classes)
        #--------------------------------------------------#
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        
        # 选取出用于训练的正样本与负样本，计算loss
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        loss_landm /= N1
        return loss_l, loss_c, loss_landm

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
