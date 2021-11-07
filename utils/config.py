cfg_mnet = {
    'name'              : 'mobilenet0.25',
    'min_sizes'         : [[16, 32], [64, 128], [256, 512]],
    'steps'             : [8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    #------------------------------------------------------------------#
    #   视频上看到的训练图片大小为640，为了提高大图状态下的困难样本
    #   的识别能力，我将训练图片进行调大
    #------------------------------------------------------------------#
    'train_image_size'  : 840,
    'return_layers'     : {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel'        : 32,
    'out_channel'       : 64
}

cfg_re50 = {
    'name'              : 'Resnet50',
    'min_sizes'         : [[16, 32], [64, 128], [256, 512]],
    'steps'             : [8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    'train_image_size'  : 840,
    'return_layers'     : {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel'        : 256,
    'out_channel'       : 256
}

