## Retinaface：人脸检测模型在Pytorch当中的实现
---

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [性能情况 Performance](#性能情况)
3. [所需环境 Environment](#所需环境)
4. [文件下载 Download](#文件下载)
5. [预测步骤 How2predict](#预测步骤)
6. [训练步骤 How2train](#训练步骤)
7. [评估步骤 Eval](#评估步骤)
8. [参考资料 Reference](#Reference)

## Top News
**`2022-03`**:**进行了大幅度的更新，支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/retinaface-pytorch/tree/bilibili

**`2020-09`**:**仓库创建，支持模型训练，大量的注释，多个主干的选择，多个可调整参数。**   

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | Easy | Medium | Hard |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| Widerface-Train | Retinaface_mobilenet0.25.pth | Widerface-Val | 1280x1280 | 89.76% | 86.96% | 74.69% |
| Widerface-Train | Retinaface_resnet50.pth | Widerface-Val | 1280x1280 | 94.72% | 93.13% | 84.48% |

## 所需环境
pytorch==1.2.0

## 文件下载
训练所需的Retinaface_resnet50.pth等文件可以在百度云下载。    
链接: https://pan.baidu.com/s/1Jt9Bo2UVP03bmEMuUpk_9Q 提取码: qknw     

数据集可以在如下连接里下载。      
链接: https://pan.baidu.com/s/1bsgay9iMihPlAKE49aWNTA 提取码: bhee    

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，运行predict.py，输入  
```python
img/timg.jpg
```  
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在retinaface.py文件里面，在如下部分修改model_path和backbone使其对应训练好的文件。  
```python
_defaults = {
    "model_path"        : 'model_data/Retinaface_mobilenet0.25.pth',
    "backbone"          : 'mobilenet',
    "confidence"        : 0.5,
    "nms_iou"           : 0.45,
    "cuda"              : True,
    #----------------------------------------------------------------------#
    #   是否需要进行图像大小限制。
    #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
    #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
    #----------------------------------------------------------------------#
    "input_shape"       : [1280, 1280, 3],
    "letterbox_image"   : True
}
```
3. 运行predict.py，输入  
```python
img/timg.jpg
```  
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

## 训练步骤
1. 本文使用widerface数据集进行训练。  
2. 可通过上述百度网盘下载widerface数据集。  
3. 覆盖根目录下的data文件夹。  
4. 根据自己需要选择**从头开始训练还是在已经训练好的权重下训练**，需要修改train.py文件下的代码，在训练时需要**注意backbone和权重文件的对应**。
使用mobilenet为主干特征提取网络的示例如下：   
从头开始训练需要将pretrained设置为True，并且注释train.py里面的权值载入部分：    
```python
backbone = "mobilenet"
#-------------------------------#
#   是否使用主干特征提取网络
#   的预训练权重
#-------------------------------#
pretrained = True
model = RetinaFace(cfg=cfg, pretrained = pretrained).train()
```
在已经训练好的权重下训练：   
```python
backbone = "mobilenet"
#-------------------------------------------#
#   权值文件的下载请看README
#   权值和主干特征提取网络一定要对应
#-------------------------------------------#
model = RetinaFace(cfg=cfg, pretrained = pretrained).train()
model_path = "model_data/Retinaface_mobilenet0.25.pth"
# 加快模型训练的效率
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print('Finished!')
```
5. 可以在logs文件夹里面获得训练好的权值文件。  

## 评估步骤  
1. 在retinaface.py文件里面，在如下部分修改model_path和backbone使其对应训练好的文件。  
```python
 = {
    "model_path"        : 'model_data/Retinaface_mobilenet0.25.pth',
    "backbone"          : 'mobilenet',
    "confidence"        : 0.5,
    "nms_iou"           : 0.45,
    "cuda"              : True,
    #----------------------------------------------------------------------#
    #   是否需要进行图像大小限制。
    #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
    #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
    #----------------------------------------------------------------------#
    "input_shape"       : [1280, 1280, 3],
    "letterbox_image"   : True
}
```
2. 下载好百度网盘上上传的数据集，其中包括了验证集，解压在根目录下。 
3. 运行evaluation.py即可开始评估。

### Reference
https://github.com/biubug6/Pytorch_Retinaface

