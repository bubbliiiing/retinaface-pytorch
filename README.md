## Retinaface：人脸检测模型在Pytorch当中的实现
---

### 目录
1. [注意事项 Attention](#注意事项)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

### 注意事项
本库下载过来可以直接进行预测，已经在model_data文件夹下存放了Retinaface_mobilenet0.25.pth文件，可用于预测。  
如果想要使用基于resnet50的retinaface请下载Retinaface_resnet50.pth进行预测。  

### 所需环境
pytorch==1.2.0

### 文件下载
训练所需的Retinaface_resnet50.pth等文件可以在百度云下载。    
数据集也可以在如下连接里下载。    
链接: https://pan.baidu.com/s/1q2E6uWs0R5GU_PFs9_vglg    
提取码: z7es 

### 预测步骤
#### 1、使用预训练权重
a、下载完库后解压，运行predict.py，输入  
```python
img/timg.jpg
```
可完成预测。  
b、利用video.py可进行摄像头检测。  
#### 2、使用自己训练的权重
a、按照训练步骤训练。  
b、在retinaface.py文件里面，在如下部分修改model_path和backbone使其对应训练好的文件。  
```python
_defaults = {
    "model_path": 'model_data/Retinaface_mobilenet0.25.pth',
    "confidence": 0.5,
    "backbone": "mobilenet",
    "cuda": True
}
```
c、运行predict.py，输入  
```python
img/timg.jpg
```
可完成预测。  
d、利用video.py可进行摄像头检测。  

### 训练步骤
1、本文使用widerface数据集进行训练。  
2、可通过上述百度网盘下载widerface数据集。  
3、覆盖根目录下的data文件夹。  
4、根据自己需要选择**从头开始训练还是在已经训练好的权重下训练**，需要修改train.py文件下的代码，在训练时需要**注意backbone和权重文件的对应**。
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
5、可以在logs文件夹里面获得训练好的权值文件。  

### Reference
https://github.com/biubug6/Pytorch_Retinaface

