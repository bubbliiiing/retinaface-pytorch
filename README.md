## Retinaface：人脸检测模型在Keras当中的实现
---

### 目录
1. [注意事项 Attention](#注意事项)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

### 注意事项
本库下载过来可以直接进行预测，已经在model_data文件夹下存放了retinaface_mobilenet025.h5文件，可用于预测。  
如果想要使用基于resnet50的retinaface请下载retinaface_resnet50.h5进行预测。  

### 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

### 文件下载
训练所需的retinaface_resnet50.h5、resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5等文件可以在百度云下载。    
数据集也可以在如下连接里下载。    
链接: https://pan.baidu.com/s/1t7-BNsZzHj2isCekc_PVtw     
提取码: 2qrs

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
    "model_path": 'model_data/retinaface_resnet50.h5',
    "backbone": "resnet50",
    "confidence": 0.5,
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
从头开始训练：    
```python
#-------------------------------#
#   创立模型
#-------------------------------#
model = RetinaFace(cfg, backbone=backbone)
model_path = "model_data/mobilenet_2_5_224_tf_no_top.h5"
model.load_weights(model_path,by_name=True,skip_mismatch=True)
```
在已经训练好的权重下训练：   
```python
#-------------------------------#
#   创立模型
#-------------------------------#
model = RetinaFace(cfg, backbone=backbone)
model_path = "model_data/retinaface_mobilenet025.h5"
model.load_weights(model_path,by_name=True,skip_mismatch=True)
```
5、可以在logs文件夹里面获得训练好的权值文件。  

### Reference
https://github.com/biubug6/Pytorch_Retinaface

