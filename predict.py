'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用cv2.imread打开图片文件进行预测。
2、如果想要保存，利用cv2.imwrite("img.jpg", r_image)即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取(b[0], b[1]), (b[2], b[3])这四个值。
4、如果想要截取下目标，可以利用获取到的(b[0], b[1]), (b[2], b[3])这四个值在原图上利用矩阵的方式进行截取。
'''
import cv2

from retinaface import Retinaface

retinaface = Retinaface()

while True:
    img = input('Input image filename:')

    image = cv2.imread(img)
    if image is None:
        print('Open Error! Try again!')
        continue
    else:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        r_image = retinaface.detect_image(image)
        r_image = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
        cv2.imshow("after",r_image)
        cv2.waitKey(0)
