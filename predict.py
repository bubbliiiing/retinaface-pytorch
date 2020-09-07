#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from retinaface import Retinaface
import cv2

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
