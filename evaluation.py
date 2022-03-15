import os

import cv2
import tqdm

from retinaface import Retinaface
from utils.utils_map import evaluation

#-------------------------------------------#
#   进行retinaface的map计算
#   需要现在retinaface.py里面修改model_path
#-------------------------------------------#
if __name__ == '__main__':
    mAP_retinaface  = Retinaface(confidence = 0.01, nms_iou = 0.45)
    save_folder     = './widerface_evaluate/widerface_txt/'
    gt_dir          = "./widerface_evaluate/ground_truth/"
    imgs_folder     = './data/widerface/val/images/'
    sub_folders     = os.listdir(imgs_folder)

    test_dataset = []
    for sub_folder in sub_folders:
        image_names = os.listdir(os.path.join(imgs_folder, sub_folder))
        for image_name in image_names:
            test_dataset.append(os.path.join(sub_folder, image_name))

    num_images = len(test_dataset)

    for img_name in tqdm.tqdm(test_dataset):
        image = cv2.imread(os.path.join(imgs_folder, img_name))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results = mAP_retinaface.get_map_txt(image)

        save_name = save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        with open(save_name, "w") as fd:
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(results)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in results:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)
        
    evaluation(save_folder, gt_dir)
