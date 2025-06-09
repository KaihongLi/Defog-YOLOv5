"""
    Created on Mon Mar 06 2023
    generate images with different levels of fogging offline for hybrid training(Voc_foggy_train and Voc_foggy_val)
    Original from : IA-YOLO(https://github.com/wenyyu/Image-Adaptive-YOLO)
"""
import numpy as np
import os
import cv2
import math
from numba import jit

# only use the image including the labeled instance objects for training
def load_annotations(annot_path):
    print(annot_path)
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations


# get images in folder
def get_all_images(target_dir):
    list_images = os.listdir(target_dir)
    return list_images


# print('*****************Add haze offline***************************')
def parse_annotation(img_name, target_dir, result_dir):
    image_path = str(target_dir + img_name)
    image_name = img_name.rsplit('.', 1)[0]
    image_name_index = img_name.rsplit('.', 1)[1]

    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " % image_path)
    image = cv2.imread(image_path)
    for i in range(10):
        @jit()
        def AddHaz_loop(img_f, center, size, beta, A):
            (row, col, chs) = img_f.shape

            for j in range(row):
                for l in range(col):
                    d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                    td = math.exp(-beta * d)
                    # aaai paper error
                    # td = math.exp(-beta)*d
                    img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
            return img_f

        img_f = image / 255
        (row, col, chs) = image.shape
        # A = 0.5
        A = np.array([0.59986858, 0.56624328, 0.70282524])
        # beta = 0.08
        beta = 0.01 * i + 0.05
        size = math.sqrt(max(row, col))
        center = (row // 2, col // 2)
        foggy_image = AddHaz_loop(img_f, center, size, beta, A)
        img_f = np.clip(foggy_image * 255, 0, 255)
        img_f = img_f.astype(np.uint8)
        img_name = result_dir + image_name + '_' + ("%.2f" % beta) + '.' + image_name_index
        print(img_name)
        cv2.imwrite(img_name, img_f)


if __name__ == '__main__':
    target_dir = 'path to your datasets/VOC_YOLO/train(val)/images/VOC2012(VOC2007)'
    result_dir = 'path to your datasets/VOC_YOLO/train(val)/foggy_images/VOC2012(VOC2007)'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    list_images = get_all_images(target_dir)
    ll = len(list_images)
    for j in range(ll):
        parse_annotation(str(list_images[j]), target_dir, result_dir)
