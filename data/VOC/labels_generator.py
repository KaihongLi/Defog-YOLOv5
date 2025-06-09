"""
    Created on Mon Mar 06 2023
    Generate VOC_foggy_test in yolo format.
"""
import os
import shutil

if __name__ == "__main__":
    foggy_images_dir = 'path to your datasets/VOC_YOLO/test/foggy_images/VOC2007'
    labels_dir = 'path to your datasets/VOC_YOLO/test/labels/VOC2007'
    foggy_labels_dir = 'path to your datasets/VOC_YOLO/test/foggy_labels/VOC2007'
    if not os.path.exists(foggy_labels_dir):
        os.makedirs(foggy_labels_dir)
    index = 0
    for images in os.listdir(foggy_images_dir):
        print(images)
        image_index = images.split('_')[0]
        label_path = os.path.join(labels_dir, image_index + '.txt')
        if not os.path.exists(label_path):
            os.remove(os.path.join(foggy_images_dir, images))
            continue
        foggy_label_path = os.path.join(foggy_labels_dir, images.split('.jpg')[0] + '.txt')
        shutil.copy(label_path, foggy_label_path)
        index += 1
    print(f'{index}')
