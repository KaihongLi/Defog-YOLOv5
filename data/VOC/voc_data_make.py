"""
    Created on Mon Mar 06 2023
    Convert VOC formate dataset to YOLO format.(VOC 2012/2007)
"""
import glob
import os
import xml.etree.ElementTree as ET
import shutil

classes = ['person', 'car', 'bus', 'bicycle', 'motorbike']


def xml2yolo(voc_data_dir, voc_yolo_data_dir):
    for split in ['train', 'test']:
        count = 0
        split_dir = os.path.join(voc_data_dir, split, 'VOCdevkit')
        for dataset in os.listdir(split_dir):
            dataset_dir = os.path.join(split_dir, dataset)
            img_inds_file = os.path.join(dataset_dir, 'ImageSets', 'Main', ('trainval' if split == 'train' else split) + '.txt')
            with open(img_inds_file, 'r') as f:
                txt = f.readlines()
                image_inds = [line.strip() for line in txt]
            anno_dir = os.path.join(dataset_dir, 'Annotations')
            image_dir = os.path.join(dataset_dir, 'JPEGImages')
            label_dir = os.path.join(voc_yolo_data_dir, split, 'labels', dataset)
            yolo_image_dir = os.path.join(voc_yolo_data_dir, split, 'images', dataset)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            if not os.path.exists(yolo_image_dir):
                os.makedirs(yolo_image_dir)
            for anno in image_inds:
                anno_path = os.path.join(anno_dir, anno + '.xml')
                print(anno_path)
                annotations = ''
                with open(anno_path, 'r') as f:
                    tree = ET.parse(anno_path)
                    root = tree.getroot()
                    size = root.find('size')
                    width = int(size.find('width').text.strip())
                    height = int(size.find('height').text.strip())
                    objects = root.findall('object')
                    for obj in objects:
                        difficult = obj.find('difficult').text.strip()
                        if int(difficult) == 1:
                            continue
                        if obj.find('name').text.lower().strip() in classes:
                            class_ind = classes.index(obj.find('name').text.lower().strip())
                            bbox = obj.find('bndbox')
                            xmin = float(bbox.find('xmin').text.strip())
                            xmax = float(bbox.find('xmax').text.strip())
                            ymin = float(bbox.find('ymin').text.strip())
                            ymax = float(bbox.find('ymax').text.strip())
                            x_center = round((xmin + xmax) / 2 / width, 5)
                            y_center = round((ymin + ymax) / 2 / height, 5)
                            w_norm = round((xmax - xmin) / width, 5)
                            h_norm = round((ymax - ymin) / height, 5)
                            annotations += ' '.join(
                                [str(class_ind), str(x_center), str(y_center), str(w_norm), str(h_norm)]) + '\n'
                if annotations:
                    label_path = os.path.join(label_dir, anno.replace('.xml', '.txt'))
                    with open(label_path, 'w') as f:
                        f.write(annotations)
                    image_path = os.path.join(image_dir, anno.replace('.xml', '.jpg'))
                    yolo_image_path = os.path.join(yolo_image_dir, anno.replace('.xml', '.jpg'))
                    shutil.copy(image_path, yolo_image_path)
                    if split == 'test':
                        val_label_path = label_path.replace('test', 'val')
                        val_yolo_image_path = yolo_image_path.replace('test', 'val')
                        shutil.copy(label_path, val_label_path)
                        shutil.copy(yolo_image_path, val_yolo_image_path)
                    count += 1
        print(f'{split}:{count}')


def get_all_images(dataset_dir):
    images = glob.glob(dataset_dir + '/*.jpg')
    return len(images)


if __name__ == '__main__':
    # generate a voc dateset in yolo formate
    voc_data_dir = 'path to your datasets/VOC'
    voc_yolo_data_dir = 'path to your datasets/VOC_YOLO'
    xml2yolo(voc_data_dir, voc_yolo_data_dir)
