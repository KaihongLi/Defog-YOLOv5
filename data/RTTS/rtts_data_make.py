"""
    Created on Mon Mar 06 2023
    Convert VOC formate dataset to YOLO format.(RTTS)
"""
import glob
import os
import xml.etree.ElementTree as ET
import shutil

classes = ['person', 'car', 'bus', 'bicycle', 'motorbike']


def xml2yolo(rtts_data_dir, rtts_yolo_data_dir):
    count = 0
    anno_dir = os.path.join(rtts_data_dir, 'annotations_xml')
    image_dir = os.path.join(rtts_data_dir, 'JPEGImages')
    label_dir = os.path.join(rtts_yolo_data_dir, 'labels')
    yolo_image_dir = os.path.join(rtts_yolo_data_dir, 'images')
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(yolo_image_dir):
        os.makedirs(yolo_image_dir)
    for anno in os.listdir(anno_dir):
        anno_path = os.path.join(anno_dir, anno)
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
            image_path = os.path.join(image_dir, anno.replace('.xml', '.png'))
            yolo_image_path = os.path.join(yolo_image_dir, anno.replace('.xml', '.png'))
            shutil.copy(image_path, yolo_image_path)
            count += 1
    print(f'images:{count}')


if __name__ == '__main__':
    voc_data_dir = 'path to your datasets/RTTS'
    voc_yolo_data_dir = 'path to your datasets/RTTS_YOLO'
    xml2yolo(voc_data_dir, voc_yolo_data_dir)
