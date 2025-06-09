# Defog-YOLOv5 (Image-Apaptive YOLO for Object Detection in Adverse Weather Conditions)
This repository is an [Image-Adaptive-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO) implementation based on the YOLOv5 Pytroch version.

<div align="center">

  ![VOC_Comparsion](results/VOC/voc_comparsion.png)

  <p align="center">Fig. 1. Visual comparison of related methods on the VOC dataset.</p>

</div>

## Introduction
- The original [Image-Adaptive-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO) was implemented based on Tensorflow version of YOLOv3, and this repository reproduces the [Image-Adaptive-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO) algorithm based on Pytorch version of YOLOv5. 
- The DIP(differentiable image process module) consists of six different filters with adjustable hyperparameters (Defog, White Balance(WB), Gamma, Contrast, Tone and Sharpen) for processing images in adverse weather conditions.
- Joint learning of CNN-PP and YOLOv5, which ensures that CNN-PP can learn an appropriate DIP to enhance the image for detection in a weakly supervised manner.

## Quick Start
### Installation
```shell
$ git clone https://github.com/KaihongLi/Defog-YOLOv5.git
$ cd Defog-YOLOv5
$ conda ceate -n Defog-YOLO python=3.9.13
$ conda activate Defog-YOLO
$ pip install -r requirements.txt
```

### Prepare Datasets
#### VOC Dataset
1. Refer to the official version of [Image-Adaptive-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO) and organise it as follows:

```bashrc
VOC           # path:  path to your datasets/VOC
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)
```

2. Convert VOC formate dataset to YOLOv5 format.
```shell
$ python data/VOC/voc_data_make.py
```
```bashrc
VOC_YOLO           # path:  path to your datasets/VOC_YOLO
├── train
|    └──images
|    └──labels
└── val
|    └──images
|    └──labels
└── test
     └──images
     └──labels
```

3. Generate VOC_foggy_train, VOC_foggy_val dataset and VOC_foggy_test labels offline.
```shell
$ python defog/foggy_data_make.py
# put Official Voc_foggy_test dataset in VOC_YOLO/test/foggy_images
$ python VOC/labels_generator.py
```
```bashrc
VOC_YOLO           # path:  path to your datasets/VOC_YOLO
├── train
|    └──images
|    └──foggy_images
|    └──labels
└── val
|    └──images
|    └──foggy_images
|    └──labels
└── test
|    └──images
|    └──foggy_images
|    └──labels
|    └──foggy_labels
```

#### RTTS Datasets
1. Refer to the official version of [Image-Adaptive-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO) and organise it as follows:
```bashrc
RTTS         # path:  path to your datasets/RTTS
├── annotations_json
├── annotations_xml
├── ImageSets
└── JPEGImages
```
2. Convert RTTS VOC formate dataset to YOLOv5 format.
```shell
$ python datartts/rtts_data_make.py
```
```bashrc
RTTS_YOLO           # path:  path to your datasets/RTTS_YOLO
├── images
└── labels
```

### Training:
Joint learning of CNN-PP and YOLOv5(Hybrid Data Training).
```shell
$ python defog_train.py --weights yolov5s.pt --cfg models/yolov5s_filter.yaml --data data/VOC/VOC.yaml --batch-size 16
```
Only hybrid data training.
```shell
$ python hybrid_train.py --weights yolov5s.pt --cfg models/yolov5s.yaml --data data/VOC/VOC.yaml --batch-size 16
```

### Evaluation:
Evaluating the performance of the Defog-YOLOv5 model.
```shell
# hybrid val
$ python defog_val.py --weights runs/train/defog_yolov5s/weights/best.pt --data data/VOC/VOC.yaml or data/RTTS/RTTS.yaml --batch-size 16
# normal val
$ python defog_val_test.py --weights runs/train/defog_yolov5s/weights/best.pt --data data/VOC/VOC.yaml or data/RTTS/RTTS.yaml --batch-size 16
```
Evaluating the performance of the Hybrid Data Training YOLOv5 model.
```shell
# hybrid val
$ python hybrid_val.py --weights runs/train/hybrid_yolov5s/weights/best.pt --data data/VOC/VOC.yaml or data/RTTS/RTTS.yaml --batch-size 16
# normal val
$ python val.py --weights runs/train/hybrid_yolov5s/weights/best.pt --data data/VOC/VOC.yaml or data/RTTS/RTTS.yaml --batch-size 16
```

### Visualization
Visualise the results produced by the Defog-YOLOv5 model.
```shell
$ python defog_detect.py --weights runs/train/defog_yolov5s/weights/best.pt --source path to your datasets/VOC_YOLO/test/images --data data/VOC/VOC.yaml or data/RTTS/RTTS.yaml
```
Visualise the results produced by tthe Hybrid Data Training YOLOv5 model.
```shell
$ python hybrid_val.py --weights runs/train/hybrid_yolov5s/weights/best.pt --source path to your datasets/VOC_YOLO/test/images --data data/VOC/VOC.yaml or data/RTTS/RTTS.yaml
```

## Main Result
<br/>

<p align="center"><strong>VOC</strong></p>

<br/>

<div align="center">

  ![VOC_Comparsion](results/VOC/voc_comparsion.png)

  <p align="center">Fig. 1. Visual comparison of related methods on the VOC dataset.</p>

</div>

## Acknowledgements
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Image-Adaptive-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO)