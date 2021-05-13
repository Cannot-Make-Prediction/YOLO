# YOLO
Exploration of object detection architectures 
# To get dataset for PAL VOC 2007, run: 
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar\
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar 
# VOC2012 DATASET                                                              
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar xf VOCtrainval_11-May-2012.tar
!tar xf VOCtrainval_06-Nov-2007.tar
!tar xf VOCtest_06-Nov-2007.tar
!wget https://pjreddie.com/media/files/voc_label.py
!python voc_label.py

#Or run:
!pip install -q kaggle
!mkdir ~/.kaggle
# upload your kaggle permission file
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d aladdinpersson/pascal-voc-dataset-used-in-yolov3-video

import zipfile
zip_ref = zipfile.ZipFile('pascal-voc-dataset-used-in-yolov3-video.zip', 'r')
zip_ref.extractall()
zip_ref.close()

# To get BCCD 2021 Dataset, run:
!git clone 'https://github.com/Shenggan/BCCD_Dataset.git'


# update albumentation package 
!pip install -U albumentations

# Key difference between YOLOv3 and YOLOv1 
Darknet53 as backnone
obj_p, class_p: obj, class.
x,y - center coor. Relative to the cell.
w,h - width, height. They can be greater than 1.

What's new in YOLOv3 compare to YOLOv1?
Three classifier branches:
This means we output three different bounding boxes for each cell.
13 * 13 for large object
26 * 26 for mid size
52 * 52 small size
Residual Connection, Concatenation, upsampling

Anchor Boxes(Also in YOLOv2):
Encode previous knowledge into the model. Maybe one for vertical object, one for horzonital object. Can the model slightly adjust the anchor box to fit the image instead of predicting everything completely new. This should be much easier. We have three difference anchor boxes for each grid for each branch. 9 in total.

How do we can the anchor boxes?
kmeas, max iou. Not covered in this project. We are going to take it from the paper.

So How do we make predications in YOLOv3?
b_x = sigmoid(t_x)
b_y = sigmoid(t_y)
b_w = p_w * e^t_w
b_h = p_h * e^t_h
p_w and p_h are anchor boxes
