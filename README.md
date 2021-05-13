# YOLO
Exploration of object detection architectures 
# To get dataset for PAL VOC 2007, run: 
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar  <br />
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar 
# VOC2012 DATASET                                                              
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar  <br />
!tar xf VOCtrainval_11-May-2012.tar  <br />
!tar xf VOCtrainval_06-Nov-2007.tar  <br />
!tar xf VOCtest_06-Nov-2007.tar  <br />
!wget https://pjreddie.com/media/files/voc_label.py  <br />
!python voc_label.py   <br />

#Or run:
!pip install -q kaggle  <br />
!mkdir ~/.kaggle  <br />
# upload your kaggle permission file
!cp kaggle.json ~/.kaggle/  <br />
!chmod 600 ~/.kaggle/kaggle.json  <br /> 
!kaggle datasets download -d aladdinpersson/pascal-voc-dataset-used-in-yolov3-video  <br />

import zipfile  <br />
zip_ref = zipfile.ZipFile('pascal-voc-dataset-used-in-yolov3-video.zip', 'r')  <br />
zip_ref.extractall()  <br />
zip_ref.close()  <br />

# To get BCCD 2021 Dataset, run:
!git clone 'https://github.com/Shenggan/BCCD_Dataset.git'


# update albumentation package 
!pip install -U albumentations

# Key difference between YOLOv3 and YOLOv1 
Darknet53 as backnone  <br />
obj_p, class_p: obj, class.  <br />
x,y - center coor. Relative to the cell.  <br />
w,h - width, height. They can be greater than 1.  <br />

What's new in YOLOv3 compare to YOLOv1?  <br />
Three classifier branches:  <br />
This means we output three different bounding boxes for each cell.  <br />
13 * 13 for large object  <br />
26 * 26 for mid size  <br />
52 * 52 small size  <br />
Residual Connection, Concatenation, upsampling  <br />

Anchor Boxes(Also in YOLOv2):  <br />
Encode previous knowledge into the model. Maybe one for vertical object, one for horzonital object. Can the model slightly adjust the anchor box to fit the image instead of predicting everything completely new. This should be much easier. We have three difference anchor boxes for each grid for each branch. 9 in total.

How do we can the anchor boxes?  <br />
kmeas, max iou. Not covered in this project. We are going to take it from the paper.

So How do we make predications in YOLOv3?  
b_x = sigmoid(t_x)  <br />
b_y = sigmoid(t_y)  <br />
b_w = p_w * e^t_w  <br />
b_h = p_h * e^t_h  <br />
p_w and p_h are anchor boxes  <br />
