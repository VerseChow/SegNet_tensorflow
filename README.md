# Image Segmentation based on SegNet 

This repo is to reimplement image segmentation via tensorflow. The training dataset is based on [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

The idea of this reimplementation is based on [SegNet](https://arxiv.org/pdf/1511.00561.pdf)

# Demo Usage
## Training
1. Download and Put [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) Dataset into the Root Folder
2. Run train.sh, it will generate a Dataset/ Folder first which is the data with specific format we need converted from VOC2012 in train Imageset
3. Then if the convertion is done, it will begin training and generate a logs/ Folder which is used for data visualization. You could use command: tensorboad --logdir=./logs/ to see the results of training
## Testing
1. Download and Put [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) Dataset into the Root Folder
2. Run train.sh, it will generate a Dataset/ Folder first which is the data with specific format we need converted from VOC2012 in val Imageset
3. Then the results of testing will be saved in a named test_results/

# Advanced Usage
Please check main.py in scripts/ folder to see the option you could choose to enhance the training results.

