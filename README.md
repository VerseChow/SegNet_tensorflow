# Image Segmentation based on SegNet 

This repo is to reimplement image segmentation via tensorflow. The training dataset is based on [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

The idea of this reimplementation is based on [SegNet](https://arxiv.org/pdf/1511.00561.pdf)

# Demo Usage
## Training
1. Download and Put [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) Dataset into the `Root` Folder
2. Run ```train.sh```, it will generate a `Dataset/` Folder first which is the data with specific format we need converted from VOC2012 in train Imageset
3. Then if the convertion is done, it will begin training and generate a `logs/` Folder which is used for data visualization. You could use command: ```tensorboad --logdir=./logs/ to see the results of training```
## Testing
1. Download and Put [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) Dataset into the Root Folder
2. Run ```train.sh```, it will generate a `Dataset/` Folder first which is the data with specific format we need converted from VOC2012 in val Imageset
3. Then the results of testing will be saved in a named `test_results/`

# Advanced Usage
Please check `main.py` in `scripts/` folder to see the option you could choose to enhance the training results.

# Demo Picture
Following pictures are the results of SegNet. Left hand side is image, right hand side is label.

![segnet1](https://user-images.githubusercontent.com/22964197/28596219-6f5e5c16-7165-11e7-9012-877a7c17adbe.png)
![segnet2](https://user-images.githubusercontent.com/22964197/28596095-fae3f6a2-7164-11e7-8675-b84731e3e38c.png)
![segnet4](https://user-images.githubusercontent.com/22964197/28596118-11ea59ae-7165-11e7-8892-b41140f5425b.png)
![segnet3](https://user-images.githubusercontent.com/22964197/28596147-3328044a-7165-11e7-9857-723890cff8a9.png)
