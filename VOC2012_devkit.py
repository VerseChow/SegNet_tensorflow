import os
import sys
import numpy as np
import tensorflow as tf
import argparse

from scipy.misc import imread, imsave

color_map = np.array([[  0,   0,   0], [128,   0,   0], [  0, 128,   0],
                [128, 128,   0], [  0,   0, 128], [128,   0, 128],
                [  0, 128, 128], [128, 128, 128], [ 64,   0,   0],
                [192,   0,   0], [ 64, 128,   0], [192, 128,   0],
                [ 64,   0, 128], [192,   0, 128], [ 64, 128, 128],
                [192, 128, 128], [  0,  64,   0], [128,  64,   0],
                [  0, 192,   0], [128, 192,   0], [  0,  64, 128],
                [224, 224, 192]], dtype=np.uint8)

label_map = {'background':  color_map[0],
            'aeroplane':    color_map[1],
            'bicycle':      color_map[2],
            'bird':         color_map[3],
            'boat':         color_map[4],
            'bottle':       color_map[5],
            'bus':          color_map[6],
            'car':          color_map[7],
            'cat':          color_map[8],
            'chair':        color_map[9],
            'cow':          color_map[10],
            'diningtable':  color_map[11],
            'dog':          color_map[12],
            'horse':        color_map[13],
            'motorbike':    color_map[14],
            'person':       color_map[15],
            'pottedplant':  color_map[16],
            'sheep':        color_map[17],
            'sofa':         color_map[18],
            'train':        color_map[19],
            'tvmonitor':    color_map[20],
            'void':         color_map[21]}

labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def ImageToLabel (image, height, width):
    image = tf.cast(image, dtype=tf.float32)
    seg = tf.equal(image, 0.0)
    for label in range(len(labels[1:])):
        seg_label = tf.equal(image, label+1)
        seg = tf.concat([seg, seg_label], -1)
    seg = tf.cast(seg, dtype=tf.float32)
    seg = tf.reshape(seg, [height, width, len(labels)])
    return seg

def LabelToRGB (image):
    shape = image.get_shape().as_list()
    shape[-1] = 3
    rgb_image = tf.cast(tf.zeros(shape), dtype=tf.uint8)
    rgb_image = tf.Variable(rgb_image)
    image = tf.concat([image, image, image], axis=-1)
    for label in range(len(labels)):
        temp = np.full(shape, 0)
        temp[:,:,:] = color_map[label,:]
        temp = tf.cast(temp, dtype=tf.uint8)
        label_bool = tf.equal(image, label)
        rgb_image = tf.assign(rgb_image,tf.where (label_bool, temp, rgb_image))
    return rgb_image

def PrepareData (voc_dir, set_):
    num_classes = color_map.shape[0] - 1
    if not os.path.isdir('./data/' + set_ + '/labels'):
        os.makedirs('./data/' + set_ + '/labels')
    if not os.path.isdir('./data/' + set_ + '/images'):
        os.makedirs('./data/' + set_ + '/images')
    with open(voc_dir + '/ImageSets/Segmentation/' + set_ + '.txt') as f:
        content = f.readlines()
        for line in content:
            fn = line[:-1]
            print('%s/%s' % (set_, fn))
            img = imread(voc_dir + '/JPEGImages/' + fn + '.jpg', mode='RGB')

            imsave('./data/' + set_ + '/images/' + fn + '.jpg', img)

            lbl = imread(voc_dir + '/SegmentationClass/' + fn + '.png', mode='RGB')
            tmp = np.zeros((lbl.shape[0], lbl.shape[1]), dtype=np.uint8) + 255

            for k in range(num_classes):
                clr = np.int32(color_map[k, :])
                e = lbl - clr[np.newaxis, np.newaxis, :]
                tmp[np.sum(e**2, axis=2) == 0] = k

            imsave('./data/' + set_ + '/labels/' + fn + '.png', tmp)
            
# Convert the VOC2012 dataset into the format we want
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VOC2012_PrepareData')

    parser.add_argument('--voc_dir', dest='voc_dir', help='specify path to VOC2012',
                        default='./VOCdevkit/VOC2012', type=str)
    parser.add_argument('--set_', dest='set_', help='data set to prepare, like train, val or trainval',
                        default='train', type=str)
    config = parser.parse_args()
    PrepareData(config.voc_dir, config.set_)

