import os
import sys
from scipy.misc import imread, imsave
from numpy import *
from util import *

colors = array([[  0,   0,   0], [128,   0,   0], [  0, 128,   0],
                [128, 128,   0], [  0,   0, 128], [128,   0, 128],
                [  0, 128, 128], [128, 128, 128], [ 64,   0,   0],
                [192,   0,   0], [ 64, 128,   0], [192, 128,   0],
                [ 64,   0, 128], [192,   0, 128], [ 64, 128, 128],
                [192, 128, 128], [  0,  64,   0], [128,  64,   0],
                [  0, 192,   0], [128, 192,   0], [  0,  64, 128],
                [224, 224, 192]], dtype=uint8)

num_classes = colors.shape[0] - 1

voc_dir = sys.argv[1] # path/to/VOC201x (without the last '/')
set_ = sys.argv[2] # either `train` or `val`

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
#        data = zeros((512, 512, 3), dtype=uint8)
#        data[:img.shape[0], :img.shape[1], :] = img
#        a0, b0 = divmod(img.shape[0], 256)
#        data[-2, -2, :] = a0
#        data[-2, -1, :] = b0
#
#        a1, b1 = divmod(img.shape[1], 256)
#        data[-1, -2, :] = a1
#        data[-1, -1, :] = b1
        imsave('./data/' + set_ + '/images/' + fn + '.jpg', img)

        lbl = imread(voc_dir + '/SegmentationClass/' + fn + '.png', mode='RGB')
        tmp = zeros((lbl.shape[0], lbl.shape[1]), dtype=uint8) + 255

        for k in range(21):
            clr = int32(colors[k, :])
            e = lbl - clr[newaxis, newaxis, :]
            tmp[sum(e**2, axis=2) == 0] = k

        #data = zeros((512, 512), dtype=uint8) + 255
        #data[:lbl.shape[0], :lbl.shape[1]] = tmp

        imsave('./data/' + set_ + '/labels/' + fn + '.png', tmp)
