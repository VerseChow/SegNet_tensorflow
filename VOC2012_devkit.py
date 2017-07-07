import numpy as np
import tensorflow as tf


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
    #seg = tf.subtract(image, label_map['background'])
    #seg = tf.reduce_sum(seg, 2, keep_dims=True)
    seg = tf.equal(image, 0.0)
    #seg = tf.reshape(seg, [height, width, 1])
    for label in range(len(labels[1:])):
        #seg_label = tf.subtract(image, label_map[label])
        #seg_label = tf.reduce_sum(seg_label, 2, keep_dims=True)
        seg_label = tf.equal(image, label+1)
        #seg_label = tf.reshape(seg_label, [height, width, 1])
        seg = tf.concat([seg, seg_label], -1)
    seg = tf.cast(seg, dtype=tf.float32)
    seg = tf.reshape(seg, [height, width, len(labels)])
    return seg

def GrayToRGB (image):
    shape = image.get_shape().as_list()
    shape[-1] = 3
    rgb_image = tf.cast(tf.zeros(shape), dtype=tf.uint8)
    rgb_image = tf.Variable(rgb_image)
    image = tf.concat([image, image, image], axis=-1)
    for label in range(len(labels)):
        temp = np.full(shape, 0)
        temp[:,:,:] = color_map[label,:]
        temp = tf.cast(temp, dtype=tf.uint8)
        #temp = tf.Variable(temp)
        label_bool = tf.equal(image, label)
        rgb_image = tf.assign(rgb_image,tf.where (label_bool, temp, rgb_image))
    return rgb_image


