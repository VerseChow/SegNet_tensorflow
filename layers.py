import tensorflow as tf

from numpy import *

vgg_weights = load('vgg16.npy', encoding='latin1').item()

def conv_relu_vgg(x, name='conv_vgg', reuse=None, training=True):
    kernel = vgg_weights[name][0]
    bias = vgg_weights[name][1]
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[0],
                padding='same', use_bias=True, reuse=reuse,
                kernel_initializer=tf.constant_initializer(kernel),
                bias_initializer=tf.constant_initializer(bias),
                name='conv2d', trainable = training)
        x = tf.layers.batch_normalization(x, name='batchnorm')
        return tf.nn.relu(x, name='relu')

def conv_relu(x, num_filters, ksize=3, stride=2, name='upconv', reuse=None, training=True):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, num_filters, ksize, stride,
                padding='same', use_bias=False, reuse=reuse,
                name='conv2d', trainable=training)
        x = tf.layers.batch_normalization(x, name='batchnorm')
        return tf.nn.relu(x, name='relu')

def upconv_relu(x, num_filters, ksize=3, stride=2, name='upconv', reuse=None, training=True):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, num_filters, ksize, stride,
                padding='same', use_bias=False, reuse=reuse,
                name='conv2d_transpose', trainable = training)
        x = tf.layers.batch_normalization(x, name='batchnorm')
        return tf.nn.relu(x, name='relu')

def max_pooling(x, ksize=2, strides=2, padding='SAME', name='maxpool'):
    with tf.variable_scope(name):
        x, argmax_indices = tf.nn.max_pool_with_argmax(x, ksize=[1,ksize,ksize,1],
                            strides=[1,strides,strides,1], padding=padding,
                            name=name)
        return x, argmax_indices

def up_sample(x, argmax, ksize=[1, 2, 2, 1], name='upsample'):
    with tf.variable_scope(name):
        input_shape = x.get_shape().as_list()
        #  calculation new shape
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        # calculation indices for batch, height, width and feature maps
    
        one_like_mask = tf.ones_like(argmax)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
        b = one_like_mask * batch_range
        h = argmax // (output_shape[2] * output_shape[3])
        w = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int64)
        f = one_like_mask * feature_range
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(x)
        indices = tf.transpose(tf.reshape(tf.stack([b, h, w, f]), [4, updates_size]))
        values = tf.reshape(x, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret