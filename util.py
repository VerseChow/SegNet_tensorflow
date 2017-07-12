import tensorflow as tf
import os, time
import argparse
import cv2
import re

from glob import glob
from numpy import *
from lxml import etree
from glob import glob
from numpy import *
from scipy import ndimage
from scipy.ndimage.filters import median_filter
from scipy.misc import imread, imsave, imresize, imshow


numbers = re.compile(r'(\d+)')

class bbox_property():
    def __init__(self, xmin, xmax, ymin, ymax, label):
        self.label = label
        self.xmin = str(xmin)
        self.xmax = str(xmax)
        self.ymin = str(ymin)
        self.ymax = str(ymax)

class path_pack():
    def __init__(self, config):
        self.data_dir = config.train_test_dataset
        self.result_dir = './results'
        self.train_result_dir = './train_results'
        self.resize_image_dir = './'+config.dataset+'/JPEGImages'
        self.xml_path = './'+config.dataset+'/Annotations'
        self.txt_path = './'+config.dataset+'/ImageSets/Main'

    def check_path(self):

        if not os.path.exists(self.train_result_dir):
            os.makedirs(self.train_result_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.resize_image_dir):
            os.makedirs(self.resize_image_dir)
        if not os.path.exists(self.xml_path):
            os.makedirs(self.xml_path)
        if not os.path.exists(self.txt_path):
            os.makedirs(self.txt_path)


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def ReadTextFile(txt_path, image_path, ext):
    with open(txt_path) as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    if image_path[-1] is '/':
        return [image_path+'/'+x+'.'+ext for x in content]
    if image_path[-1] is not '/':
        return [image_path+'/'+x+'.'+ext for x in content]
        
def str2bool(parameter):
    if parameter.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if parameter.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')            

def bbox_generate(image):

    mask = image>0
    max_index = 0
    max_size = 0

    label_im, nb_labels = ndimage.label(mask)

    # Find the largest connect component
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))#sizes of connected component. a lists

    mask_size = sizes < 1000
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = unique(label_im)

    for i in labels:
        if i > 0:
            set_size = sum(label_im==i)
            if set_size>max_size:
                max_index = i
                max_size = set_size

    # Now that we have only one connect component, extract it's bounding box
    slice_x, slice_y = ndimage.find_objects(label_im == max_index)[0] #find the largest one

    return slice_x, slice_y 

def write_txt(datapath, writepath, set_name, label):

    if not os.path.exists(writepath):
            os.makedirs(writepath)
    

    if set_name == 'train':
        im_list = sorted(glob(datapath+'/*'+label+'*.jpg'), key=numericalSort)
        
        with open(writepath+'/'+set_name+'.txt', 'a+') as f:
            for im in im_list:
                im = os.path.basename(im)
                im = os.path.splitext(im)[0]
                f.write(im+'\n')
    elif set_name == 'test':
        with open(writepath+'/'+set_name+'.txt', 'a+') as f:
            f.write('\n')
    else:
        try:
            assert set_name == 'train' or set_name == 'test'
        except ValueError:
            print 'set_name must be train or test!!'





def write_xml(file_name, writepath, bbox):

    if not os.path.exists(writepath):
            os.makedirs(writepath)

    label = bbox.label

    xml_file_name = os.path.basename(file_name)
    xml_file_name = os.path.splitext(xml_file_name)[0]+'.xml'
    with open(writepath +'/'+ xml_file_name, 'w') as out:

        img = imread(file_name)
        print img.shape

        root = etree.Element('annotation')
        chd_folder = etree.Element('folder')
        chd_folder.text = 'progress'
        root.append(chd_folder)
        chd_fname = etree.Element('filename')
        chd_fname.text = os.path.basename(file_name)
        root.append(chd_fname)

        chd_size = etree.Element('size')
        chd_size_width = etree.Element('width')
        chd_size_width.text = str(img.shape[1]) 
        chd_size_height = etree.Element('height')
        chd_size_height.text = str(img.shape[0])
        chd_size_depth = etree.Element('depth')
        chd_size_depth.text = str(img.shape[2])
        chd_size.append(chd_size_width)
        chd_size.append(chd_size_height)
        chd_size.append(chd_size_depth)         
        root.append(chd_size)

        chd_obj = etree.Element('object')
        chd_obj_name = etree.Element('name')
        chd_obj_name.text = bbox.label
        chd_obj.append(chd_obj_name)
        chd_obj_bbox = etree.Element('bndbox')
        chd_obj_bbox_xmin = etree.Element('xmin')   
        chd_obj_bbox_xmin.text = bbox.xmin
        chd_obj_bbox.append(chd_obj_bbox_xmin)
        chd_obj_bbox_ymin = etree.Element('ymin')   
        chd_obj_bbox_ymin.text = bbox.ymin
        chd_obj_bbox.append(chd_obj_bbox_ymin)
        chd_obj_bbox_xmax = etree.Element('xmax')   
        chd_obj_bbox_xmax.text = bbox.xmax 
        chd_obj_bbox.append(chd_obj_bbox_xmax)
        chd_obj_bbox_ymax = etree.Element('ymax')   
        chd_obj_bbox_ymax.text = bbox.ymax 
        chd_obj_bbox.append(chd_obj_bbox_ymax)
        chd_obj.append(chd_obj_bbox)
        root.append(chd_obj)


        s = etree.tostring(root, pretty_print=True)

        print s
        out.write(s)

        print('Writing Done!')
