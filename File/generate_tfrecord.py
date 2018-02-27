#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:28:58 2017

@author: wzg
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import scipy.io as sio


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))
    
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))


root_path = './Data/'
tfrecords_filename = root_path + 'tfrecords/test.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)


height = 300
width = 300
meanfile = sio.loadmat(root_path + 'mats/mean300.mat')
meanvalue = meanfile['mean'] #mean value of images in training set

txtfile = root_path + 'txt/test.txt'
fr = open(txtfile)

for i in fr.readlines():
    item = i.split()
    img = np.float64(misc.imread(root_path + '/images/test_images/' + item[0]))
    img = img - meanvalue
    maskmat = sio.loadmat(root_path + '/mats/test_mats/' + item[1])
    mask = np.float64(maskmat['seg_mask'])
    label = int(item[2])
    img_raw = img.tostring()
    mask_raw = mask.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'name': _bytes_feature(item[0]),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(mask_raw),
        'label': _int64_feature(label)}))
    
    writer.write(example.SerializeToString())
    
writer.close()
fr.close()

################### Test Correctness #####################################
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
i = 0

for string_record in record_iterator:
    if i>0:
        break
    example = tf.train.Example()
    example.ParseFromString(string_record)
    height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])
    
    width = int(example.features.feature['width']
                                .int64_list
                                .value[0])
    
    name = (example.features.feature['name']
                                  .bytes_list
                                  .value[0])
    
    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])
    
    mask_string = (example.features.feature['mask_raw']
                                  .bytes_list
                                  .value[0])
    
    label = (example.features.feature['label']
                                .int64_list
                                .value[0])
    
    img = np.fromstring(img_string, dtype=np.float64)
    mask = np.fromstring(mask_string, dtype=np.float64)
    reconstructed_img = img.reshape((height,width,-1))
    reconstructed_img = reconstructed_img + meanvalue
    reconstructed_mask = mask.reshape((height,width))
    
    print name
    print 'label: ' + str(label)
    plt.subplot(1,2,1)
    plt.imshow(np.uint8(reconstructed_img))
    plt.subplot(1,2,2)
    plt.imshow(np.uint8(reconstructed_mask))
    
    
    i += 1
