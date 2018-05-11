#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:08:27 2018

@author: yonic
"""

import os
from cleverhans.attacks import FastGradientMethod
from io import BytesIO
import IPython.display
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
import matplotlib.pyplot as plt
import cv2


tensorflow_master = ""
checkpoint_path   = "./input/inception-v3/inception_v3.ckpt"
input_dir_data        = "./data/"
input_dir_images     = os.path.join(input_dir_data,"images")
max_epsilon       = 16.0
image_width       = 299
image_height      = 299
batch_size        = 16

eps = 2.0 * max_epsilon / 255.0
batch_shape = [batch_size, image_height, image_width, 3]
num_classes = 1001

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png'))):
        with tf.gfile.Open(filepath, "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def show_orig_image(fname):
    image = cv2.imread(os.path.join(input_dir,fname))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    

def show_image(a, fmt='png'):
    a = np.uint8((a+1.0)/2.0*255.0)    
    plt.imshow(a)
    
    
