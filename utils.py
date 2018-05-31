#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 20:37:11 2018

@author: yonic
"""

import numpy as np
import glob
import pandas as pd
import os
import json
from scipy.misc import imsave

'''Calculate euclidian distance between two image arrays
'''
def image_saver(images,fnames,path):
    for image,fname in zip(images,fnames):
        png =  make_png_image(image)
        ffname = os.path.join(path,fname)
        imsave(ffname,png)

def make_png_image(a):
    a = np.uint8((a+1.0)/2.0*255.0)        
    return a     

def dissimilariry(orig_images,adv_images):
    S = 0
    for orig,adversal in zip(orig_images,adv_images):
        S += np.linalg.norm([orig-adversal])/np.linalg.norm([orig])
    return S
     
        
    
def create_perf_vector(folder):    
    pattern = os.path.join(folder,"stat_eps*")    
    dissimilarity = []
    win_loss = []
    for fname in glob.glob(pattern):        
        with open(fname, 'r') as file:
           d = json.load(file) 
           dissimilarity.append(float(d['dissimilarity']) / 1000.0)#we work with data set of 1000 images!
           win_loss.append(float(d['win_loss']))
           
    df = pd.DataFrame(win_loss,index=dissimilarity)
    return df.sort_index()
    