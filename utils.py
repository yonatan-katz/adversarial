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
import matplotlib.pyplot as plt
import pandas as pd
from optparse import OptionParser
from scipy.misc import imread
import sys

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
        print(S)
    return S 
        
    
def create_perf_vector(folder):    
    path = os.path.join(folder)    
    dissimilarity = []
    win_loss = []    
    for root, subdirs, files in os.walk(path):        
        for file in sorted(files):                    
            if "stat_eps" in file:                
                with open(os.path.join(root,file), 'r') as f:
                   d = json.load(f) 
                   dissimilarity.append(float(d['dissimilarity']) / 1000.0)#we work with data set of 1000 images!
                   win_loss.append(float(d['win_loss']))
           
    df = pd.DataFrame(win_loss,index=dissimilarity)
    return df.sort_index()

def plot_attack():
    #fgsm = create_perf_vector("output/adversarial/fgsm")
    #ifgsm = create_perf_vector("output/adversarial/ifgsm")
    deep_fool = create_perf_vector("output/adversarial/deep_fool/replicated")
    return deep_fool,
    merged = pd.concat([fgsm,ifgsm,deep_fool],axis=1)
    merged.columns = ['fgsm','ifgsm','deep_fool']
    merged = merged.fillna(method='bfill')
    merged.plot()
    
    
    
#    fig, ax1 = plt.subplots()
#    color = 'tab:red'
#    ax1.set_xlabel('dissimilarity')
#    ax1.set_ylabel('accuracy (%)', color=color)
#    ax1.plot(fgsm.index, fgsm.ix[:,0], color=color)
#    ax1.tick_params(axis='y', labelcolor=color)
#    
#    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#    color = 'tab:blue'
#    ax2.set_ylabel('accuracy (%)', color=color)  # we already handled the x-label with ax1
#    ax2.plot(ifgsm.index, ifgsm.ix[:,0], color=color)
#    ax2.tick_params(axis='y', labelcolor=color)
#    
#    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
if __name__ == "__main__":
   parser = OptionParser()
   parser.add_option("-m","--mode",dest="mode",help="mode:[diff] ")
   parser.add_option("--image_orig",dest="image_orig",type=str,help="image orig for dissimilarity check")    
   parser.add_option("--image_adv",dest="image_adv",type=str,help="image adv for dissimilarity check")    
   
   (options, args) = parser.parse_args()    
   if not options.mode:
       parser.print_help()
       sys.exit(1)        
   
   if options.mode == 'diff':
       print(options.image_orig,options.image_adv)
       image_orig = imread(options.image_orig, mode='RGB').astype(np.float32)*2.0/255.0 - 1.0
       image_adv = imread(options.image_adv, mode='RGB').astype(np.float32)*2.0/255.0 - 1.0
       diff = dissimilariry([image_orig],[image_adv])
       print("Dissimilarity: {}".format(diff))
       
   
       
    #plot_attack()
    
    
    
    