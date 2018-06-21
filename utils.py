#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 20:37:11 2018

@author: yonic
"""

import numpy as np
import adversarial.importer as importer
import pandas as pd
import os
import json
from scipy.misc import imsave
import matplotlib.pyplot as plt
import pandas as pd
from optparse import OptionParser
from scipy.misc import imread
import sys
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

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
                   dissimilarity.append(float(d['dissimilarity']))#we work with data set of 1000 images!
                   win_loss.append(float(d['win_loss']))
           
    df = pd.DataFrame(win_loss,index=dissimilarity)
    return df.sort_index()


def pred(full_image_path):
    images = np.zeros(importer.batch_shape,np.float32)
    image = imread(full_image_path, mode='RGB').astype(np.float32)*2.0/255.0 - 1.0
    images[0, :, :, :] = image
    graph_eval = tf.Graph()
    with graph_eval.as_default():    
        x_input = tf.placeholder(tf.float32, shape=importer.batch_shape)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(x_input, num_classes=importer.num_classes, is_training=False)
        
        predicted_labels = tf.argmax(end_points['Predictions'], 1)       
        session_creator = tf.train.ChiefSessionCreator(
                          checkpoint_filename_with_path=importer.checkpoint_path,
                          master=importer.tensorflow_master)
    
        #TODO: I run out of memory when loading all images to the memory,
        #      Make one image prediction in each iteration
        #image_iterator = importer.load_images_generator(importer.input_dir_images, importer.batch_shape)                              
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            filename = os.path.basename(full_image_path)
            true_classes = importer.filename_to_class([filename])
            predicted_classes = sess.run(predicted_labels, feed_dict={x_input: images})
            print("True class: {}, predicted: {}".format(true_classes,predicted_classes))
            #(true_classes==predicted_classes)[0])

def plot_attack():
    fgsm = create_perf_vector("output/adversarial/fgsm")
    ifgsm = create_perf_vector("output/adversarial/ifgsm")
    #deep_fool = create_perf_vector("output/adversarial/deep_fool/replicated")    
    m = pd.concat([fgsm,ifgsm],axis=1)
    m.columns = ['fgsm','ifgsm']    
    M = m[m.index<0.1]    
    M.fgsm.dropna().plot()    
    M.ifgsm.dropna().plot()
    plt.legend(['fgsm','ifgsm']) 
    plt.show()
    
if __name__ == "__main__":
   parser = OptionParser()
   parser.add_option("-m","--mode",dest="mode",help="mode:[diff,pred,viz] ")
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
      
   elif options.mode == 'pred':
       pred(options.image_orig)
      
   elif  options.mode == 'viz':
       image_orig = imread(options.image_orig, mode='RGB').astype(np.float32)*2.0/255.0 - 1.0
       image_adv = imread(options.image_adv, mode='RGB').astype(np.float32)*2.0/255.0 - 1.0
       diff_image = make_png_image(image_orig - image_adv)
       importer.show_image(diff_image)
       
       
  
       
   
       
    #plot_attack()
    
    
    
    