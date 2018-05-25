#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 14:23:07 2018

@author: yonic
"""

'''Evaluate model accuracy on the ImageNet original data set
'''
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import adversarial.importer as importer
import adversarial.models as models
import adversarial.utils as utils
import pandas as pd
import numpy as np
import os
import sys
import json
import adversarial.config as config
import sys
from optparse import OptionParser


slim = tf.contrib.slim



def evaluate_model(image_iterator):    
    accuracy_vector = []
    graph_eavl = tf.Graph()
    print ("evaluate_model graph is ready!")
    S = 0
    filenames, adv_images,orig_images = next(image_iterator,(None,None))
    S += utils.dissimilariry(orig_images,adv_images)
    with graph_eavl.as_default():    
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
        counter = 1
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            while True:                    
                if filenames is None: break                    
                true_classes = importer.filename_to_class(filenames)
                predicted_classes = sess.run(predicted_labels, feed_dict={x_input: adv_images})            
                accuracy_vector.append((true_classes==predicted_classes)[0])
                print("Pricessing image num:{}".format(counter))                            
                sys.stdout.flush()                
                filenames, adv_images, orig_images = next(image_iterator,(None,None))
                S += utils.dissimilariry(orig_images,adv_images)
                counter +=1 
                if counter > 999:
                    break
                            
                
    
        
        
    true_labels = np.sum(accuracy_vector)
    false_labels = len(accuracy_vector) - true_labels
    win_loss = (np.float32(true_labels) / len(accuracy_vector)) * 100.0
    dissimilarity = S
    print ("dissimilarity:{},true labels:{}, false labels:{},win_loss:{}".format(S,true_labels,false_labels,win_loss))        
    return win_loss,dissimilarity
    

def main():   
    parser = OptionParser()
    parser.add_option("-m","--mode",dest="mode",help="mode:[orig,fgsm,manual]")    
    parser.add_option("-d","--dir",dest="directory",help="directory contains images for testing")    
    parser.add_option("--eps",dest="eps",help="FGSM eps parameter",type=float,default=importer.eps)    
    
    (options, args) = parser.parse_args()    
    
    if not options.mode:
        parser.print_help()
        sys.exit(1)
        
    print ("Options:{}".format(options))    
    if options.mode == "orig":
        generator = importer.load_images_generator(importer.batch_shape)
    elif options.mode == "fgsm":
        folder_path = os.path.join(config.ADVERSARIAL_FOLDER,"fgsm")
        os.makedirs(folder_path,exist_ok=True)
        generator = models.fgsm_generator(importer.batch_shape, eps=options.eps,is_return_orig_images=True)
    elif options.mode == "manual":
        importer.input_dir_images = options.directory
        generator = importer.load_images_generator(importer.batch_shape)        
        
    win_loss,dissimilarity = evaluate_model(generator)    
    fname = os.path.join(folder_path,"stat_eps_{}".format(options.eps))
    stat = {"win_loss":win_loss,"dissimilarity":dissimilarity}
    with open(fname, 'w') as file:
        file.write(json.dumps(stat))     
    
    
if __name__== "__main__":
    main()
    
    

    
    