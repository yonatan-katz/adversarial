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
import adversarial.attack_replicator as attack_replicator
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
from functools import partial

slim = tf.contrib.slim

def evaluate_model(image_iterator,image_saver):    
    accuracy_vector = []
    graph_eavl = tf.Graph()
    print ("evaluate_model graph is ready!")
    S = 0
    filenames, adv_images,orig_images = next(image_iterator,(None,None))
    image_saver(orig_images,filenames)
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
                image_saver(adv_images,filenames)
                S += utils.dissimilariry(orig_images,adv_images)
                counter +=1 
                if counter > 999:                
                    break  
        
    true_labels = np.sum(accuracy_vector)
    false_labels = len(accuracy_vector) - true_labels
    win_loss = (np.float32(true_labels) / len(accuracy_vector)) * 100.0
    dissimilarity = S / counter
    print ("dissimilarity:{},true labels:{}, false labels:{},win_loss:{}".format(S,true_labels,false_labels,win_loss))        
    return win_loss,dissimilarity
    

def main():   
    parser = OptionParser()
    parser.add_option("-m","--mode",dest="mode",help="mode:[fgsm,ifgsm,deep_fool,carlini_wagner,manual], manual mode is just evalute inception model vs given image folder")    
    parser.add_option("-r","--replicate",dest="replicate",help="replicate finished attack with different eps, is applied to deep_fool and carlini wagner attacks only")
    parser.add_option("--eps",dest="eps",help="eps parameter",type=float,default=0.0)
    
    (options, args) = parser.parse_args()    
    
    if not options.mode:
        parser.print_help()
        sys.exit(1)        
    
    folder_path = os.path.join(config.ADVERSARIAL_FOLDER,options.mode,str(options.eps))
    os.makedirs(folder_path,exist_ok=True)
    if options.replicate is not None:
        print("Attack eps: {}".format(options.eps))
        generator = attack_replicator.replicate_attack(options.mode,options.eps)
    else:
        if options.mode == "fgsm" or options.mode == "ifgsm":
            generator = models.adversarial_generator_basic(options.mode, importer.batch_shape,eps=options.eps,is_return_orig_images=True)
        elif options.mode == "deep_fool" or options.mode == 'carlini_wagner':
            generator = models.adversarial_generator_deep_fool(options.mode, importer.batch_shape,eps=options.eps,is_return_orig_images=True)
        else:
            raise Exception("Bad attack mode!")
        
        
    image_saver = partial(utils.image_saver,path=folder_path)
    win_loss,dissimilarity = evaluate_model(generator,image_saver)
    fname = os.path.join(folder_path,"stat_eps_{}".format(options.eps))
    stat = {"win_loss":win_loss,"dissimilarity":dissimilarity}
    with open(fname, 'w') as file:
        file.write(json.dumps(stat))     
    
    
if __name__== "__main__":
    main()
    
    

    
    