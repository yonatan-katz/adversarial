#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 21:36:44 2018

@author: yonic
"""
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import adverserial_2017.importer as importer
import pandas as pd
import numpy as np
import os


slim = tf.contrib.slim

class InceptionModel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                            x_input, num_classes=self.num_classes, is_training=False,
                            reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        probs = output.op.inputs[0]
        return probs
    
    
'''Evaluate model accuracy on the ImageNet original data set
'''
def evaluate_orig_test():    
    image_classes = pd.read_csv(os.path.join(importer.input_dir_data,"images.csv"))    
    accuracy_vector = []

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=importer.batch_shape)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(x_input, num_classes=importer.num_classes, is_training=False)
        
        predicted_labels = tf.argmax(end_points['Predictions'], 1) 
    

        #saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(                          
                          #scaffold=tf.train.Scaffold(saver=saver),
                          checkpoint_filename_with_path=importer.checkpoint_path,
                          master=importer.tensorflow_master)
    
        #TODO: I roun out of memory when loading all images to the memory,
        #      Make one image prediction in each iteration
        image_iterator = importer.load_images(importer.input_dir_images, importer.batch_shape)                      
        counter = 0        
        while True:                    
            with tf.train.MonitoredSession(session_creator=session_creator) as sess:
                while True:
                    filenames, images = next(image_iterator,(None,None))                    
                    if filenames is None: break
                    image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes,on="ImageId")
                    true_classes = image_metadata["TrueLabel"].tolist()
                    predicted_classes = sess.run(predicted_labels, feed_dict={x_input: images})            
                    accuracy_vector.append((true_classes==predicted_classes)[0])
                    print("Pricessing image num:{}".format(counter))            
                    counter += 1
                    if counter >10:
                        break
            break            
            
        
        return accuracy_vector
            
                
            
            
        
        
        
