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
    
    
    
def test():
    categories = pd.read_csv(os.path.join(importer.input_dir_data,"categories.csv"))
    image_classes = pd.read_csv(os.path.join(importer.input_dir_data,"images.csv"))
    image_iterator = importer.load_images(importer.input_dir_images, importer.batch_shape)
    
    # get first batch of images
    filenames, images = next(image_iterator)
    image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes,on="ImageId")
    true_classes = image_metadata["TrueLabel"].tolist()
    target_classes = true_labels = image_metadata["TargetClass"].tolist()
    true_classes_names = (pd.DataFrame({"CategoryId": true_classes})
                        .merge(categories, on="CategoryId")["CategoryName"].tolist())
    target_classes_names = (pd.DataFrame({"CategoryId": target_classes})
                          .merge(categories, on="CategoryId")["CategoryName"].tolist())


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
    
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            predicted_classes = sess.run(predicted_labels, feed_dict={x_input: images})
        
        print ("Net output: {}".format(predicted_classes))
            
        predicted_classes_names = (pd.DataFrame({"CategoryId": predicted_classes})
            .merge(categories, on="CategoryId")["CategoryName"].tolist())
            
        for i in range(len(images)):
            print("UNMODIFIED IMAGE (left)",
                  "\n\tPredicted class:", predicted_classes_names[i],
                  "\n\tTrue class:     ", true_classes_names[i])
            
        
        
        
