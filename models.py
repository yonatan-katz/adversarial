#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 21:36:44 2018

@author: yonic
"""
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import DeepFool
from cleverhans.attacks import CarliniWagnerL2
from tensorflow.contrib.slim.nets import inception
import adversarial.importer as importer
import adversarial.config as config
import pandas as pd
import numpy as np
import sys
import os
from cleverhans.model import Model
from optparse import OptionParser
#import matplotlib.image as mpimg
from scipy.misc import imsave
from scipy.misc import imread
import json


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
    
class InceptionModelLogits(object):
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
        output = end_points['Logits']
        logits = output.op.inputs[0]
        return logits
    
    
'''Create adversarial images from the orignal data set
'''
        
def adversarial_generator(mode,batch_shape,eps,is_return_orig_images=False):                         
    
    def next_images():
        tf.logging.set_verbosity(tf.logging.INFO)
        graph_fgsm = tf.Graph()
        print("{} generator graph is ready!".format(mode))
        with graph_fgsm.as_default():    
            x_input = tf.placeholder(tf.float32, shape=batch_shape)
                   
            
            params = {'eps':eps}
            if mode == 'fgsm':
                model = InceptionModel(importer.num_classes) 
                graph = FastGradientMethod(model)
            elif mode == 'ifgsm':
                model = InceptionModel(importer.num_classes) 
                params['nb_iter'] = 10
                graph  = BasicIterativeMethod(model)                
            elif mode == 'deep_fool':
                model = InceptionModelLogits(importer.num_classes) 
                graph = DeepFool(model)
                params['max_iter'] = 5
            elif mode == 'carlini_wagner':
                graph = CarliniWagnerL2(model)
                params["confidence"] = 0
                params["initial_const"] = 10
                #TODO: to set hyper parameter
                
            print('graph params: {}'.format(params))
            x_adv = graph.generate(x_input, **params)
        
            saver = tf.train.Saver(slim.get_model_variables())
            session_creator = tf.train.ChiefSessionCreator(
                              scaffold=tf.train.Scaffold(saver=saver),
                              checkpoint_filename_with_path=importer.checkpoint_path,
                              master=importer.tensorflow_master)
        
            image_iterator = importer.load_images_generator(batch_shape)
            with tf.train.MonitoredSession(session_creator=session_creator) as sess: 
                while True:
                    filenames, images = next(image_iterator,(None,None))
                    if filenames is None: break
                    adversarial_images = sess.run(x_adv, feed_dict={x_input: images})
                    #print("Image:{}, diff:{}".format(filenames[0],np.sum(np.abs(images[0]-adversarial_images[0]))))
                    if is_return_orig_images:
                        yield  filenames,adversarial_images,images
                    else:                        
                        yield  filenames,adversarial_images
    
    return  next_images()   



def test_deep_full_simple_model():
    class SimpleModel(Model):
        """A very simple neural network
        """
        def get_logits(self, x):
            W1 = tf.constant([[1.5, .3], [-2, 0.3]], dtype=tf.as_dtype(x.dtype))
            h1 = tf.nn.sigmoid(tf.matmul(x, W1))
            W2 = tf.constant([[-2.4, 1.2], [0.5, -2.3]], dtype=tf.as_dtype(x.dtype))
    
            res = tf.matmul(h1, W2)
            return res
    
    
    sess = tf.Session()
    model = SimpleModel()
    attack = DeepFool(model=model, sess=sess)
    x_val = np.random.rand(1, 2)
    x_val = np.array(x_val, dtype=np.float32)

    x_adv = attack.generate_np(
            x_val, over_shoot=0.02, max_iter=5,
            nb_candidate=2, clip_min=-5,
            clip_max=5)    
    adversarial_images = sess.run(model(x_adv))    
    print("adversarial_images shape:{}".format(adversarial_images.shape))
    
    
    
def test_deep_full_inception_v3_model():
    class Inception_V3_Model(Model):
        """A very simple neural network
        """        
        def get_logits(self, x_input):
            NUM_CLASSES = 1000
            """Constructs model and return probabilities for given input."""
            #reuse = True if self.built else None
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points = inception.inception_v3(
                                x_input, num_classes=NUM_CLASSES, is_training=False,
                                reuse=False)
            self.built = True
            output = end_points['Logits']
            probs = output.op.inputs[0]
            return probs
    
    
    sess = tf.Session()
    model = Inception_V3_Model()
    attack = DeepFool(model=model, sess=sess)
    x_val = np.random.rand(1, 299,299,3)
    x_val = np.array(x_val, dtype=np.float32)

    x_adv = attack.generate_np(
            x_val, over_shoot=0.02, max_iter=5,
            nb_candidate=2, clip_min=-5,
            clip_max=5)    
    adversarial_images = sess.run(model(x_adv))    
    print("adversarial_images shape:{}".format(adversarial_images.shape))
    
    
def test_deep_full_inception_v3_model__():
    class Inception_V3_Model(Model):
        """A very simple neural network
        """        
        def get_logits(self, x_input):
            NUM_CLASSES = 1000
            """Constructs model and return probabilities for given input."""
            #reuse = True if self.built else None
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points = inception.inception_v3(
                                x_input, num_classes=NUM_CLASSES, is_training=False,
                                reuse=False)
            self.built = True
            output = end_points['Logits']
            probs = output.op.inputs[0]
            return probs
        
        
    
    x_val = np.random.rand(1, 299,299,3)
    x_val = np.array(x_val, dtype=np.float32)
    model = Inception_V3_Model()
    x_input = tf.placeholder(tf.float32, shape=[1,299,299,3])
    tf.logging.set_verbosity(tf.logging.DEBUG)    
    #saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
                      #scaffold=tf.train.Scaffold(saver=saver),
                      checkpoint_filename_with_path=importer.checkpoint_path,
                      master=importer.tensorflow_master)    
    
    #attack = DeepFool(model=model, sess=sess)
    attack = DeepFool(model=model)
    x_adv = attack.generate(
                x_val, over_shoot=0.02, max_iter=5,
                nb_candidate=2, clip_min=-5,
                clip_max=5)    
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:        
        adversarial_images = sess.run(x_adv,feed_dict={x_input: x_val})
        print("adversarial_images shape:{}".format(adversarial_images.shape))

if __name__ == "__main__":
    test_deep_full_inception_v3_model()
    #test_deep_full_simple_model()
   
    
    
   
    

    
    
            
                
            
            
        
        
        

