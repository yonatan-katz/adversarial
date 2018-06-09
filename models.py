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
import adversarial.utils as utils
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

class InceptionModelProb(object):
    def __init__(self, num_classes,x_input):
        self.num_classes = num_classes        
        with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points = inception.inception_v3(
                    x_input, num_classes=importer.num_classes, is_training=False,
                    reuse=False)       

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""        
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                            x_input, num_classes=self.num_classes, is_training=False,
                            reuse=False)            
        
        output = end_points['Predictions']
        probs = output.op.inputs[0]
        return probs
    
class InceptionModelLogits(object):
    def __init__(self, num_classes,x_input):
        self.num_classes = num_classes
        self.num_classes = num_classes        
        with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points = inception.inception_v3(
                    x_input, num_classes=importer.num_classes, is_training=False,
                    reuse=False)       

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""        
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                            x_input, num_classes=self.num_classes, is_training=False,
                            reuse=True)
        
        output = end_points['Logits']
        logits = output.op.inputs[0]
        return logits
    
    
'''Create adversarial images from the orignal data set
    supports fgsm and ifgsm modes 
'''
def adversarial_generator_basic(mode,batch_shape,eps,is_return_orig_images=False):                         
    
    def next_images():
        tf.logging.set_verbosity(tf.logging.INFO)
        graph_fgsm = tf.Graph()
        print("{} generator graph is ready!".format(mode))
        with graph_fgsm.as_default():    
            x_input = tf.placeholder(tf.float32, shape=batch_shape)
            
            model = InceptionModelProb(importer.num_classes,x_input) 
            params = {'eps':eps}
            if mode == 'fgsm': 
                graph = FastGradientMethod(model)
            elif mode == 'ifgsm':
                params['nb_iter'] = 10
                graph  = BasicIterativeMethod(model)                
            else:
                raise Exception("Not supported mode")
                
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

 
'''Create adversarial images from the orignal data set
    supports deep fool and  and carlini wagner modes modes 
'''
def adversarial_generator_advanced(mode,batch_shape,eps,is_return_orig_images=False):                         
    print("adversarial attack mode:{}".format(mode))
    def next_images():
        tf.logging.set_verbosity(tf.logging.INFO)
        print("{} generator graph is ready!".format(mode))
        tf.reset_default_graph()        
        sess = tf.Session()
        x_input = tf.placeholder(tf.float32, shape=importer.batch_shape)
        params = {}
        model = InceptionModelLogits(importer.num_classes,x_input)         
        if mode == 'deep_fool':
            graph = DeepFool(model,sess=sess)
            params['max_iter'] = 5
        elif mode == 'carlini_wagner':
            graph = CarliniWagnerL2(model,sess=sess)
            params["confidence"] = 0
            params["initial_const"] = 10
            params['learning_rate'] = 0.1
            params['max_iterations'] = 10
        else:
            raise Exception("Not supported mode")             
            
        print('graph params: {}'.format(params))
        variables = tf.get_collection(tf.GraphKeys.VARIABLES)        
        saver = tf.train.Saver(variables)
        saver.restore(sess, importer.checkpoint_path) 
        image_iterator = importer.load_images_generator(batch_shape)
        while True:
            filenames, images = next(image_iterator,(None,None))
            if filenames is None: break
            true_classes = importer.filename_to_class(filenames)
            target = np.expand_dims(np.zeros(importer.num_classes),1)
            if mode == 'carlini_wagner':
                assert(len(true_classes) == 1)
                target[true_classes[0]] = 1                    
                params["y"] = target
            x_adv = graph.generate(x_input, **params)
            adversarial_images = sess.run(x_adv, feed_dict={x_input: images})
            print("Image:{}, diff:{}".format(filenames[0],np.sum(np.abs(images[0]-adversarial_images[0]))))
            if is_return_orig_images:
                yield  filenames,adversarial_images,images
            else:                        
                yield  filenames,adversarial_images
        
    return  next_images()   


def test_inception_v3_model(atack_type):
    class Inception_V3_Model(Model):
        """A very simple neural network
        """         
        def __init__(self,x_input):
            
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points = inception.inception_v3(
                                x_input, num_classes=importer.num_classes, is_training=False,
                                reuse=False)       
            
        
            
        def get_logits(self, x_input):            
            """Constructs model and return probabilities for given input."""
            #reuse = True if self.built else None          
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points = inception.inception_v3(
                                x_input, num_classes=importer.num_classes, is_training=False,
                                reuse=True)
            
            self.built = True
            output = end_points['Logits']            
            probs = output.op.inputs[0]                                    
            return probs
    
    counter = 0
    image_iterator = importer.load_images_generator(importer.batch_shape)
    while True:
        tf.reset_default_graph()        
        sess = tf.Session()
        x_input = tf.placeholder(tf.float32, shape=importer.batch_shape)
        folder_path = os.path.join(config.ADVERSARIAL_FOLDER,atack_type+"_test")
        os.makedirs(folder_path,exist_ok=True)        
        with tf.Session() as sess:                            
            filenames, images = next(image_iterator,(None,None))
            print(filenames)            
            true_classes = importer.filename_to_class(filenames)
            model = Inception_V3_Model(np.float32(images))
            params = {}
            if atack_type == "deep_fool":
                attack = DeepFool(model=model,sess=sess)
                params['max_iter'] = 5
            elif atack_type == "carlini":
                attack = CarliniWagnerL2(model,sess=sess)
                params["confidence"] = 0
                params["initial_const"] = 10
                params['learning_rate'] = 0.001
                params['max_iterations'] = 100
                params['clip_min'] = -1
                params['clip_max'] = 1
                target = np.expand_dims(np.zeros(importer.num_classes),1)
                assert(len(true_classes) == 1)
                target[true_classes[0]] = 1
                params["y"] = target
            else:
                raise("Bad attack type!")              
            
            variables = tf.get_collection(tf.GraphKeys.VARIABLES)                
            saver = tf.train.Saver(variables)
            saver.restore(sess, importer.checkpoint_path)            
            x_adv = attack.generate(x_input,**params)
            #writer = tf.summary.FileWriter("/tmp/log/", sess.graph)
            adversarial_images = sess.run(x_adv, feed_dict={x_input: images})
            utils.image_saver(adversarial_images,filenames,folder_path)
            print("adversarial_images counter:{}".format(counter))
            #writer.close()
            counter += 1
            if counter == 1000:
                break

if __name__ == "__main__":
   test_inception_v3_model("carlini")
   #test_deep_full_simple_model()
   
    
    
   
    

    
    
            
                
            
            
        
        
        

