#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 21:36:44 2018

@author: yonic
"""
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from tensorflow.contrib.slim.nets import inception
import adversarial.importer as importer
import pandas as pd
import numpy as np
import sys
import os
from optparse import OptionParser
#import matplotlib.image as mpimg
from scipy.misc import imsave



ADVERSARIAL_FOLDER = "output/adversarial"

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
    
    
'''Create adversarial images from the orignal data set
'''
        
def fgsm_generator(batch_shape,eps,is_return_orig_images=False):                         
    
    def next_images():
        tf.logging.set_verbosity(tf.logging.INFO)
        graph_fgsm = tf.Graph()
        print("fgsm_generator graph is ready!")    
        with graph_fgsm.as_default():    
            x_input = tf.placeholder(tf.float32, shape=batch_shape)
            model = InceptionModel(importer.num_classes)
        
            fgsm  = FastGradientMethod(model)
            #x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)
            x_adv = fgsm.generate(x_input, eps=eps)
        
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
                    print("Image:{}, diff:{}".format(filenames[0],np.sum(np.abs(images[0]-adversarial_images[0]))))
                    if is_return_orig_images:
                        yield  filenames,adversarial_images,images
                    else:                        
                        yield  filenames,adversarial_images
    
    return  next_images()



def fgsm_attack(eps):
    generator = fgsm_generator(importer.batch_shape,eps)
    print("fgsm eps:{}".format(eps))
    folder_path = os.path.join(ADVERSARIAL_FOLDER,"fgsm.{}".format(eps))
    os.makedirs(folder_path,exist_ok=True)
    counter = 0
    while counter < 10 :
        filenames, images = next(generator,(None,None))
        if filenames is None:
            break
        for fname,image in zip(filenames, images):
            image = np.uint8((image+1.0)/2.0*255.0)
            ffname = os.path.join(folder_path,fname)
            imsave(ffname, image)
            print("Adversarial {} is generated,min:{},max:{}".format(ffname,np.min(image),np.max(image)))       
            if counter > 10:
                break
            counter += 1
        
            


def main():
    parser = OptionParser()
    parser.add_option("-a","--attack",dest="attack",help="mode:[fgsm]")
    parser.add_option("--eps",dest="eps",help="FGSM eps parameter",type=float,default=importer.eps)
    (options, args) = parser.parse_args()    
    
    if not options.attack:
        parser.print_help()
        sys.exit(1)
    if options.attack == "fgsm":
        fgsm_attack(options.eps)
        sys.exit(1)
        
    parser.print_help()
    
    
if __name__ == "__main__":
    main()
    
    

    
    
            
                
            
            
        
        
        
