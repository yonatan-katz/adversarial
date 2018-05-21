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
from scipy.misc import imread
import json



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
    generator = fgsm_generator(importer.batch_shape,eps=eps,is_return_orig_images=True)
    print("fgsm eps:{}".format(eps))
    folder_path = os.path.join(ADVERSARIAL_FOLDER,"fgsm.{}".format(eps))
    os.makedirs(folder_path,exist_ok=True)
    counter = 0
    S = 0.0    
    while counter < 10 :
        filenames,adversarial_images,images = next(generator,(None,None))
        if filenames is None:
            break
        for fname,adversarial_image,image in zip(filenames, adversarial_images,images):
            s = np.linalg.norm([image - adversarial_image]) / np.linalg.norm([image])
            S += s            
            
            #save image
            ffname = os.path.join(folder_path,fname)
            adversarial_image = np.uint8((adversarial_image+1.0)/2.0*255.0)
            imsave(ffname, adversarial_image)
            
            if counter > 10:
                break
            counter += 1
    
    stat = {"eps":eps,
            "dissimilarity":S/np.float32(counter),
            
    }
    fname = os.path.join(ADVERSARIAL_FOLDER,"fgsm.{}".format(eps),"dissimilarity.json")
    with open(fname, 'w') as file:
        file.write(json.dumps(stat))     
        
    
        
        
def test1():
    def get_diff(a,b):
        return np.linalg.norm([a-b]) 
    
    def transform(a):
        a = np.uint8((a+1.0)/2.0*255.0)
        #a = np.float32(a)*2.0/255.0 - 1.0
        return a
    
    fname = "data/images/000b7d55b6184b08.png"
    a = imread(fname, mode='RGB').astype(np.float32)*2.0/255.0 - 1.0
    a = transform(a)
    b = imread("/tmp/a.png", mode='RGB')    
    diff = get_diff(a.flatten(),b.flatten())
    print("Diff:{}".format(diff))
    print("a:sum:{},b:sum:{}".format(np.sum(a),np.sum(b)))
    
    
    
        
def test():
    def get_diff(a,b):
        return np.linalg.norm([a-b])
    
    def save(a):
        a = np.uint8((a+1.0)/2.0*255.0)
        imsave("/tmp/a.png", a)
    
    def load():
        #a = imread("/tmp/a.png", mode='RGB').astype(np.float32)*2.0/255.0 - 1.0        
        a = imread("/tmp/a.png", mode='RGB')
        return a
    
    def transform(a):
        a = np.uint8((a+1.0)/2.0*255.0)
        #a = np.float32(a)*2.0/255.0 - 1.0
        return a
        
        
        
        
        
    generator = fgsm_generator(importer.batch_shape,eps=0,is_return_orig_images=True)
    filenames,adversarial_images,images = next(generator)
    
    diff1 = get_diff(adversarial_images[0].flatten(),images[0].flatten())
    
    save(adversarial_images[0])
    loaded_adversarial_image = load()
    transformed = transform(adversarial_images[0])
    orig_transformed = transform(images[0])
    
    diff2 = get_diff(loaded_adversarial_image.flatten(),images[0].flatten())
    diff3 = get_diff(loaded_adversarial_image.flatten(),adversarial_images[0].flatten())
    diff4 = get_diff(orig_transformed.flatten(),loaded_adversarial_image.flatten())
    diff5 = get_diff(transformed.flatten(),orig_transformed.flatten())
    
    print("Diff1:{},diff2:{},diff3:{},diff4:{},diff5:{}".format(diff1,diff2,diff3,diff4,diff5))
    
    
    
            


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
    #test1()
    
    

    
    
            
                
            
            
        
        
        

