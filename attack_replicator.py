#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 17:24:05 2018

@author: yonic
"""

import adversarial.importer as importer
import adversarial.utils as utils
import adversarial.config as config
import os
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

def replicate_attack(attack_type,eps):
    attack_base_folder = os.path.join(config.ADVERSARIAL_FOLDER,attack_type,'base')    
    image_iterator = importer.load_images_generator(importer.batch_shape)
    attack_images_diff = np.zeros(importer.batch_shape,np.float32)
    count = 0
    while True:
        if count<1000:
            filenames, images = next(image_iterator,(None,None))
            idx = 0
            for fname,image in zip(filenames,images):
                attack_image_path = os.path.join(attack_base_folder,fname)
                adv_image = imread(attack_image_path, mode='RGB').astype(np.float32)*2.0/255.0 - 1.0
                attack_images_diff[idx, :, :, :] = image + eps * (image - adv_image)                
                idx +=1                          
        
            count +=1       
            yield filenames,attack_images_diff,images            
        else:
            yield None,None,None
        
        
if __name__ == "__main__":    
     replicate_attack("deep_fool",10)
    
    