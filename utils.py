#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 20:37:11 2018

@author: yonic
"""

import numpy as np

'''Calculate euclidian distance between two image arrays
'''
def dissimilariry(orig_images,adv_images):
    S = 0
    for orig,adversal in zip(orig_images,adv_images):
        S += np.linalg.norm([orig-adversal])/np.linalg.norm([orig])
    return S
     
        
    
    