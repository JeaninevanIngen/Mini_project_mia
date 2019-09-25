# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:27:10 2019

@author: s163666

    # ---------------------------------
    # Append the following code in jupiter to run:
    
%matplotlib inline
import sys
sys.path.append("C:/Users/s163666/Documents/2019-2020/Medical Image Analysis/Mini_project_mia")
from project_fout import point_based_registration_demo
point_based_registration_demo()
---------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output

def point_based_registration_demo():
    # ---------------------------------
    # Append the following code in jupiter to run:
    #   %matplotlib inline
    #   import sys
    #   sys.path.append("C:/Users/s163666/Documents/2019-2020/Medical Image Analysis/Mini_project_mia")
    #   from project_fout import point_based_registration_demo
    # point_based_registration_demo()
    # ---------------------------------
    
    # read the fixed and moving images, 2x T1 or T1&T2
    # change these in order to read different images
    I_path = './data/image_data/1_1_t1.tif'
    Im_path = './data/image_data/1_1_t1_d.tif'
    
    #Select set of corresponding points using my_cpselect
    X, Xm0 = util.my_cpselect(I_path, Im_path)

    #Compute the affine transformation between the pair of images
    Xm=np.ones((Xm0.shape[0],3))
    #wside=np.expand_dims(wside, axis=1)
    
    Xm[:,0]=Xm0[:,0]
    Xm[:,1]=Xm0[:,1]
    Xm=np.transpose(Xm)
    T = reg.ls_affine(X, Xm)
    
    #read image
    Im=plt.imread(Im_path)
    
    #Apply the affine transformation to the moving image
    It, Xt = reg.image_transform(Im, T)
    
    return It, Xt
   
    