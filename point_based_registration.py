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

def point_based_error(I_path,Im_path,T):
    
    #Select set of corresponding points using my_cpselect
    X0, Xm0 = util.my_cpselect(I_path, Im_path)
    
    Xm = np.ones((3, Xm0.shape[1])) 
    Xm[0,:] =Xm0[0,:]
    Xm[1,:]=Xm0[1,:]
    X = np.ones((3, X0.shape[1])) 
    X[0,:] =X0[0,:]
    X[1,:]=X0[1,:]
    b1 = np.transpose(X[0,:])
    b2 = np.transpose(X[1,:])
    #Compute the affine transformation between the pair of images
    _, Etestx = reg.ls_solve(T,b1)
    _, Etesty = reg.ls_solve(T,b2)
    Etest=np.array([[Etestx],[Etesty]])
    
    return Etest


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
    I_path = '../data/image_data/3_1_t1.tif'
    Im_path = '../data/image_data/3_1_t2.tif'
    
    #Select set of corresponding points using my_cpselect
    X0, Xm0 = util.my_cpselect(I_path, Im_path)
    
    #Compute the affine transformation between the pair of images
    Xm = np.ones((3, Xm0.shape[1])) 
    Xm[0,:] =Xm0[0,:]
    Xm[1,:]=Xm0[1,:]
    X = np.ones((3, X0.shape[1])) 
    X[0,:] =X0[0,:]
    X[1,:]=X0[1,:]
    T, Etrain = reg.ls_affine(X, Xm)
    
    Etest=point_based_error(I_path,Im_path,T)
    
    #read image
    Im=plt.imread(Im_path)
    
    #Apply the affine transformation to the moving image
    It, Xt = reg.image_transform(Im, T)
    
    #plot figure
    fig = plt.figure(figsize=(12,5))
    #fig, ax = plt.subplots()
    
    ax = fig.add_subplot(121)
    im = ax.imshow(It)
    
    ax.set_title("Transformed image")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    print(Etrain)
    print(Etest)
    
    display(fig)
    fig.savefig('C:/Users/s163666/Documents/2019-2020/Medical Image Analysis/Mini_project_mia/pointbased/3_1_t1 + 3_1_t2.png')
    

   
    