# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:18:04 2019

@author: s163666
"""
"""
Registration project code.
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
    I_path = '../data/image_data/1_1_t1.tif'
    Im_path = '../data/image_data/1_1_t1_d.tif'
    
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
   
    
    
def intensity_based_registration_demo():

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread('C:/Users/miria/Documents/Technische Universiteit Eindhoven/Medical Image Analysis/Mini_project_mia/data/image_data/1_1_t1.tif')
    Im = plt.imread('C:/Users/miria/Documents/Technische Universiteit Eindhoven/Medical Image Analysis/Mini_project_mia/data/image_data/1_1_t1_d.tif')

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    x = np.array([0., 0., 0.])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    fun = lambda x: reg.rigid_corr(I, Im, x)

    # the learning rate
    mu = 0.001

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        S, Im_t, _ = fun(x)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)
