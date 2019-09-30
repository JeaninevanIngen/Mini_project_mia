"""
Registration project code.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output
    
def intensity_based_registration_for_loop():
    
    # Defining of images, learning rates and number of iterations that you want to test.
    # Look at lines 24/25 and 112 to properly define the image paths and savenames to your needs.
    imagepaths = ['2_2','2_3','3_1','3_2','3_3']
    mu_s = [0.00018,0.0001,0.0002,0.00013,0.00015]
    iterationlist = [350,350,350,350,350]
    savenames = ['2_2','2_3','3_1','3_2','3_3']
    
    for i in range(len(imagepaths)):
        
        # read the fixed and moving images
        # change these in order to read different images
        I = plt.imread('./data/image_data/{}_t1.tif'.format(imagepaths[i]))
        Im = plt.imread('./data/image_data/{}_t1_d.tif'.format(imagepaths[i]))

        # initial values for the parameters
        # we start with the identity transformation
        # most likely you will not have to change these
        similarity_measure = reg.affine_corr

        if similarity_measure == reg.rigid_corr:
            x = np.array([0., 0., 0.])

        else:
            x = np.array([0., 1., 1., 0., 0., 0., 0.])

        # NOTE: for affine registration you have to initialize
        # more parameters and the scaling parameters should be
        # initialized to 1 instead of 0

        # the similarity function
        # this line of code in essence creates a version of rigid_corr()
        # in which the first two input parameters (fixed and moving image)
        # are fixed and the only remaining parameter is the vector x with the
        # parameters of the transformation
        fun = lambda x: reg.affine_mi(I, Im, x)

        # the learning rate
        mu = mu_s[i]

        # number of iterations
        num_iter = iterationlist[i]

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
        if similarity_measure == reg.rigid_corr:
            txt = ax1.text(0.3, 0.95,
            np.array2string(x, precision=5, floatmode='fixed'),
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
            transform=ax1.transAxes)
        else: 
            txt = ax1.text(-0.02, 1.02,
            np.array2string(x, precision=5, floatmode='fixed'),
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
            transform=ax1.transAxes)
        # 'learning' curve
        ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

        learning_curve, = ax2.plot(iterations, similarity, lw=2)
        ax2.set_title("mu =" + str(mu))
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
            
            # display the figure
            display(fig)
        
        # save the figure
        # Currently optimized for speed. If one wants to save an image every iteration one has to tab these lines. 
        savename = '{}_t1 + {}_t1_d affine_mi mu = {} integer = {}.png'.format(savenames[i],savenames[i],mu,num_iter)
        fig.savefig(savename) 
