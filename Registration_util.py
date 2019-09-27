"""
Utility functions for registration.
"""

import numpy as np
from cpselect.cpselect import cpselect


def test_object(centered=True):
    # Generate an F-like test object.
    # Input:
    # centered - set the object centroid to the origin
    # Output:
    # X - coordinates of the test object

    X = np.array([[4, 4, 4.5, 4.5, 6, 6, 4.5, 4.5, 7, 7, 4], [10, 4, 4, 7, 7, 7.5, 7.5, 9.5, 9.5, 10, 10]])

    if centered:
        X[0, :] = X[0, :] - np.mean(X[0, :])
        X[1, :] = X[1, :] - np.mean(X[1, :])

    return X


def c2h(X):
    # Convert cartesian to homogeneous coordinates.
    # Input:
    # X - cartesian coordinates
    # Output:
    # Xh - homogeneous coordinates

    n = np.ones([1,X.shape[1]])
    Xh = np.concatenate((X,n))

    return Xh


def t2h(T, Xt):
    # Convert a 2D transformation matrix to homogeneous form.
    # Input:
    # T - 2D transformation matrix
    # Xt - 2D translation vector
    # Output:
    # Th - homogeneous transformation matrix

    #------------------------------------------------------------------#
    # TODO: Implement conversion of a transformation matrix and a translation vector to homogeneous transformation matrix.
    Th = np.eye(3)
    Th[0:2,0:2] = T
    Th[0:2,2] = Xt
    return Th
    # remove the 'pass' once implemented
    #------------------------------------------------------------------#
    

def plot_object(ax, X):
    # Plot 2D object.

    # Input:
    # X - coordinates of the shape

    ax.plot(X[0,:], X[1,:], linewidth=2)


def my_cpselect(I_path, Im_path):
    # Wrapper around cpselect that returns the point coordinates
    # in the expected format (coordinates in rows).
    # Input:
    # I - fixed image
    # Im - moving image
    # Output:
    # X - control points in the fixed image
    # Xm - control points in the moving image

    #------------------------------------------------------------------#
    # TODO: Call cpselect and modify the returned point coordinates.
   
    controlpointdictlist = cpselect(I_path, Im_path) #indicate corresponding points on images
    k = len(controlpointdictlist) 

    if k <= 2:
        raise AssertionError("More points must be selected")
        
    X = np.zeros((2, k)) #reserve memory
    Xm = np.zeros((2, k)) #reserve memory

    #Take the values of the i'th dictionaries using the corresponding keys
    for i in range(k): 
        
        I_x = controlpointdictlist[i]['img1_x'] 
        I_y = controlpointdictlist[i]['img1_y'] 
        Im_x = controlpointdictlist[i]['img2_x'] 
        Im_y = controlpointdictlist[i]['img2_y'] 


        X[:,i] = np.array([I_x, I_y])
        Xm[:,i] = np.array([Im_x, Im_y])
        
    #------------------------------------------------------------------#

    return X, Xm