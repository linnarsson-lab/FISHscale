import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Polygon
from sklearn.neighbors import KernelDensity
import pandas as pd

def rotate(p, origin=(0, 0), angle=0, degrees=None):
    """
    Angle in radians between -pi and pi
    If degree is given it will ignore angle
    Adapted from https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    """
    if degrees != None:
        angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def get_rotation(x1, y1, x2, y2):
    """
    Calculates rotation of vector between 2 points.
    Return the angle in radians.
    
    """
    dx = x2 - x1
    dy = y2 - y1
    
    angle = math.atan2(dy, dx)

    return angle

def to_rotate(angle):
    "Calculate angle to rotate to turn objec straigt from origin to y max, in radians "
    return 0.5*np.pi - angle

def density(points, bandwidth=1, kernel='gaussian',start=None, stop=None, resolution=50):
    """
    Calculates the kernel density estimate over a set of 1D points.
    Input:
    `points`(np.array): 1D Array of points over which to calculate the KDE.
    `bandwidth`(float): Bandwidth for KDE. 
    `kernel`(str): Kernel for KDE.
    `start`(float): Start of the sampling. If None takes points.min().
    `stop`(float): Stop of the sampling. In None takes points.max()
    `resolution`(int): Number of points between start and stop.
    See scipy.KernelDensity for details.
    
    """
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    if start == None:    
        start = points.min()
    if stop == None:
        stop = points.max()
    sample_space = np.linspace(start, stop, resolution)
      
    kde.fit(points[:, None])
    
    logprob = kde.score_samples(sample_space[:,None])
    
    probability = np.exp(logprob)
    
    return probability, sample_space

def calc_density(points, line, width, bandwidth=1, kernel='gaussian', resolution=50, plot=False, bandwidth_extention=False, s=2):
    """
    Function to calcualted probablity of a kernel density estimate over a line
    with a certain width. 
    Input:
    `points`(np.array): Numpy array with X and Y coordinates in the columns.
    `line`(list): Begin and end coordinates of the line over which to calculate 
        the density. Format: [x1, y1, x2, y2].
    `width`(float): Width over which the density is sampled. Same units as the 
        points coordinates.
    
    `bandwidth`(float): Bandwidth for KDE. 
    `kernel`(str): Kernel for KDE.
    `resolution`(int): Number of points for KDE sampling.
    `plot`(bool): If True it plots the points, line and area over which it 
        calculated the density. 
    `bandwidth_extention`(bool): If True, it elongates the line on both sides
        for making the KDE, so that the KDE does show less edge effects. 
        The elongation is 3 bandwidths on either side.
    `s`(float): Size of dots for plotting
    
    Output:
    `probability`(np.array): Probability under the KDE at the sampling points.
        Area under curve is 1 if bandwidth_extention is False.
    `sampling_points`(np.array): Points where the KDE has been sampled.
    `filter`(np.array): Boolean array for the selected points.
    
    """
    
    x1, y1, x2, y2 = line
    
    
    #get angle of line
    angle = get_rotation(x1, y1, x2, y2)

    #Get angle to rotate to turn line straight
    angle_to_rotate = to_rotate(angle)

    #Rotate points
    rotated = rotate(points, origin=(0,0), angle=angle_to_rotate)
    #Rotate line
    line_rotated = rotate(np.column_stack(([x1, x2], [y1, y2])), origin=(0,0), angle=angle_to_rotate)
    
    #Get min an max to select points
    x_min = line_rotated[0][0] - 0.5*width
    x_max = line_rotated[0][0] + 0.5*width
    y_min = line_rotated[0][1]
    y_max = line_rotated[1][1]
    
    #Calculate bandwidth if needed.
    if bandwidth == None:
        bandwidth = abs(y_max - y_min) / resolution

    #Select points
    if bandwidth_extention:
        factor = 3
        filt = ((rotated >= [x_min, y_min- factor*bandwidth]) & (rotated <= [x_max, y_max+ factor*bandwidth])).all(axis=1)
    else:
        filt = ((rotated >= [x_min, y_min]) & (rotated <= [x_max, y_max])).all(axis=1)
    rotated_select = rotated[filt,:]
    
    #Calculated KDE on y coordinates
    probability, sample_space = density(rotated_select[:,1], bandwidth=bandwidth, kernel=kernel, start=y_min, stop=y_max, resolution=resolution)
    
    if plot:
        fig, ax = plt.subplots(ncols=2, figsize=(20,10))
        
        ax0 = ax[0]
        #Plot all points
        ax0.scatter(points[:,0], points[:,1], s=s, color='gray')

        #Plot original line
        ax0.plot([x1, x2], [y1, y2], color='r')

        #Shift line up and down a half width
        line_min = line_rotated.copy()
        line_min[:,0] = line_min[:,0] - (0.5*width) 
        line_max = line_rotated.copy()
        line_max[:,0] = line_max[:,0] + (0.5*width) 

        #Rotate the shifted angles back
        rotate_back_angle = 2*np.pi - to_rotate(angle)
        line_min_rotate_back = rotate(line_min, angle=rotate_back_angle)
        line_max_rotate_back = rotate(line_max, angle=rotate_back_angle)

        #Plot the area as a polygon
        rectangle = Polygon(np.row_stack((line_min_rotate_back, np.flipud(line_max_rotate_back))), alpha=0.5)
        ax0.add_patch(rectangle)
        ax0.set_aspect('equal')
        ax0.set_title('Coordinates with sample line')
        
        ax1 = ax[1]
        ax1.plot(sample_space, probability)
        ax1.set_title('KDE')
        ax1.set_ylabel('Probability')
        ax1.set_xlabel('Position')
    
    return probability, sample_space, filt
