import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Polygon
from sklearn.neighbors import KernelDensity
from typing import Tuple
from numpy.lib.stride_tricks import sliding_window_view

class Density1D:

    def rotate(self, p: np.ndarray, origin: tuple=(0, 0), angle: float=0, degrees: bool=None) -> np.ndarray:
        """Rotate points around an origin with a certain angle.

        Args:
            p (np.ndarray): Array with xy coordinates
            origin (tuple, optional): Point to rotate around. Defaults to (0, 0).
            angle (float, optional): Angle in radians between -pi and pi.
                Defaults to 0.
            degrees (bool, optional): Angle to rotate in degree, if this is given 
                the function will ignore any "angle" input. Defaults to None.

        Returns:
            np.ndarray: Rotated points. 
        """
        if degrees != None:
            angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)

    def get_rotation_rad(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculates angle of vector between 2 points.

        Args:
            x1 (float): x coordinate first point.
            y1 (float): y coordinate first point.
            x2 (float): x coordinate second point.
            y2 (float): y coordinate second point.

        Returns:
            float: Angle in radians. 
        """
        dx = x2 - x1
        dy = y2 - y1
        
        angle = math.atan2(dy, dx)

        return angle

    def to_rotate(self, angle: float) -> float:
        """Calculate angle to rotate object straight from origin to y max.

        Args:
            angle ([type]): Input angle in radians.

        Returns:
            [type]: Output angle in radians.
        """
        return 0.5*np.pi - angle

    def density(self, points: np.ndarray, bandwidth: float=1, kernel: str='gaussian',start: float=None, stop: float=None, 
                resolution: int=50) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the kernel density estimate over a set of 1D points.
        
        See Sklearn.neighbours.KernelDensity for documentation on KDE.

        Args:
            points (np.ndarray): 1D Array of points over which to calculate the 
                KDE.
            bandwidth (float, optional): Bandwidth for KDE. Defaults to 1.
            kernel (str, optional): Kernel for KDE. Defaults to 'gaussian'.
            start (float, optional): Start of the sampling. If None takes 
                points.min(). Defaults to None.
            stop (float, optional): Stop of the sampling. In None takes 
                points.max(). Defaults to None.
            resolution (int, optional):  Number of points between start and stop.
                Defaults to 50.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                Array with probabilities along the sampling space. 
                Array with locations of sampling points.
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

    def density_probability(self, points: np.ndarray, line: list, width: float, bandwidth: float=1, kernel: str='gaussian', 
                    resolution: int=50, plot: bool=False, bandwidth_extention: bool=False, s: float=2) -> Tuple[np.ndarray,
                                                                                                                np.ndarray,
                                                                                                                np.ndarray,
                                                                                                                np.ndarray]:
        """Calculate KDE density of points along a line with a certain width.
        
        Calculates the kernel density estimate (KDE) in 1 dimension over a line. 
        Use the width to determine how close point need to be to be included in the 
        sampling. Use the plot function to visualize the chosen area and the 
        density estimate.
        The KDE will normaly have a edge effect, especially if the bandwidth is
        large. To partially circumvent this the points at the start and finish can
        can be mirrored by setting "bandwidth_extention" to True.
        
        See Sklearn.neighbours.KernelDensity for documentation on KDE.

        Args:
            points (np.ndarray): Numpy array with X and Y coordinates as columns.
            line (list): Begin and end coordinates of the line over which to 
                calculate the density. Format: [x1, y1, x2, y2].
            width (float): Width over which the density is sampled. Same units as
                the points coordinates.
            bandwidth (float, optional): Bandwidth for KDE. Defaults to 1.
            kernel (str, optional): Kernel for KDE, see sklearn documentation for
                options. Defaults to 'gaussian'.
            resolution (int, optional): Number of points for KDE sampling.
                Defaults to 50.
            plot (bool, optional): If True it plots the points, line and area over
                which it calculated the density. Defaults to False.
            bandwidth_extention (bool, optional): If True, it elongates the line on
                both sides for making the KDE, so that the KDE does show less edge 
                effects. The elongation is 3 bandwidths on either side. 
                Defaults to False.
            s (float, optional): Size of dots for plotting. Defaults to 2.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                Array with probabilities along the sampling space. 
                Array with locations of sampling points.
                Boolean array for all points that fall within the selected
                rectangle.
                Array with the points over which the KDE is calculated.
        """
        x1, y1, x2, y2 = line
        
        #get angle of line
        angle = self.get_rotation_rad(x1, y1, x2, y2)

        #Get angle to rotate to turn line straight
        angle_to_rotate = self.to_rotate(angle)

        #Rotate points
        rotated = self.rotate(points, origin=(0,0), angle=angle_to_rotate)
        #Rotate line
        line_rotated = self.rotate(np.column_stack(([x1, x2], [y1, y2])), origin=(0,0), angle=angle_to_rotate)
        
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
            filt = ((rotated >= [x_min, y_min - factor*bandwidth]) & (rotated <= [x_max, y_max+ factor*bandwidth])).all(axis=1)
        else:
            filt = ((rotated >= [x_min, y_min]) & (rotated <= [x_max, y_max])).all(axis=1)
        rotated_select = rotated[filt,:]
        
        #Calculated KDE on y coordinates
        y_coords = rotated_select[:,1]
        probability, sample_space = self.density(rotated_select[:,1], bandwidth=bandwidth, kernel=kernel, start=y_min, stop=y_max, resolution=resolution)
        
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
            rotate_back_angle = 2*np.pi - self.to_rotate(angle)
            line_min_rotate_back = self.rotate(line_min, angle=rotate_back_angle)
            line_max_rotate_back = self.rotate(line_max, angle=rotate_back_angle)

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
        
        return probability, sample_space, filt, y_coords
    
    
    def density_count(self, points: np.ndarray, line: list, width: float, nbins: int=20, plot: bool=False, 
                      s: float=2) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate count density of points along a line with a certain width.

        Args:
            points (np.ndarray): Numpy array with X and Y coordinates as columns.
            line (list): Begin and end coordinates of the line over which to 
                calculate the density. Format: [x1, y1, x2, y2].
            width (float): Width over which the density is sampled. Same units as
                the points coordinates.
            kernel (str, optional): Kernel for KDE, see sklearn documentation for
                options. Defaults to 'gaussian'.
            resolution (int, optional): Number of points for KDE sampling.
                Defaults to 50.
            plot (bool, optional): If True it plots the points, line and area over
                which it calculated the density. Defaults to False.
            s (float, optional): Size of dots for plotting. Defaults to 2.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
            Histogram along the line.
            Bin edges.
        """
        
        x1, y1, x2, y2 = line
        
        #get angle of line
        angle = self.get_rotation_rad(x1, y1, x2, y2)

        #Get angle to rotate to turn line straight
        angle_to_rotate = self.to_rotate(angle)

        #Rotate points
        rotated = self.rotate(points, origin=(0,0), angle=angle_to_rotate)
        #Rotate line
        line_rotated = self.rotate(np.column_stack(([x1, x2], [y1, y2])), origin=(0,0), angle=angle_to_rotate)
        
        #Get min an max to select points
        x_min = line_rotated[0][0] - 0.5*width
        x_max = line_rotated[0][0] + 0.5*width
        y_min = line_rotated[0][1]
        y_max = line_rotated[1][1]
        
        #Select points
        filt = ((rotated >= [x_min, y_min]) & (rotated <= [x_max, y_max])).all(axis=1)
        rotated_select = rotated[filt,:]
               
        #Bin the data
        y_coords = rotated_select[:,1]
        hist, bin_edges = np.histogram(y_coords, bins=nbins)
        
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
            rotate_back_angle = 2*np.pi - self.to_rotate(angle)
            line_min_rotate_back = self.rotate(line_min, angle=rotate_back_angle)
            line_max_rotate_back = self.rotate(line_max, angle=rotate_back_angle)

            #Plot the area as a polygon
            rectangle = Polygon(np.row_stack((line_min_rotate_back, np.flipud(line_max_rotate_back))), alpha=0.5)
            ax0.add_patch(rectangle)
            ax0.set_aspect('equal')
            ax0.set_title('Coordinates with sample line')
            
            ax1 = ax[1]
            x = list(map(np.mean, sliding_window_view(bin_edges, window_shape=2)))
            ax1.plot(x, hist)
            ax1.set_title('Histogram')
            ax1.set_ylabel('Count')
            ax1.set_xlabel('Position')
        
        return hist, bin_edges
        
