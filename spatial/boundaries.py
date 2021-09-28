import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from scipy.stats import spearmanr
import warnings
from dask.diagnostics import ProgressBar
import dask
import math
from scipy.ndimage import binary_erosion
from typing import Tuple
from scipy.spatial import distance

def _worker_bisect(points: np.ndarray, grid: np.ndarray, radius: float, 
                   lines: list, n_angles: int):
    """Bisect cloud of points around grid points.
    
    For a set of points and an overlaying grid, find which points are within
    the radius of every grid point. Afterwards it devides the points with a 
    bisection line for various angles and counts the number of points on 
    either side.

    Args:
        points (np.ndarray): Numpy array or Pandas dataframe with XY
            coordinates for points.
        grid (np.ndarray): XY coordinates of grid points.
        radius (float): Search radius around each grid point. Radius can be
            larger than grid spacing.
        lines (list): List with two numpy arrays with the XY coordinates of
            the start and end coordinates of a line that bisects the circle.
            (Output of the self.bisect() function)
        n_angles (int): Number of angles to test.
        
    Returns:
    angle_counts (np.ndarray): For each grid point the count of points above 
        the bisection line.
    count (np.ndarray): Total count for each grid point. Can be used to
        calculate the number of points below the bisection line.
    
    """
    def isabove(p, line):
        "Find points that are above bisection line."
        return np.cross(p-line[0], line[1]-line[0]) < 0

    #Build the tree
    tree = KDTree(points)

    #Query the grid for points within the radius
    index = tree.query_radius(grid, radius)

    #Count number of molecules
    count = np.array([i.shape[0] for i in index])

    #Get coordinates
    coords = [points.iloc[i].to_numpy() for i in index]
    
    #find how many are in each half
    angle_counts = np.zeros((grid.shape[0], n_angles), dtype='int32')

    #Iterate over grid points
    for i, (c,g) in enumerate(zip(coords, grid)):
        if c.shape[0] > 0:
            #Center the points to (0,0)
            cc = c - g

            #Iterate over angles
            for j, l in enumerate(lines):
                angle_counts[i, j] = isabove(cc, l).sum()

    #Return count for each half, and total count
    return angle_counts, count 

def _worker_angle(n_angles: int, d1: np.ndarray, d2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Spearman r for all devisions of the points in a circle.

    Args:
        n_angles (int): Number of angles to devide the circle by.
        d1 (np.ndarray): Count of point above the bisection lines.
        d2 (np.ndarray): Count of points below the bisection lines.

    Returns:
        Tuple[np.ndarray, np.ndarray]: [description]
    """
    r = np.zeros(n_angles)
    dist = np.zeros(n_angles)
    for j in range(n_angles):
        #Catch warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r[j] = (spearmanr(d1[:,j], d2[:,j]))[0]
            dist[j] = distance.euclidean(d1[:,j], d2[:,j])
    
    return r.min(), r.argmin(), dist.max(), dist.argmax()

class Boundaries:
    
    def bisect(self, r: float, a: float, offset=None) -> Tuple[np.ndarray, np.ndarray]:
        """make line that bisects a circle.
        
        Returns the start and end coordinates of a line that bisects a circle
        with its center at (0,0)

        Args:
            r (float): Radius of circle.
            a (float): Angle in degree.
            offset (np.ndarray): Offset from center coordinate, which normally
                is at(0,0) .

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            XY coordinate of start of line.
            XY coordinate of end of line.
        """
        a = np.deg2rad(a)
        p = np.array([np.cos(a) * r, np.sin(a) * r])
        q = -p
        if not offset is None:
            p += offset
            q += offset
            
        return p, q
    
    def square_grid(self, bin_size:float, x_extent:float, y_extent:float, x_min:float, x_max:float, y_min:float, 
                    y_max:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make square grid that matches the shape of a dataset.
        
        Extends the grid to match the extent of the dataset so that the grid 
        point evenly cover the dataset.

        Args:
            bin_size (float): Distance between grid points.
            x_extent (float): X extent of the dataset.
            y_extent (float): Y extent of the dataset.
            x_min (float): X minimimum of dataset.
            x_max (float): X maximum of dataset.
            y_min (float): Y minimum of dataset.
            y_max (float): Y maximum of dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            Grid: Array with grid XY positions.
            Xi: X coordinates in shape of grid.
            Yi: Y coordinates in shape of grid.
        """

        # Create grid values
            #Adjust X extend to make regular grid
        nx = math.ceil(x_extent / bin_size)
        d_x = (nx * bin_size) - x_extent
        xi = np.linspace(x_min - (0.5 * d_x), x_max + (0.5 * d_x), nx)
            #Adjust X extend to make regular grid
        ny = math.ceil(y_extent / bin_size)
        d_y = (ny * bin_size) - y_extent
        yi = np.linspace(y_min - (0.5 * d_y), y_max + (0.5 * d_y), ny)
            #Make grid
        Xi, Yi = np.meshgrid(xi, yi)
        grid = np.column_stack((Xi.ravel(), Yi.ravel()))

        return grid, Xi, Yi

    def boundaries_make(self, bin_size: int = 100, radius: int = 200, n_angles: int = 6,
                normalize: bool = False, normalization_mode: str = 'log', 
                n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate local correlation to investigate border strenth.
        
        Overlays the sample with a grid of points. For each point the molecules
        within the radius are selected. This circle is then devided in two 
        halves, at different angles. The Spearman correlation between the 
        molecule counts of the two halves is correlated and the value and angle
        of the angle with the lowest correlation is returned.
        
        The lower the correlation the stronger the border is, and the angle 
        indicates the direction of the (potential) border.

        Args:
            bin_size (int, optional): The distance between the grid points in 
                the same unit as the dataset. Defaults to 100.
            radius (int, optional): Search radius for asigning molecules to 
                grid points. May be larger than the bin_size. Defaults to 200.
            n_angles (int, optional): Number of angles to test. Defaults to 6.
            normalize (bool, optional): If True normalizes the count data.
                Defaults to False.
            normalization_mode (str, optional): Normalization method to use.
                Defaults to 'log'.
            n_jobs (int, optional): Number of processes to use. If None, 
                the max number of cpus is used. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            results: Array with in the first colum the r values and in the
                second column the angle with the lowest correlation. These are
                only caluclated for grid points that fall within the dataset.
            grid: XY coordinates of all grid points. 
            grid_filt: XY coordinates of valid grid points.
            filt_grid: boolean filter to filter the `grid` to get `grid_filt`
        """
        
        if n_jobs == None:
            n_jobs = self.cpu_count()
        
        #Make the grid overlaying the data       
        grid, Xi, Yi = self.square_grid(bin_size, self.x_extent, self.y_extent, 
                                        self.x_min, self.x_max, self.y_min, self.y_max)

        #make lines that bisect a circle for each required angle
        lines = [self.bisect(radius, a) for a in np.linspace(0, 180 - (180 / n_angles), n_angles)]

        #Find number of molecules in each division
        results = []
        for g in self.unique_genes:
            points = self.get_gene(g)
            y = dask.delayed(_worker_bisect)(points, grid, radius, lines, n_angles)
            results.append(y)

        #Compute
        print('Computation 1 / 2: ', end='\r') 
        with ProgressBar():
            results = dask.compute(*results, scheduler='processes', n_workers=n_jobs)
        print('Computation 2 / 2: ', end='\r') 

        #Identify grid points without molecules
        count_matrix = np.array([i[1] for i in results])
        results_angle_counts = [i[0] for i in results]
        filt = count_matrix.sum(axis=0) > 0
        #Remove contour
        filt_grid = filt.reshape(Xi.shape[0], Xi.shape[1])
        filt_grid = binary_erosion(filt_grid, iterations=math.ceil(radius / bin_size))
        filt = filt_grid.ravel()
        
        #Filter data
        results_angle_counts = [r[filt] for r in results_angle_counts]
        count_matrix = count_matrix[:, filt]
        grid_filt = grid[filt, :]

        #Calculate counts of other half of the circle
        stack = np.stack(results_angle_counts)
        cm_repeat = np.repeat(count_matrix[:,:,np.newaxis], n_angles, axis=2)
        stack_other = cm_repeat - stack

        #Normalize the data
        if normalize:
            #Normalize the count matrix
            cm_norm = self.normalize(pd.DataFrame(count_matrix), mode=normalization_mode).to_numpy()
            cm_norm = np.repeat(cm_norm[:,:,np.newaxis], n_angles, axis=2)
            #Take fraction in hemicircle and multiply with normalized data
            stack = cm_norm * np.nan_to_num((stack / cm_repeat))
            stack_other = cm_norm * np.nan_to_num((stack_other / cm_repeat))

        #Calculate correlation coefficient and pick angle with lowest correlation coefficient
        results2 = []
        for i in range(count_matrix.shape[1]):
            d1 = stack[:,i,:]
            d2 = stack_other[:,i,:]
            z = dask.delayed(_worker_angle)(n_angles, d1, d2)
            results2.append(z)
            
        #Compute
        with ProgressBar():
            results2 = dask.compute(*results2, scheduler='processes')
        results2 = np.array(results2)
        results2 = np.nan_to_num(results2)
        
        #Convert angle index to angle in degree
        angle_used = (np.linspace(0, 180 - (180/n_angles), n_angles))
        results2[:,1] = [angle_used[int(i)] for i in results2[:,1]]
        results2[:,3] = [angle_used[int(i)] for i in results2[:,3]]
        
        return results2, grid, grid_filt, filt_grid