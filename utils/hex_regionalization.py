import math
from itertools import  permutations
import geopandas as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.pyplot import hexbin
from shapely.geometry import  MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage.measure import subdivide_polygon
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from pint import UnitRegistry
from FISHscale.utils.fast_iteration import Iteration
from FISHscale.utils.decomposition import Decomposition
from FISHscale.utils.inside_polygon import inside_multi_polygons
#Mypy types
from typing import Tuple, Union, Any, List
import gc
from memory_profiler import profile
from scipy.spatial import KDTree
from collections import Counter
from matplotlib.patches import Polygon as mpl_polygon
from matplotlib.collections import PatchCollection
import copy
from functools import lru_cache
import colorsys
from sklearn.manifold import TSNE, SpectralEmbedding

class Regionalize(Iteration, Decomposition):
    """Class for regionalization of multidimensional 2D point data.

    """
           
    def hexagon_shape(self, spacing: float, closed=True) -> np.ndarray:
        """Make coordinates of hexagon with the point up.      

        Args:
            spacing (float): Distance between centroid and centroid of 
                horizontal neighbour. 
            closed (bool, optional): If True first and last point will be
                identical. Needed for some polygon algorithms. 
                Defaults to True.

        Returns:
            [np.ndarray]: Array of coordinates.
        """
        
        # X coodrinate 4 corners
        x = 0.5 * spacing
        # Y coordinate 4 corners, tan(30)=1/sqrt(3)
        y = 0.5 * spacing *  (1 / np.sqrt(3))
        # Y coordinate top/bottom
        h = (0.5 * spacing) / (np.sqrt(3)/2)
        
        coordinates = np.array([[0, h],
                                [x, y,],
                                [x, -y],
                                [0, -h],
                                [-x, -y],
                                [-x, y]])
        
        if closed:
            coordinates = np.vstack((coordinates, coordinates[0]))
            
        return coordinates
            
    def hexbin_make(self, spacing: float, min_count: int, n_jobs: int=-1) -> Tuple[Any, np.ndarray, Any]:
        """
        Bin 2D point data with hexagonal bins.
        
        Stores the centroids of the exagons under self.hexagon_coordinates,
        and the hexagon shape under self.hexbin_hexagon_shape.

        Args:
            spacing (float): distance between tile centers, in same units as 
                the data. The function makes hexagons with the point up: ⬡
            min_count (int): Minimal number of molecules in a tile to keep the 
                tile in the dataset.
            n_jobs (int, optional): Number of workers. If -1 it takes the max
                cpu count. Defaults to -1.

        Returns:
            Tuple[pd.DataFrame, np.ndarray, np.ndarray]: 
            Pandas Dataframe with counts for each valid tile.
            Numpy Array with centroid coordinates for the tiles.
            
        """
        #store settings for plotting
        self._hexbin_params = f'spacing_{spacing}_min_count_{min_count}'
        
        #workers
        if n_jobs == -1:
            n_jobs = self.cpu_count
            
        #Get canvas size
        max_x = self.x_max
        min_x = self.x_min
        max_y = self.y_max
        min_y = self.y_min
        
        #Find X range 
        n_points_x = math.ceil((max_x - min_x) / spacing)
        #Correct x range to match whole number of tiles
        full_x_extent = n_points_x * spacing
        difference_x = full_x_extent - self.x_extent
        min_x = min_x - (0.5 * difference_x)
        max_x = max_x + (0.5 * difference_x)
        
        #Find Y range 
        y_spacing = (spacing * np.sqrt(3)) / 2
        n_points_y = math.ceil((max_y - min_y) / y_spacing)
        #Correct y range to match whole number of tiles
        full_y_extent = n_points_y * y_spacing
        difference_y = full_y_extent - self.y_extent
        min_y = min_y - (0.5 * difference_y)
        max_y = max_y + (0.5 * difference_y)
            
        #make hexagonal grid
        x = np.arange(min_x, max_x, spacing, dtype=float)
        y = np.arange(min_y, max_y, y_spacing, dtype=float)
        xx, yy = np.meshgrid(x, y)
        #Offset every second row 
        xx[::2, :] += 0.5*spacing
        coordinates = np.array([xx.ravel(), yy.ravel()]).T
        
        #make KDTree
        tree = KDTree(coordinates)
        
        #Make Results dataframe
        n_genes = len(self.unique_genes)
        n_tiles = coordinates.shape[0]
        df_hex = pd.DataFrame(data=np.zeros((n_genes, n_tiles)),
                            index=self.unique_genes, 
                            columns=[f'{self.dataset_name}_{j}' for j in range(n_tiles)])
        
        #Hexagonal binning of data
        for i, g in enumerate(self.unique_genes):
            data = self.get_gene(g)
            #Query nearest neighbour ()
            dist, idx = tree.query(data, distance_upper_bound=spacing, workers=n_jobs)
            #Count the number of hits
            count = np.zeros(n_tiles)
            counter = Counter(idx)
            count[list(counter.keys())] = list(counter.values())
            #Add to dataframe
            df_hex.loc[g] = count
        
        #make hexagon coordinates
        self.hexbin_hexagon_shape = self.hexagon_shape(spacing, closed=True)
            
        #Filter on number of counts
        filt = df_hex.sum() >= min_count
        df_hex = df_hex.loc[:, filt]
        coordinates = coordinates[filt]
        self.hexbin_coordinates = coordinates

        return df_hex, coordinates
                  
    @lru_cache(maxsize=1)
    def _hexbin_PatchCollection_make(self, params: str):
        """Generate hexbin patch collection for plotting

        Args:
            params (str): The hexbin function saves its parameters under:
                self._hexbin_params. This is used to check if the patch 
                collection needs to be recalculated when the hexbin function 
                has been run with different parameters.

        Returns:
            Matplotlib patch collection
        """
        patches = []
        for i in self.hexbin_coordinates:
            pol = mpl_polygon(self.hexbin_hexagon_shape + i, closed=True)
            patches.append(pol)
        self._hexbin_PatchCollection = PatchCollection(patches)
        return PatchCollection(patches)
                
    def hexbin_plot(self, c: Union[np.ndarray, list], cm=None, ax:Any=None, 
                    figsize=None, save:bool=False, savename:str='',
                    vmin:float=None, vmax:float=None):
        """Plot hexbin results. 

        Args:
            c (np.ndarray, list): Eiter an Array with color values as a float
                between 0 an 1. Or a list of RGB color values.
            cm (plt color map): The color map to use when c is an array.
                Defaults to plt.cm.viridis.
            ax (Any, optional): Axes object to plot on. Defaults to None.
            figsize (tuple): Size of figure if not defined by ax. 
                If None uses: (10,10). Defaults to None.
            save (bool, optional): Save the plot as .pdf. Defaults to False.
            savename (str, optional): Name of the plot. Defaults to ''.
            vmin (float, optional): If c is an Array you can set vmin.
                Defaults to None.
            vmax (float, optional): If c is an Array you can set vmax.
                Defaults to None.
        """
        #Input handling
        colorbar=False
        if ax == None: 
            if figsize == None:
                figsize = (10,10)
            fig, ax = plt.subplots(figsize=figsize)
            colorbar=True
        if cm == None:
            cm = plt.cm.viridis

        #Get PatchCollection, patchCollection can only be added once to a figue so make a deep copy.
        if not hasattr(self, '_hexbin_params'):
            raise Exception('Hexbin has not been calculated yet. Please make using: self.hexbin_make()')
        p = copy.deepcopy(self._hexbin_PatchCollection_make(self._hexbin_params))
        ax.add_collection(p)
        #Set colors from an RGB list
        if type(c) == list:
            p.set_color(c)
            p.set_linewidth(0.1) #To hide small white lines between the polygons
            p.set_edgecolor(c)
            
        #Set colors from an array of values
        else:
            c = c / c.max()
            p.set_array(c)
            p.set_cmap(cm)
            p.set_linewidth(0.1) #To hide small white lines between the polygons
            if vmin!= None and vmax!=None:
                p.set_clim(vmin=vmin, vmax=vmax)
                c_scaled = c - vmin
                c_scaled[c_scaled < 0] = 0
                c_scaled = c_scaled / (vmax - vmin)
                p.set_edgecolor(cm(c_scaled))
            else:
                c_scaled = c - c.min()
                c_scaled = c_scaled / c_scaled.max()
                p.set_edgecolor(cm(c_scaled))

        #Colorbar
        if colorbar:
            fig.colorbar(p, ax=ax)
            
        #Scale
        d = 0.5 * int(self._hexbin_params.split('_')[1])
        ax.set_xlim(self.x_min - d, self.x_max + d)
        ax.set_ylim(self.y_min - d, self.y_max + d)
        ax.set_aspect('equal')
        
        #Save
        if save:
            plt.savefig(f'{savename}_hexbin.pdf')
            
    def hexbin_tsne_plot(self, data = None, tsne:np.ndarray = None, components: int = 2, save:bool=False, savename:str=''):
        """Calculate tSNE on hexbin and plot spatial identities.
        
        Calculates a tSNE and makes a color scheme based on the tSNE 
        coordinates, which is then used to color the spatial hexbin data.
        With 3 components the 3 tSNE dimenstions are used for RGB colors.
        With 2 components the angle and distance are used for HSV colors.

        Args:
            data ([pd.DataFrame]): DataFrame with features in rows and samples
                in columns, like from the self.hexbin_make() function.
                Best to normalize the data and/or apply PCA first.
                Defaults to None.
            tsne (np.ndarray, optional): Pre computed tSNE embedding of hexbin
                data with 2 or 3 components. If not provided, tSNE will be 
                calculated.
            components (int, optional): Nuber of components to calculate the 
                tSNE with, either 2 or 3. Defaults to 2.
            save (bool, optional): Save the plot as .pdf. Defaults to False.
            savename (str, optional): Name of the plot. Defaults to ''.
            
        Retruns:
            tsne (np.ndarray) tSNE coordinates.
        """
        if not (components ==2 or components==3):
            raise Exception(f'Number of components should be 2 or 3, not: {components}')
        
        #Calculate tSNE
        if isinstance(tsne, np.ndarray):
            components = tsne.shape[1]
        else:        
            tsne = TSNE(n_components=components, n_jobs=self.cpu_count).fit_transform(data.T)
               
        #make figure
        fig = plt.figure(figsize=(15,7))
        gs = fig.add_gridspec(2, 4)
        
        if components == 2:
            #Make axes
            ax0 = fig.add_subplot(gs[:, :2])
            ax1 = fig.add_subplot(gs[:, 2:])
            
            #Use rotation and distance from centroid to calculate HSV colors. 
            #Calculate rotation
            rotation = np.array([self.get_rotation_rad(0,0, i[0], i[1]) for i in tsne])
            rotation = (rotation + np.pi) / (2 * np.pi)
            #Calculate distance
            origin= np.array([0,0])
            dist = np.array([np.linalg.norm(origin - i) for i in tsne])
            dist = dist / dist.max()
            dist = (dist + 0.5) / 1.5
            #Make colors
            c = np.column_stack((rotation, dist, np.ones(dist.shape[0])))
            c = [colorsys.hsv_to_rgb(i[0], i[1], i[2]) for i in c]
            
            #plot tSNE
            ax0.scatter(tsne[:,0], tsne[:,1], s=3, c=c)
            ax0.set_aspect('equal')
            ax0.set_axis_off()
            ax0.set_title('tSNE', fontsize=14)
            
            #plot spatial
            self.hexbin_plot(c, ax=ax1)
            ax1.set_axis_off()
            ax1.set_title('Spatial', fontsize=14)
            
        if components == 3:
            #Make axes
            ax0_0 = fig.add_subplot(gs[0,0])
            ax0_1 = fig.add_subplot(gs[0,1])
            ax0_2 = fig.add_subplot(gs[1,0])
            ax1 = fig.add_subplot(gs[:, 2:])
            
            #Make colors Translate tSNE coordinates into RGB
            c = tsne + np.abs(tsne.min(axis=0))
            c = c / c.max(axis=0)
            c = [list(i) for i in c]
            
            #tSNE
            ax0_0.scatter(tsne[:,0], tsne[:,1], s=1, c=c)
            ax0_0.set_xlabel('tSNE 0')
            ax0_0.set_ylabel('tSNE 1')
            ax0_0.spines['right'].set_visible(False)
            ax0_0.spines['top'].set_visible(False)
            ax0_0.set_title('tSNE', fontsize=14)
            
            ax0_1.scatter(tsne[:,1], tsne[:,2], s=1, c=c)
            ax0_1.set_xlabel('tSNE 1')
            ax0_1.set_ylabel('tSNE 2')
            ax0_1.spines['right'].set_visible(False)
            ax0_1.spines['top'].set_visible(False)
            ax0_1.set_title('tSNE',fontsize=14)
            
            ax0_2.scatter(tsne[:,0], tsne[:,2], s=1, c=c)
            ax0_2.set_xlabel('tSNE 0')
            ax0_2.set_ylabel('tSNE 2')
            ax0_2.spines['right'].set_visible(False)
            ax0_2.spines['top'].set_visible(False)
            ax0_2.set_title('tSNE', fontsize=14)
            
            #Spatial
            self.hexbin_plot(c, ax=ax1)
            ax1.set_axis_off()
            ax1.set_title('Spatial', fontsize=14)

        plt.tight_layout()
        
        #Save
        if save:
            plt.savefig(f'{savename}_hexbin_tsne.pdf')
        
        return tsne
        
    def hexbin_decomposition_plot(self, data, components:list = [0, 9], save: bool=False, savename:str=''):
        """Plot decomposition components spatially.
        
        Usefull to plot PCA or LDA components spatially. 

        Args:
            data ([np.ndarray, pd.DataFrame]): Array with components in
                columns. 
            components (list, optional): List with range of components to plot.
                Defaults to [0, 9].
            save (bool, optional): Save the plot as .pdf. Defaults to False.
            savename (str, optional): Name of the plot. Defaults to ''.
        """
        
        n = components[1] - components[0]
        n_grid = math.ceil(np.sqrt(n))
        
        fig, axes = plt.subplots(figsize=(3*n_grid, 3*n_grid), ncols=n_grid, nrows=n_grid)

        #Plot components
        for i in range(n):
            j = i + components[0]
            ax = axes[int(i/n_grid), i%n_grid]
            self.hexbin_plot(c = data[:,j] / data[:,j].max(), ax=ax)
            ax.set_aspect('equal')
            ax.set_title(j)
            ax.axis('off')
        
        #Hide unused axes   
        n_res = (n_grid **2) - n
        for i in range(n_res):
            i += components[1]
            ax = axes[int(i/n_grid), i%n_grid]
            ax.axis('off')
            
        plt.tight_layout()
        
        #Save
        if save:
            plt.savefig(f'{savename}_hexbin_decomposition_[{components[0]}, {components[1]}].png', dpi=200)

    def clust_hex_connected(self, df_hex, hex_coord: np.ndarray, distance_threshold: float = None, 
                            n_clusters:int = None, neighbor_rings:int = 1, n_jobs:int=-1) -> np.ndarray:
        """Cluster hex-bin data, with a neighborhood embedding.

        Clusters with AggolmerativeClustering that uses a distance matrix 
        made from the tiles and their neighbours within the neighbour_radius.
        Can either cluster with a pre-defined number of resulting clusters
        by passing a number to "n_clusters", or clusters with a 
        "distance_threshold" that determines the cutoff. When passing
        multiple datasets that require a different number of clusters the
        "distance_threshold" will be more suitable.

        Args:
            df_hex (pd.Dataframe): Pandas Dataframe with molecule counts for
                each hexagonal tile
            hex_coord (np.array): Numpy Array with XY coordinates of the
                centroids of the hexagonal tiles. 
            distance_threshold (float, optional): Distance threshold for Scipy
                Agglomerative clustering. Defaults to None.
            n_clusters (int, optional): Number of desired clusters. 
                Defaults to None.
            neighbor_rings (int, optional): Number of rings around a central 
                tile to make connections between tiles for Agglomerative 
                Clustering with connectivity. 1 means connections with the 6 
                imediate neighbors. 2 means the first and second ring, making 
                18 neigbors, etc.. Defaults to 1.

        Raises:
            Exception: If "distance_threshold" and "n_clusters" are not 
                properly defined

        Returns:
            [np.array]: Numpy array with cluster labels.

        """
        #Input check
        if distance_threshold!=None and n_clusters!=None:
            raise Exception('One of "distance_threshold" or "n_clusters" should be defined, not both.')
        if distance_threshold==None and n_clusters==None:
            raise Exception('One of "distance_threshold" or "n_clusters" should be defined.')

        #Make graph to connect neighbours 
        n_neighbors = (neighbor_rings * 6)
        Kgraph = kneighbors_graph(hex_coord, n_neighbors, include_self=False, n_jobs=n_jobs)

        #Cluster
        clust_result = AgglomerativeClustering(n_clusters=n_clusters,
                                               distance_threshold=distance_threshold, 
                                               compute_full_tree=True, 
                                               connectivity=Kgraph,
                                               linkage='ward').fit(df_hex)
        #Return labels
        return clust_result.labels_

    def smooth_hex_labels(self, hex_coord: np.ndarray, labels: np.ndarray, neighbor_rings: int = 1, cycles: int = 1,
                        return_all: bool = False, n_jobs: int = -1) -> Union[np.ndarray, list]:
        """Smooth labels of a hexagonal tile matrix by neighbour majority vote.

        For each tile, the identity is set to the most predominant label in 
        its local environment, set by the neighbour radius.
        The original label is taken into account. If a tie is encountered,
        the label is set to the original label that was the input of that
        smoothing round.

        Args:
            hex_coord (np.array): Numpy Array with XY coordinates of the
                centroids of the hexagonal tiles.
            labels (np.ndarray): Numpy Array with cluster labels.
            neighbor_rings (int, optional): Number of rings around a central 
                tile to smooth over. 1 means connections with the 6 
                imediate neighbors. 2 means the first and second ring, making 
                18 neigbors, etc. Defaults to 1.
            cycles (int, optional): Number of smoothing cycles. Defaults to 1.
            retrun_all (bool, optional): If True, return results of every 
                smoothing round as a list of arrays, including the original 
                labels. If False, it just returns the last round. 
                Defaults to False.
            n_jobs (int, optional): Number of jobs to use for generating the 
                neighbourhood graph. -1 means as many jobs as CPUs.
                Defaults to -1.

        Returns:
            [np.ndarray]: Numpy Array with the smoothed cluster labels. 

        """
        def smooth(Kgraph, label):
            """Smooth labels with neigbouring labels"""
            new_label = []
            for i, l in enumerate(label):
                #Get labels of neigbours and select most predominant label(s).
                neighbour_labels = label[Kgraph[i].indices]
                values, counts = np.unique(neighbour_labels, return_counts=True)
                predominant_label = values[counts == counts.max()]
                #Check if there is a tie between neighbouring labels
                if predominant_label.shape[0] > 1:
                    #There is a tie, set to original label
                    new_label.append(l)
                else:
                    #Set to new label
                    new_label.append(predominant_label[0])        
            return np.array(new_label)

        n_neighbors = 1 + (neighbor_rings * 6)
        Kgraph = kneighbors_graph(hex_coord, n_neighbors, include_self=True, n_jobs=n_jobs)

        results = [labels]
        for iteration in range(cycles):
            results.append(smooth(Kgraph, results[-1]))
        
        if return_all:
            return results
        else:
            return results[-1]

    def cluster_mean_make(self, df_hex: Any, labels: np.ndarray) -> Any:
        """Calculate cluster mean.

        For a DataFrame with samples in columns, calculate the mean expression
            values for each unique label in labels.

        Args:
            df_hex (pd.DataFrame): Pandas DataFrame with samples in columns.
            labels (np.ndarray): Numpy array with cluster labels

        Returns:
            [pd.DataFrame]: Pandas Dataframe with mean values for each label.

        """
        unique_labels = np.unique(labels)
        cluster_mean = pd.DataFrame(data=np.zeros((df_hex.shape[0], len(unique_labels))), index = df_hex.index,
                                    columns=unique_labels)

        #Loop over clusters
        for l in unique_labels:
            filt = labels == l
            #Get mean expression of cluster
            cluster_mean.loc[:, l] = df_hex.loc[:, filt].mean(axis=1)

        return cluster_mean

    def cluster_sum_make(self, df_hex, labels: np.ndarray) -> Any: 
        """Calculate cluster sum.

        For a DataFrame with samples in columns, calculate the sum expression
        counts for each unique label in labels.

        Args:
            df_hex (pd.DataFrame): Pandas DataFrame with samples in columns.
            labels (np.ndarray): Numpy array with cluster labels

        Returns:
            [pd.DataFrame]: Pandas Dataframe with sum values for each label.

        """
        unique_labels = np.unique(labels)
        cluster_sum = pd.DataFrame(data=np.zeros((df_hex.shape[0], len(unique_labels))), index = df_hex.index,
                                    columns=unique_labels)

        #Loop over clusters
        for l in unique_labels:
            filt = labels == l
            #Get mean expression of cluster
            cluster_sum.loc[:, l] = df_hex.loc[:, filt].sum(axis=1)

        return cluster_sum

    def hex_neighbour_coord(self, x:float, y:float, s:float) -> np.ndarray:
        """Calculate cartesian coordinates of neighbour centroids.

        Assumes hexagonal grid with point of hexagons up: ⬡

        Args:
            x (float): X coordinate of center point.
            y (float): Y coordinate of center point.
            s (float): Distance between center points.

        Returns:
            np.ndarray: Numpy array with XY coordinates of neighbouring hexagon
                centroids. 

        """
        delta_y = 0.5 * s * math.sqrt(3) #(0.5 * s) / tan(30)
        neighbour_coord = np.array([[x-0.5*s, y + delta_y],
                                    [x+0.5*s, y + delta_y],
                                    [x-s, y],
                                    [x+s, y],
                                    [x-0.5*s, y - delta_y],
                                    [x+0.5*s, y - delta_y],])
        return neighbour_coord

    def get_rotation_deg(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculates the rotation of a vector between 2 points and the origin.

        Args:
            x1 (float): X coordinate of point 1.
            y1 (float): Y coordinate of point 1.
            x2 (float): X coordinate of point 2.
            y2 (float): Y coordinate of point 2.

        Returns:
            float: Angle in degrees
        
        """
        dx = x2 - x1
        dy = y2 - y1
        
        angle = np.rad2deg(math.atan2(dy, dx))

        return angle
    
    def hex_get_shared_corners(self, angle, corner_dict, x=0, y=0):
        """
        Get the corners that are shared with a neighbouring tile.

        Input:
        `angle`(int): Angle the centroid of the neighbouring tile makes
            with the tile of interest, relative to the origin at (0,0)
            Uses the "shared_corner" dictionary.
        `x`(float): centroid x coordinate of tile of interest.
        `y`(float): centroid y coordinate of tile of interest.
        Returns:
        Coordinates of shared corner points.

        """
        #Dictionary coupling angle with neighbour to angle of shared corners
        shared_corner = {0: [-30, 30],
                        60: [30, 90],
                        120: [90, 150],
                        180: [150, -150],
                        -120: [-150, -90],
                        -60: [-90, -30]}
        
        shared_angles = shared_corner[angle]        
        corner1 = (corner_dict[shared_angles[0]]) + [x, y]
        corner2 = (corner_dict[shared_angles[1]]) + [x, y]
        return corner1, corner2
    
    def find_nearest_value(self, array, value):
        """Find the nearest value from an given array."""
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def hex_region_boundaries(self, hex_coord: np.ndarray, hexagon_shape: Any, hexbin_spacing: float, labels: np.ndarray, 
                              decimals: int = 7) -> dict:
        """ Find border coordinates of regions in a hexagonal grid.

        Finds the border coordinates for each connected group of hexagons 
        represented by a unique label. Assumes hexagonal grid with point up: ⬡

        Args:
            hex_coord (np.ndarray): Numpy Array with centroid coordinates as XY
                columns for all tiles in a hexagonal grid. 
            hexagon_shape (matplotlib polygon): polygon path object for the 
                hexagon shape.
            hexbin_spacing (float): Centroid spacing used to make the hexbin
                plot.
            labels (np.ndarray): Array with cluster labels for each tile
            decimals (int): Due to inaccuracies in the original hexagon tile 
                generation and rounding errors, the points need to be rounded 
                to return all unique points. If you experience errors with the 
                generation of polygons downstream, lower the number of 
                decimals. Default suggestion: 7

        Returns:
            dict: Dictionary with for each label the boundary point of the 
                hexagonal tile as Numpy Array. 
        
        """
        angle_array_corners = np.array([30, 90, 150, -150, -90, -30])
        angle_array_neighbours = np.array([0, 60, 120, 180, -120, -60])

        #make result dictionary
        boundary_points = {l:[] for l in np.unique(labels)}

        #Determine corner locations relative to tile centroid
        #Because of small devitions of a perfect hexagonal grid it is best to get the hexagon border point form the 
        #Matplotlib Polygon.
        corner = {}
        for c in hexagon_shape[:6]: #There are points in the polygon for the closing vertice.

            angle = self.find_nearest_value(angle_array_corners, self.get_rotation_deg(0,0, c[0], c[1]))
            corner[angle] = c

        #Find neighbours
        neighbour_radius = hexbin_spacing + (0.1 * hexbin_spacing)
        Kgraph = radius_neighbors_graph(hex_coord, neighbour_radius, include_self=False)
        
        #Iterate over all tiles
        for i, l in enumerate(labels):
            #Find neignbour identities
            indices = Kgraph[i].indices
            neighbour_identity = labels[indices]

            #Check if tile has neighbours of different identity
            if not np.all(neighbour_identity == l) or len(indices) < 6:
                own_coord = hex_coord[i]
                neighbour_coords = hex_coord[indices]
                #Find the angles of the neigbouring tiles
                angles = np.array([self.find_nearest_value(angle_array_neighbours, self.get_rotation_deg(own_coord[0], own_coord[1], c[0], c[1])) for c in neighbour_coords])

                #Iterate over all possible neighbour angles
                to_add = []
                for a in [0, 60, 120, 180, -120, -60]:
                    #Check if a neighbour exists
                    if a in angles:
                        angle_index = np.where(angles == a)
                        #Check if this neighbour is of a different identity
                        if neighbour_identity[angle_index] != l:
                            to_add.append(a)
                    #No there is no neigbour at that angle, this is a border tile   
                    else:
                        to_add.append(a)
                #Get the shared corner point for neighbours that are missing or have a different identity      
                for a2 in to_add:
                    c1, c2 = self.hex_get_shared_corners(a2, corner, x=own_coord[0], y=own_coord[1])
                    boundary_points[l].append(c1)
                    boundary_points[l].append(c2)
                    #Add intermediate point to help making circles
                    boundary_points[l].append((np.mean([c1, c2], axis=0)))
                    #boundary_points[d][l].append([])

        #Clean duplicated points
        for l in np.unique(labels):
            boundary_points[l] = np.unique(np.array(boundary_points[l]).round(decimals=decimals), axis=0)

        return boundary_points

    def polygon_order_points(self, boundary_points: dict) -> dict:
        """Order set of point for making polygons.

        This function makes a network out of a set of points, and uses this to 
        find circles, in order to order the points to make a valid polygon. It 
        assumes that for any point its 2 closest neighbours are the previous 
        and next points in the polygons. If this is not the case the algorithm 
        will make shortcuts and not return all data.
        Closes the polygon by adding the first point to the end of the array.

        Args:
            boundary_points (dict): Dictunary with for each label the boundary
                point of the hexagonal tile as Numpy Array. Polygon should not
                be closed, i.e. no duplication of first point. 

        Returns:
            dict: Dictionary in the same shape as the input but now the points
                are ordered. Will close the Polygon, meaning that the first and
                last point are identical.

        """      
        results = {l:[] for l in boundary_points.keys()}
        #Loop over labels
        for l in boundary_points.keys():
            label_boundaries = boundary_points[l]
            #Connect closest 2 points to make a circle
            Kgraph = kneighbors_graph(label_boundaries, 2)
            G = nx.Graph()
            #Iterate over all nodes
            for i in range(boundary_points[l].shape[0]):
                #Find neignbour indices
                indices = Kgraph[i].indices
                #Add edges to graph
                for j in indices:
                    G.add_edge(i,j)

            #Find connected components in G
            S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
            for i in range(len(S)):
                #Find cycle to order the points
                cycle = np.array(nx.cycles.find_cycle(S[i])) 
                polygon = label_boundaries[cycle[:,1],:]
                #Close the polygon
                polygon = np.vstack((polygon, polygon[0]))
                results[l].append(polygon)
        
        return results

    def polygon_smooth_points(self, ordered_points: dict, degree: int=2) -> dict:
        """Smooth polygon point using skimage.measure.subdivide_polygon().

        This is not a very strong smoothing on hexagon border data. 
        Warning: Will cause neighbouring polygons to detached in corners were 3 
        polygons meet.

        Args:
            ordered_points (dict): Dictionary with datasets as keys. For each 
                dataset a dictionary with labels as keys, with a list of 
                polygon coordinates.
            degree (int, optional): Degree of B-spline smooting. Max 7.
                Defaults to 2.

        Returns:
            dict: Smoothed points in same format as the input.

        """
        results = {l : [] for l in ordered_points.keys()}

        #Loop over labels
        for l in ordered_points.keys():
            #Loop over polygons
            for pol in ordered_points[l]:
                results[l].append(subdivide_polygon(pol, degree=degree, preserve_ends=True))
    
        return results
        
    def to_Shapely_polygons(self, ordered_points: dict) -> dict:
        """Make Shapely polygons out of a set of ordered points.

        Converts orderd points to polygons, and makes complex polygons 
        (MulitPolygons) if a region consists of multiple polygons. These sub-
        polygons can be inside the main polygon, in which case a hole is cut.

        Args:
            ordered_points (dict): Dictionary with keys for each label and a 
                list of orderd points for each polygon.

        Returns:
            dict: Dictionary with for every label a Shapely Polygon, or a 
                MultiPolygon if the region consists of multiple polygons.

        """
        #datasets = list(ordered_points.keys())
        results = {}

        #Loop over labels
        for l in ordered_points.keys():
            polygons = []
            clean_polygons = []
            #Loop over point sets and make polygons
            for circle in ordered_points[l]:
                polygons.append(Polygon(circle))         
            
            #Check if a polygon contains another polygon
            if len(polygons) > 1:
            
                r_polygons = range(len(polygons))    
                intersections = {i:[] for i in r_polygons}
                interior = []
                for i,j in permutations(r_polygons, 2):
                    if polygons[i].contains(polygons[j]):
                        intersections[i].append(j)
                        interior.append(j)

                #Clean the polygons by making holes for the interior polygons
                for i in r_polygons:
                    inside = intersections[i]
                    if inside != []:
                        for j in inside:
                            intersect = unary_union(polygons[i].intersection(polygons[j]))
                            polygons[i] = unary_union(polygons[i].difference(intersect))
                        clean_polygons.append(polygons[i])
                    elif i not in interior:
                        clean_polygons.append(polygons[i])
                    else:
                        pass
                    
                if len(clean_polygons) > 1:
                    clean_polygons = MultiPolygon(clean_polygons)
                else:
                    clean_polygons = clean_polygons[0]                
                results[l] = clean_polygons
            #No this polygon is whole            
            else:
                results[l] = polygons[0]
        
        return results

    def geoSeries_make(self, polygons: dict) -> Any:
        """Convert dictionary with shapely polygons to geoPandas GeoSeries.

        Args:
            polygons ([type]): Dictionary with for every label a Shapely 
                Polygon, or a MultiPolygon if the region consists of multiple 
                polygons.

        Returns:
            GeoSeries: Geopandas series of the polygons.

        """
        return gp.GeoSeries(polygons, index=list(polygons.keys()))

    def geoDataFrame_make(self, data: np.ndarray, index: Union[List[str], np.ndarray], 
                          columns: Union[List[str], np.ndarray], geometry: Any) -> Any:
        """Construct geoDataFrame. 

        Args:
            data (np.ndarray): Array with data.
            index (Union[list[str], np.ndarray]): Dataframe region index labels
            columns (Union[list[str], np.ndarray]): Dataframe Column gene 
                labels. 
            geometry (gp.geoSeries): geoSeries of the polygons of each Index

        Returns:
            [gp.geoDataFrame]

        """
        gdf = gp.GeoDataFrame(data=data, index=index, columns=columns, geometry=geometry)
        return gdf

    def bounding_box_get(self, polygon: np.ndarray) -> np.ndarray:
        """Get the bounding box of a single polygon.

        Args:
            polygon (np.ndarray): Array of X, Y coordinates of the polython as
                columns.

        Returns:
            np.ndarray: Array with the Left Bottom and Top Right corner 
                coordinates: np.array([[X_BL, Y_BL], [X_TR, Y_TR]])

        """
        xmin, ymin = polygon.min(axis=0)
        xmax, ymax = polygon.max(axis=0)
        return np.array([[xmin, ymin], [xmax, ymax]])

    def bounding_box_make(self, polygon_points: dict) -> dict:
        """Find bounding box coordinates of dictionary with polygon points.

        Args:
            polygon_points (dict): Dictionary with labels of each (multi-) 
                polygon and a list of (sub-)polygon(s) as numpy arrays.

        Returns:
            dict: Dictionary in the same shape as input, but for every (sub-)
                polygon the bounding box.
        
        """
        results = {}
        labels = list(polygon_points.keys())
        #Loop over labels
        for l in labels:
            results[l] = []
            #Loop over polygon list
            for poly in polygon_points[l]:
                results[l].append(self.bounding_box_get(poly))
        return results

    def bbox_filter_points(self, bbox: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Filter point that fall within bounding box.

        Args:
            bbox (np.ndarray): Array with the Left Bottom and Top Right corner 
                coordinates: np.array([[X_BL, Y_BL], [X_TR, Y_TR]])
            points (np.ndarray): Array with X and Y coordinates of the points
                as columns

        Returns:
            np.ndarray: Boolean array with True for points that are in the 
                bounding box.

        """
        filt_x = np.logical_and(points[:,0]>=bbox[0][0], points[:,0]<=bbox[1][0])
        filt_y = np.logical_and(points[:,1]>=bbox[0][1], points[:,1]<=bbox[1][1])
        return np.logical_and(filt_x, filt_y)

    def point_in_region(self, polygon_points: dict, normalize: bool = True, normalize_unit: str = 'millimeter') -> Any:
        """Make GeoPandas Dataframe of region and count points inside.

        Takes a dictionary of (Multi-)polygons and converts these to Shapely
        (Mulit-)Polygons. Then it takes the point data and counts the points
        that fall inside each region. The results are stored as GeoPandas 
        geoDataFrame, that contains the counts for all genes in each region.
        Optionally normalizes for area. 

        Args:
            polygon_points (dict): Dictionary with keys for each label and a 
                list of orderd points for each polygon.
            normalize (bool, optional): If True normalizes count by area. 
                Defaults to True.
            normalize_unit (str, optional): Unit of the normalization. Defaluts
                to "milimeter", which means that data will be normalized by 
                square milimeter. 

        Returns:
            [gp.GeoDataFrame]: geoDataFrame with (normalized) counts for each
            gene in each region. Every region has a Shapely (Multi-)Polygon in
            the geometry column.

        """
        #Make base geopandas dataframe
        polygons = self.to_Shapely_polygons(polygon_points)
        gs = self.geoSeries_make(polygons)
        labels = list(polygon_points.keys())
        gdf = self.geoDataFrame_make(data=np.zeros((len(labels), len(self.unique_genes)), dtype='int64'), index=labels, 
                                            columns=self.unique_genes,  geometry=gs)

        #Recount which points fall in which (multi-)polygon
        df = pd.DataFrame(data = self.df.g, columns = ['gene'])
        points = self.df.loc[:, ['x', 'y']].compute().to_numpy() #Loading all points in RAM
        for l, filt in inside_multi_polygons(polygon_points, points): 
            #get the sum of every gene and asign to gene count for that area. 
            gdf.loc[l, self.unique_genes] = df[filt].groupby('gene').size() 
        del points

        #Normalize data    
        if normalize:
            area_in_original_unit = gdf.area
            conversion = self.area_scale.to(self.ureg(normalize_unit) ** 2)
            area_in_desired_unit = area_in_original_unit * conversion.magnitude
            gdf.iloc[:,:-1] = gdf.iloc[:,:-1].divide(area_in_desired_unit, axis=0)  
            self.vp(f'Points per area normalized per {conversion.units}')          

        return gdf
    
    def regionalize(self, spacing: float, 
                        min_count: int,
                        normalization_mode: str = 'APR',
                        dimensionality_reduction: str = 'PCA', 
                        n_components: list = [0,100],
                        clust_dist_threshold: float = 70, 
                        clust_neighbor_rings: int = 1,
                        smooth: bool = False,
                        smooth_neighbor_rings: int = 1, 
                        smooth_cycles: int = 1,
                        n_jobs=-1) -> Union[Any, np.ndarray, np.ndarray, Any]:
        """Regionalize dataset.
        
        Performs the following steps:
        - Hexagonal binning
        - Normalization
        - Dimensionality reduction (PCA or LDA)
        - Clustering
        - Optional smoothing
        - Calculating mean region expression matrix

        Args:
             spacing (float): distance between tile centers, in same units as 
                the data. The function makes hexagons with the point up: ⬡
            min_count (int):  Minimal number of molecules in a tile to keep the 
                tile in the dataset. The algorithm will generate a lot of empty 
                tiles, which are later discarded using the min_count threshold.
                Suggested to be at least 1.
            normalization_mode (str, optional):Normalization method. Choose 
                from: "log", "sqrt",  "z", "APR" or None. for log +1 transform,
                square root transform, z scores or Analytic Pearson residuals
                respectively. Also possible to not normalize, in which case the
                input should be None. Usefull for LDA. Defaults to 'APR'.
            dimensionality_reduction (str, optional): Method for dimentionality
                reduction. Implmented PCA, LDA. Defaults to 'PCA'.
            n_components (list, optional): Components to use for clustering.
                In some cases the PCA component 0 signifies to total expression
                which should be excluded for clustering. Defaults to [0, 100].
            clust_dist_threshold (float, optional): Distance threshold for 
                Scipy Agglomerative clustering. Defaults to 70.
            clust_neighbor_rings (int, optional): Number of rings around a 
                central tile to make connections between tiles for 
                Agglomerative Clustering with connectivity. 1 means connections
                with the 6 imediate neighbors. 2 means the first and second 
                ring, making 18 neigbors, etc. Defaults to 1.
            smooth (bool, optional): If True performs label smoothing after
                clustering. Defaults to False.
            smooth_neighbor_rings (int, optional):  Number of rings around a 
                central tile to smooth over. 1 means connections with the 6 
                imediate neighbors. 2 means the first and second ring, making 
                18 neigbors, etc. Defaults to 1.
            smooth_cycles (int, optional): Number of smoothing cycles.
                Defaults to 1.
            n_jobs (int, optional): Number op processes. If -1 uses the max 
                number of CPUs. Defaults to -1.

        Returns:
            Union[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, 
                pd.DataFrame]: Tuple containing:
                - df_hex: Dataframe with counts for each hexagonal tile.
                - labels: Numpy array with cluster labels for each tile.
                - hex_coord: XY coordinates for each hexagonal tile.
                - df_mean: Dataframe with mean count per region.
                - df_norm: Dataframe with mean normalized count per region.
        """
        #Bin the data with a hexagonal grid
        df_hex, hex_coord = self.hexbin_make(spacing, min_count, n_jobs=n_jobs)
        
        #Normalize data
        df_hex_norm = self.normalize(df_hex, mode=normalization_mode)
        
        #Dimensionality reduction
        if dimensionality_reduction.lower() == 'pca':
            #Calculate PCA
            dr = self.PCA(df_hex_norm.T)
        elif dimensionality_reduction.lower() == 'lda':
            #Calculate Latent Dirichlet Allocation
            dr = self.LDA(df_hex_norm.T, n_components=n_components, n_jobs=n_jobs)
            
        #Cluster dataset
        labels = self.clust_hex_connected(dr[:, n_components[0] : n_components[1]], hex_coord, 
                                          distance_threshold=clust_dist_threshold, 
                                          neighbor_rings=clust_neighbor_rings, n_jobs=n_jobs)
        
        #Spatially smooth cluster labels
        if smooth:
            labels = self.smooth_hex_labels(hex_coord, labels, smooth_neighbor_rings, smooth_cycles, n_jobs=n_jobs)
        
        #make mean expression
        df_mean = self.cluster_mean_make(df_hex, labels)
        df_norm = self.cluster_mean_make(df_hex_norm, labels)
        
        return df_hex, labels, hex_coord, df_mean, df_norm

    def geopandas_make(self, spacing: float, 
                            df_hex: Any,
                            labels: np.ndarray,
                            hex_coord: np.ndarray,
                            boundary_decimals: int = 7,
                            smooth_polygon: bool = False, 
                            smooth_polygon_degree:int = 7, 
                            recount: bool = False,
                            area_normalize: bool = True, 
                            area_normalize_unit: str = 'millimeter') -> None:
        """Make a GeoPandas dataframe of clustered hexbin data.
        
        Output is a GeoPandas dataframe which contains the (normalized) counts
        per region and a polygon for each region.

        Args:
            spacing (float): Spacing used to generate the hexbins.
            df_hex (pandas DataFrame): Dataframe with features in rows and 
                hexagonal tiles in columns with count values.
            labels: (np.ndarray): Cluster labels matching the number of tiles.
            hex_coord (np.ndarray): Array with XY coordinates of the hexagonal
                tiles.
            boundary_decimals (int, optional): Due to inaccuracies in the 
                original hexagon tile  generation and rounding errors, the 
                points need to be rounded to return all unique points. If you 
                experience errors with the  generation of polygons downstream, 
                lower the number of decimals. Defaults to 7.
            smooth_polygon (bool, optional): Whether or not smooting of the 
                polygons is performed. Advised not to smooth, because smoothing
                has little effect and can cause holes inbetween polygons where
                3 polygons meet. Defaults to False.
            smooth_polygon_degree (int, optional): Degreen by which to smooth.
                Range between 1 and 7, wehere 7 causes the most smothing. 
                Defaults to 7.
            recount (bool, optional): Recount which points fall in which
                polygon. Uses paralellization to speed up, so not thread safe!
                Defaults to False.
            area_normalize (bool, optional): If True normalizes the counts by 
                area. Defaults to True.
            area_normalize_unit (str, optional): Unit to use for normalization.
                Use the 1D unit: "micrometer" will mean sqare micrometer.
                Defaults to 'micrometer'.

        Retruns:
            [None]: Data saved as self.regions. 

        """
        #Get boundary points of regions
        boundary_points = self.hex_region_boundaries(hex_coord, self.hexbin_hexagon_shape, spacing, labels, decimals = boundary_decimals)
        
        #Order boundary point for making polygons
        ordered_points = self.polygon_order_points(boundary_points)
        
        #Smooth boundary points
        if smooth_polygon == True:
            ordered_points = self.polygon_smooth_points(ordered_points, degree = smooth_polygon_degree)
            if recount == False:
                recount = True
                print('Recount set to True after "Smooth_polygon" was set to True.')
        
        #Recount points in polygons and make geoDataFrame
        if recount == True:
            gdf = self.point_in_region(ordered_points, normalize=area_normalize, normalize_unit=area_normalize_unit)
            return gdf
        
        #make geoDataFrame with region data
        else:
            sum_counts = self.cluster_sum_make(df_hex, labels).T #Transpose because geopandas wants regions in index
            shapely_polygons = self.to_Shapely_polygons(ordered_points)
            new_geoseries = self.geoSeries_make(shapely_polygons)

            if area_normalize:
                area_in_original_unit = new_geoseries.area
                conversion = self.area_scale.to(self.ureg(area_normalize_unit) ** 2)
                area_in_desired_unit = area_in_original_unit * conversion.magnitude
                sum_counts = sum_counts.divide(area_in_desired_unit, axis=0)
                self.vp(f'Points per area normalized per {conversion.units}')

            gdf = self.geoDataFrame_make(data = sum_counts.to_numpy(), index = sum_counts.index, 
                                                 columns = sum_counts.columns, geometry=new_geoseries)
            
            return gdf
            

