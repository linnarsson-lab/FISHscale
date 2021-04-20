import math
from itertools import  permutations
from sklearn.decomposition import PCA
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
from FISHscale.utils.inside_polygon import inside_multi_polygons
#Mypy types
from typing import Tuple, Union, Any, List

class regionalize(Iteration):
    """Class for regionalization of multidimensional 2D point data.

    """
    def __init__(self, 
        filename: str,
        x: str = 'r_px_microscope_stitched',
        y: str = 'c_px_microscope_stitched',
        gene_column: str = 'below3Hdistance_genes',
        other_columns: list = [],
        unique_genes: np.ndarray = None,
        pixel_size: str = None,
        verbose: bool = False):
        """Class for regionalization of 2D point data with hexagonal tiles.

        Args:
            filename (str): Path to the pandas dataframe saved in .parquet 
                format with point location and label.
            x (str, optional): Name of the column with the X coordinates.
                Defaults to 'r_px_microscope_stitched'.
            y (str, optional): Name of the column with the Y coordinates.
                Defaults to 'c_px_microscope_stitched'.
            gene_column (str, optional): Name of the column with the gene 
                labels for every point. Defaults to 'below3Hdistance_genes'.
            other_columns (list, optional): List of other columns. 
                Defaults to [].
            unique_genes (np.ndarray, optional): Numpy array with unique gene
                names. If not defined, the unique genes will be found but this
                can take time for milions of points.
            pixel_size (str, optional): Input for Pint's UnitRegistry. Like:
                "5 micrometer" will result in a pixel_size of 5 with the 
                correct unit. See Pint documentation for help. 
                Defaults to None.
            verbose (bool, optional): If True prints additional output.

        """
        self.filename = filename
        self.dataset_name = self.filename.split('/')[-1].split('.')[0]
        self.x, self.y = x, y
        self.data = pd.read_parquet(filename)
        self.gene_column = gene_column
        self.other_columns = other_columns
        if not isinstance(unique_genes, np.ndarray):
            self.unique_genes = np.unique(self.data[self.gene_column])
        else:
            self.unique_genes = unique_genes
        self.ureg = UnitRegistry()
        self.pixel_size = self.ureg(pixel_size)
        self.pixel_area = self.pixel_size ** 2

    def vp(self, *args):
        """Function to print output if verbose mode is True
        """
        if self.verbose:
            for arg in args:
                print('    ' + arg)

    def make_hexbin(self, spacing: float, min_count: int) -> Tuple[pd.DataFrame, np.ndarray, Any]:
        """
        Bin 2D point data with hexagonal bins.

        Args:
            spacing (float): distance between tile centers, in same units as 
                the data. The actual spacing will have a verry small deviation 
                (tipically lower than 2e-9%) in the y axis, due to the 
                matplotlib hexbin function. The function makes hexagons with 
                the point up: ⬡
            min_count (int): Minimal number of molecules in a tile to keep the 
                tile in the dataset. The algorithm will generate a lot of empty 
                tiles, which are later discarded using the min_count threshold.
                Suggested to be at least 1.

        Returns:
            Tuple[pd.DataFrame, np.ndarray, plt.Path]: 
            Pandas Dataframe with counts for each valid tile.
            Numpy Array with controid coordinates for the tiles.
            Matplotlib path object for a single polygon

        """
        #Get number of genes
        n_genes = len(self.unique_genes)
            
        #Get canvas size
        max_x = self.x.max()
        min_x = self.x.min()
        max_y = self.y.max()
        min_y = self.y.min()

        #Determine largest axes and use this to make a hexbin grid with square extent.
        #If it is not square matplotlib will stretch the hexagonal tiles to an asymetric shape.
        xlength = max_x - min_x
        ylength = max_y - min_y

        if xlength > ylength:
            #Find number of points   
            n_points = math.ceil((max_x - min_x) / spacing)
            #Correct x range to match whole number of tiles
            extent = n_points * spacing
            difference_x = extent - xlength
            min_x = min_x - (0.5 * difference_x)
            max_x = max_x + (0.5 * difference_x)
            # *
            #Adjust the y scale to match the number of tiles in x
            #For the same lengt the number of tiles in x is not equal
            #to the number of rows in y.
            #For a hexagonal grid with the tiles pointing up, the distance
            #between rows is (x_spacing * sqrt(3)) / 2
            #(sqrt(3)/2 comes form sin(60))
            xlength = max_x - min_x
            y_spacing = (spacing * np.sqrt(3)) / 2
            n_points_y = int(xlength / y_spacing)
            extent_y = n_points_y * y_spacing
            difference_y = extent_y - ylength
            min_y = min_y - (0.5 * difference_y)
            max_y = max_y + (0.5 * difference_y)

        else:
            #Find number of points  
            n_points = math.ceil((max_y - min_y) / spacing)
            #Correct y range to match whole number of tiles
            extent = n_points * spacing
            difference_y = extent - ylength
            min_y = min_y - (0.5 * difference_y)
            max_y = max_y + (0.5 * difference_y)
            #Correct x range to match y range
            #because the x dimension is driving for the matplotlib hexbin function 
            ylength = max_y - min_y
            difference_x = ylength - xlength
            min_x = min_x - (0.5 * difference_x)
            max_x = max_x + (0.5 * difference_x)
            #Adjust the y scale to match the number of tiles in x
            #See explantion above at *
            y_spacing = (spacing * np.sqrt(3)) / 2
            n_points_y = int(ylength / y_spacing)
            extent_y = n_points_y * y_spacing
            difference_y = extent_y - ylength
            min_y = min_y - (0.5 * difference_y)
            max_y = max_y + (0.5 * difference_y)

        #Perform hexagonal binning for each gene
        for i, (g, x, y) in enumerate(self.xy_groupby_gene_generator()):
            #Make hexagonal binning of the data
            hb = hexbin(x, y, gridsize=int(n_points), 
                        extent=[min_x, max_x, min_y, max_y], visible=False)
            #For the first iteration initiate the output data
            if i == 0:
                #Get the coordinates of the tiles, parameters should be the same regardles of gene.
                hex_coord = hb.get_offsets()
                #Get shape of the hexagon
                hexbin_hexagon_shape = hb.get_paths()
                #Make dataframe and add first results
                n_tiles = hex_coord.shape[0]
                df_hex = pd.DataFrame(data=np.zeros((n_genes, n_tiles)),
                                    index=self.unique_genes, 
                                    columns=[f'{self.dataset_name}_{j}' for j in range(n_tiles)])
                df_hex.loc[g] = hb.get_array()
            
            else:
                df_hex.loc[g] = hb.get_array()

        #Filter on number of molecules
        filt = df_hex.sum() >= min_count
        df_hex = df_hex.loc[:, filt]
        hex_coord = hex_coord[filt]

        return df_hex, hex_coord, hexbin_hexagon_shape

    def hexbin_normalize(self, df_hex: Any, mode:str = 'log') -> Any:
        """Simple data normalization.

        Args:
            df_hex (pd.DataFrame): DataFrame with gene counts per hexagonal 
                tile.
            mode (str, optional): Normalization method. Choose from: "log",
                "sqrt" or "z" for log +1 transform, square root transform, or
                z scores respectively. Defaults to 'log'.

        Raises:
            Exception: If mode is not properly defined

        Returns:
            [pd.Dataframe]: Normalzed dataframe.
        """

        if mode == 'log':
            result = np.log(df_hex + 1)
        elif mode == 'sqrt':
            result = np.sqrt(df_hex)
        elif mode == 'z':
            mean = df_hex.mean(axis=1)
            std = df_hex.std(axis=1)
            result = (df_hex.subtract(mean, axis=0)).divide(std, axis=0)
        else:
            raise Exception(f'Invalid "mode": {mode}')

        return result

    def hexbin_PCA(self, df_hex: Any) -> np.ndarray:
        """Calculate principle components

        Args:
            df_hex (pd.DataFrame): DataFrame with gene counts per hexagonal 
                tile.

        Returns:
             [np.array]: Array with principle components as rows

        """

        pca = PCA()
        return pca.fit_transform(df_hex.T)


    def clust_hex_connected(self, df_hex, hex_coord: np.ndarray, distance_threshold: float = None, 
                            n_clusters:int = None, neighbor_rings:int = 1) -> np.ndarray:
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
        #Make graph to connect neighbours 
        n_neighbors = (neighbor_rings * 6)
        Kgraph = kneighbors_graph(hex_coord, n_neighbors, include_self=False)

        #Cluster
        if distance_threshold!=None and n_clusters!=None:
            raise Exception('One of "distance_threshold" or "n_clusters" should be defined, not both.')

        elif distance_threshold != None:
            clust_result = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, 
                                                compute_full_tree=True, connectivity=Kgraph).fit(df_hex)

        elif n_clusters != None:
            clust_result = AgglomerativeClustering(n_clusters=n_clusters, 
                                                connectivity=Kgraph).fit(df_hex)

        else:
            raise Exception('One of "distance_threshold" or "n_clusters" should be defined.')

        #Return labels
        labels = clust_result.labels_
        return labels

    def plot_hex_labels(self, hex_coord: np.ndarray, labels: np.ndarray, s:int =4, 
                        save: bool = False, save_path: str = ''):
        """Plot clustering results of hexagonal tiles.

        Args:
            hex_coord (np.ndarray): Numpy Array with XY coordinates of the
                centroids of the hexagonal tiles.
            labels (np.ndarray): Numpy Array with cluster labels.
            s (int, optional): Size of the hexagon in the plot. Defaults to 4.
            save (bool, optional): If True saves the plot. Defaults to False.
            save_path (str, optional): Path to where the plot should be saved. 
                Defaults to ''.

        """
        n_clust = len(np.unique(labels))
        plt.figure(figsize=(10,10))
        colors = np.multiply(plt.cm.gist_ncar(labels/max(labels)), [0.9, 0.9, 0.9, 1])
        xdata = hex_coord[:,0]
        ydata = hex_coord[:,1]
        #There must be some smart way to determine the size of the marker so that it matches the spacing between the 
        #tiles and the hexagons line up nicely. But the definition of the size is too convoluted to make a quick fix.
        plt.scatter(xdata, ydata, s=s, marker=(6, 0, 0), c=colors)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.title(f'{self.dataset_name} {n_clust} clusters')
        if save == True:
            plt.savefig(f'{save_path}{self.dataset_name}_hexagon_clustering.pdf')

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

    def make_cluster_mean(self, df_hex: Any, labels: np.ndarray) -> Any:
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

    def make_cluster_sum(self, df_hex, labels: np.ndarray) -> Any: 
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

    def get_rotation(self, x1: float, y1: float, x2: float, y2: float) -> float:
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
            dict: Dictunary with for each label the boundary point of the 
                hexagonal tile as Numpy Array. 
        
        """
        #Dictionary coupling angle with neighbour to angle of shared corners
        shared_corner = {0: [-30, 30],
                        60: [30, 90],
                        120: [90, 150],
                        180: [150, -150],
                        -120: [-150, -90],
                        -60: [-90, -30]}

        def get_shared_corners(angle, x=0, y=0):
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
            shared_angles = shared_corner[angle]
            corner1 = corner[shared_angles[0]] + [x, y]
            corner2 = corner[shared_angles[1]] + [x, y]
            return corner1, corner2

        #make result dictionary
        boundary_points = {l:[] for l in np.unique(labels)}

        #Determine corner locations relative to tile centroid
        #Because of small devitions of a perfect hexagonal grid it is best to get the hexagon border point form the 
        #Matplotlib Polygon.
        corner = {}
        for c in hexagon_shape[0].vertices[:6]: #There are points in the polygon for the closing vertice.
            angle = round(self.get_rotation(0,0, c[0], c[1]))
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
                angles = np.array([round(self.get_rotation(own_coord[0], own_coord[1], c[0], c[1])) for c in neighbour_coords])
                #Iterate over all possible neighbour angles
                to_add = []
                for a in [0, 60, 120, 180, -120, -60]:
                    #Check if a neighbour exists
                    if a in angles:
                        angle_index = np.where(angles == a)
                        #Check of this neighbour is of a different identity
                        if neighbour_identity[angle_index] != l:
                            to_add.append(a)
                    #No there is no neigbour at that angle, this is a border tile   
                    else:
                        to_add.append(a)
                #Get the shared corner point for neighbours that are missing or have a different identity      
                for a2 in to_add:
                    c1, c2 = get_shared_corners(a2, x=own_coord[0], y=own_coord[1])
                    boundary_points[l].append(c1)
                    boundary_points[l].append(c2)
                    #Add intermediate point to help making circles
                    boundary_points[l].append((np.mean([c1, c2], axis=0)))
                    #boundary_points[d][l].append([])

        #Clean duplicated points
        for l in np.unique(labels):
            boundary_points[l] = np.unique(np.array(boundary_points[l]).round(decimals=decimals), axis=0)

        return boundary_points

    def order_points(self, boundary_points: dict) -> dict:
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

    def smooth_points(self, ordered_points: dict, degree: int=2) -> dict:
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

    def make_geoSeries(self, polygons: dict) -> Any:
        """Convert dictionary with shapely polygons to geoPandas GeoSeries.

        Args:
            polygons ([type]): Dictionary with for every label a Shapely 
                Polygon, or a MultiPolygon if the region consists of multiple 
                polygons.

        Returns:
            GeoSeries: Geopandas series of the polygons.

        """
        return gp.GeoSeries(polygons, index=list(polygons.keys()))

    def make_geoDataFrame(self, data: np.ndarray, index: Union[List[str], np.ndarray], 
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

    def get_bounding_box(self, polygon: np.ndarray) -> np.ndarray:
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

    def make_bounding_box(self, polygon_points: dict) -> dict:
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
                results[l].append(self.get_bounding_box(poly))
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
        gs = self.make_geoSeries(polygons)
        labels = list(polygon_points.keys())
        gdf = self.make_geoDataFrame(data=np.zeros((len(labels), len(self.unique_genes)), dtype='int64'), index=labels, 
                                            columns=self.unique_genes,  geometry=gs)

        #Recount which points fall in which (multi-)polygon
        df = pd.DataFrame(data = self.gene, columns = ['gene'])
        for l, filt in inside_multi_polygons(polygon_points, np.column_stack((self.x, self.y))):
            #get the sum of every gene and asign to gene count for that area. 
            gdf.loc[l, self.unique_genes] = df[filt].groupby('gene').size() 

        #Normalize data    
        if normalize:
            area_in_pixels = gdf.area
            conversion = self.area_scale.to(self.ureg(normalize_unit) ** 2)
            area_in_desired_unit = area_in_pixels * conversion.magnitude
            gdf.iloc[:,:-1] = gdf.iloc[:,:-1].divide(area_in_desired_unit, axis=0)            

        return gdf

    def run_regionalization(self, spacing: float, min_count: int,
                            normalization_mode: str = 'log', 
                            n_components: int = 100, clust_dist_threshold: float = 70, clust_neighbor_rings: int = 1,
                            smooth_neighbor_rings: int = 1, smooth_cycles: int = 1,
                            boundary_decimals: int = 7,
                            smooth_polygon: bool = False, smooth_polygon_degree:int = 7, 
                            recount: bool = False,
                            area_normalize: bool = True, area_normalize_unit: str = 'micrometer') -> None:
        """Run the regionalization pipeline.

        Chains all functions to go from raw point data to regions. Use the
        individual functions for more control.

        Args:
            spacing (float): distance between tile centers, in same units as 
                the data. The actual spacing will have a verry small deviation 
                (tipically lower than 2e-9%) in the y axis, due to the 
                matplotlib hexbin function. The function makes hexagons with 
                the point up: ⬡
            min_count (int):  Minimal number of molecules in a tile to keep the 
                tile in the dataset. The algorithm will generate a lot of empty 
                tiles, which are later discarded using the min_count threshold.
                Suggested to be at least 1.
            normalization_mode (str, optional): normalization method for 
                clustering. Choose from: "log", "sqrt" or "z" for log +1 
                transform, square root transform, or z scores respectively. 
                Defaults to 'log'.
            n_components (int, optional): Number of PCA components to use for
                clustering. Defaults to 100.
            clust_dist_threshold (float, optional): Distance threshold for 
                Scipy Agglomerative clustering. Defaults to 70.
            clust_neighbor_rings (int, optional): Number of rings around a 
                central tile to make connections between tiles for 
                Agglomerative Clustering with connectivity. 1 means connections
                with the 6 imediate neighbors. 2 means the first and second 
                ring, making 18 neigbors, etc. Defaults to 1.
            smooth_neighbor_rings (int, optional):  Number of rings around a 
                central tile to smooth over. 1 means connections with the 6 
                imediate neighbors. 2 means the first and second ring, making 
                18 neigbors, etc. Defaults to 1.
            smooth_cycles (int, optional): Number of smoothing cycles.
                Defaults to 1.
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
        #Bin the data with a hexagonal grid
        df_hex, hex_coord, hexagon_shape = self.make_hexbin(spacing, min_count)
        #Normalize data
        df_hex_norm = self.hexbin_normalize(df_hex, mode=normalization_mode)
        #Calculate PCA
        pc = self.hexbin_PCA(df_hex_norm)
        #Cluster dataset
        labels = self.clust_hex_connected(pc[:,:n_components], hex_coord, distance_threshold=clust_dist_threshold, neighbor_rings=clust_neighbor_rings)
        #Spatially smooth cluster labels
        labels = self.smooth_hex_labels(hex_coord, labels, smooth_neighbor_rings, smooth_cycles, n_jobs=1)
        #Get boundary points of regions
        boundary_points = self.hex_region_boundaries(hex_coord, hexagon_shape, spacing, labels, decimals = boundary_decimals)
        #Order boundary point for making polygons
        ordered_points = self.order_points(boundary_points)
        #Smooth boundary points
        if smooth_polygon == True:
            ordered_points = self.smooth_points(ordered_points, degree = smooth_polygon_degree)
            if recount == False:
                recount = True
                self.vp('Recount set to True after "Smooth_polygon" was set to True.')
        #Recount points in polygons and make geoDataFrame
        if recount == True:
            self.regions = self.point_in_region(ordered_points, normalize=area_normalize,normalize_unit=area_normalize_unit)
            self.vp(f'{self.dataset_name} regionalized, result can be found in "self.regions"')
        #make geoDataFrame with region data
        else:
            sum_counts = self.make_cluster_sum(df_hex, labels).T #Transpose because geopandas wants regions in index
            shapely_polygons = self.to_Shapely_polygons(ordered_points)
            new_geoseries = self.make_geoSeries(shapely_polygons)

            if area_normalize:
                area_in_pixels = new_geoseries.area
                conversion = self.area_scale.to(self.ureg(area_normalize_unit) ** 2)
                area_in_desired_unit = area_in_pixels * conversion.magnitude
                sum_counts = sum_counts.divide(area_in_desired_unit, axis=0)

            gdf = self.make_geoDataFrame(data = sum_counts.to_numpy(), index = sum_counts.index, 
                                                 columns = sum_counts.columns, geometry=new_geoseries)
            self.regions = gdf
            self.vp(f'{self.dataset_name} regionalized, result can be found in "self.regions"')

            

