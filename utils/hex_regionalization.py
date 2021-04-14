import gc
import glob
import math
from itertools import combinations, permutations
from multiprocessing import cpu_count
import geopandas as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed
from matplotlib.pyplot import axes, hexbin, xcorr
from shapely.geometry import MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage.measure import subdivide_polygon
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
#Mypy types
from typing import Tuple

class regionalize:
    """Class for regionalization of multidimentional point data.

    """

    def __init__(self, 
        filename: str,
        x: str = 'r_px_microscope_stitched',
        y: str = 'c_px_microscope_stitched',
        gene_column: str = 'below3Hdistance_genes',
        other_columns: list = None,
        unique_genes: np.ndarray = None):
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
                Defaults to None.
            unique_genes (np.ndarray, optional): Numpy array with unique gene
                names. If not defined, the unique genes will be found but this
                can take time for milions of points.

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

    def make_hexbin(self, spacing: float, min_count: int) -> Tuple[pd.DataFrame, np.ndarray]:
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
            Tuple[pd.DataFrame, np.ndarray]: 
            Pandas Dataframe with counts for each valid tile.
            Numpy Array with controid coordinates for the tiles.

        Spacing and the Matplotlib hexagon Polygon will be saved as:
        self.hexbin_spacing
        self.hexbin_hexagon_shape

        """
        self.hexbin_spacing = spacing
        #Get number of genes
        n_genes = len(self.unique_genes)
            
        #Get canvas size
        max_x = self.data.loc[:, self.x].max()
        min_x = self.data.loc[:, self.x].max()
        max_y = self.data.loc[:, self.y].max()
        min_y = self.data.loc[:, self.y].max()

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
        for i,  (g, c) in enumerate(self.data.loc[:, [self.x, self.y, self.gene_column]].groupby(self.gene_column)):
            #Make hexagonal binning of the data
            hb = hexbin(c.loc[:, self.x], c.loc[:, self.y], gridsize=int(n_points), 
                        extent=[min_x, max_x, min_y, max_y], visible=False)
            #For the first iteration initiate the output data
            if i == 0:
                #Get the coordinates of the tiles, parameters should be the same regardles of gene.
                hex_coord = hb.get_offsets()
                #Get shape of the hexagon
                self.hexbin_hexagon_shape = hb.get_paths()
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
        df_hex = df_hex[filt]
        hex_coord = hex_coord[filt]

        return df_hex, hex_coord

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
                        return_all: bool = False, n_jobs: int = -1) -> np.ndarray:
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
                tile to make connections between tiles for Agglomerative 
                Clustering with connectivity. 1 means connections with the 6 
                imediate neighbors. 2 means the first and second ring, making 
                18 neigbors, etc.. Defaults to 1.
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
        print('CHECK IF N_JOBS IS REALLY NEEDED FOR THE KNEIGHBORS_GRAPH, MAYBE IT IS FAST ENOUGH TO RUN ON 1 CORE. ')
        print('IT WOULD SIMPLIFY THE PARALEL STUFF LATER')
        Kgraph = kneighbors_graph(hex_coord, n_neighbors, include_self=True, n_jobs=n_jobs)

        results = [labels]
        for iteration in range(cycles):
            results.append(smooth(Kgraph, results[-1]))
        
        if return_all:
            return results
        else:
            return results[-1]

    def make_cluster_mean(self, df_hex, labels: np.ndarray):
        """Calculate cluster mean.

        For a DataFrame with samples in columns, calculate the mean values for 
        each unique label in labels.

        Args:
            df_hex (pd.DataFrame): Pandas DataFrame with samples in columns.
            labels (np.ndarray): Numpy array with cluster labels

        Returns:
            [pd.DataFrame]: Pandas Dataframe with mean values for each label.

        """
        unique_labels = np.unique(labels)
        cluster_mean = pd.DataFrame(data=np.zeros((df_hex.shape[0], len(unique_labels))), index = df_hex.index,
                                    columns=[f'{self.dataset_name}_{l}' for l in unique_labels])

        #Loop over clusters
        for l in unique_labels:
            filt = labels == l
            #Get mean expression of cluster
            cluster_mean.loc[:, l] = df_hex.loc[:, filt].mean(axis=1)

        return cluster_mean

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

    def hex_region_boundaries(self, hex_coord: np.ndarray, labels: np.ndarray, decimals: int = 7) -> dict:
        """ Find border coordinates of regions in a hexagonal grid.

        Finds the border coordinates for each connected group of hexagons 
        represented by a unique label. Assumes hexagonal grid with point up: ⬡

        Args:
            hex_coord (np.ndarray): Numpy Array with centroid coordinates as XY
                columns for all tiles in a hexagonal grid. 
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
        for c in self.hexbin_hexagon_shape[0].vertices[:6]: #There are points in the polygon for the closing vertice.
            angle = round(self.get_rotation(0,0, c[0], c[1]))
            corner[angle] = c

        #Find neighbours
        neighbour_radius = self.hexbin_spacing + (0.1 * self.hexbin_spacing)
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

    def order_points(boundary_points: dict) -> dict:
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
        
    def to_polygons(self, ordered_points: dict) -> dict:
        """Make Shapely polygons out of a set of ordered points.

        Converts orderd points to polygons, and makes complex polygons 
        (MulitPolygons) if a region consists of multiple polygons. These sub-
        polygons can be inside the main polygon, in which case a hole is cut.

        Args:
            ordered_points (dict): Dictionary with keys for each label and a 
                list of orderd points for each polygon

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

    def make_geoSeries(self, polygons: dict):
        """Convert dictionary with shapely polygons to geoPandas GeoSeries.

        Args:
            polygons ([type]): Dictionary with for every label a Shapely 
                Polygon, or a MultiPolygon if the region consists of multiple 
                polygons.

        Returns:
            GeoSeries: Geopandas series of the polygons.

        """
        return gp.GeoSeries(polygons, index=list(polygons.keys()))

    # GOT TO HERE with reformating. Code not tested yet. 

    @jit(nopython=True)
    def is_inside_sm(polygon, point):
        """[summary]

        From: https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
        From: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

        Args:
            polygon ([type]): [description]
            point ([type]): [description]

        Returns:
            [type]: [description]
        """
        length = len(polygon)-1
        dy2 = point[1] - polygon[0][1]
        intersections = 0
        ii = 0
        jj = 1

        while ii<length:
            dy  = dy2
            dy2 = point[1] - polygon[jj][1]

            # consider only lines which are not completely above/bellow/right from the point
            if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

                # non-horizontal line
                if dy<0 or dy2<0:
                    F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                    if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                        intersections += 1
                    elif point[0] == F: # point on line
                        return 2

                # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
                elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                    return 2

            ii = jj
            jj += 1

        #print 'intersections =', intersections
        return intersections & 1  

    @njit(parallel=True)
    def is_inside_sm_parallel(points, polygon):
        """[summary]

        From: https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
        From: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

        Args:
            points ([type]): [description]
            polygon ([type]): [description]

        Returns:
            [type]: [description]
        """
        ln = len(points)
        D = np.empty(ln, dtype=numba.boolean) 
        for i in numba.prange(ln):
            D[i] = is_inside_sm(polygon,points[i])
        return D  

    def get_bounding_box(polygon):
        """
        Get the bounding box of a single polygon.
        
        Input:
        `polygon`(np.array): Array of X, Y coordinates of the polython in the columns.
        
        Returns:
        Numpy array with the left bottom and top right corner coordinates. 
        
        """
        
        xmin, ymin = polygon.min(axis=0)
        xmax, ymax = polygon.max(axis=0)
        return np.array([[xmin, ymin], [xmax, ymax]])

    def make_bounding_box(polygon_points):
        """
        Find bounding box coordinates of dictionary with polygon points.
        
        Input:
        `polygon_points`(dict): Dictionary with keys for datasets, and for every dataset a dictionary
            with labels of each polygon and a list of polygon(s) as numpy arrays. 
            
        Returns:
        Bounding box coordinates (left_bottom, top_right) for each polygon. Output is a dictionary 
        in the same shape as the input. 
        
        
        """
        datasets = list(polygon_points.keys())
        results = {d:{} for d in datasets}
        for d in datasets:
            labels = list(polygon_points[d].keys())
            #results[d] = {l:[] for l in labels}
            for l in labels:
                results[d][l] = []
                for poly in polygon_points[d][l]:
                    xmin, ymin = poly.min(axis=0)
                    xmax, ymax = poly.max(axis=0)
                    results[d][l].append(get_bounding_box(poly))
        return results

    def bbox_filter_points(bbox, points):
        """
        Filter point that fall within bounding box.
        
        Input:
        `bbox`(np.array): Array with bottom left and top right corner of the bounding
            box: np.array([[xmin, ymin], [xmax, ymax]])
        `points`(np.array): Array with X and Y coordinates of all points 
        
        Retruns:
        Boolean array filter with True for point that fall within the bounding box
        
        
        """
        #return np.all(np.logical_and(points >= bbox[0], points <= bbox[1]), axis=1)
        return np.logical_and(np.logical_and(points[:,0]>=bbox[0][0], points[:,0]<=bbox[1][0]), np.logical_and(points[:,1]>=bbox[0][1], points[:,1]<=bbox[1][1]))

    def bbox_filter_points_multi(bbox, spots):
        """
        Filter point that fall within bounding box of a polygon.

        Works but takes up a lot of memory, implemented on the fly in other 
        functions to prevent memory overload. 
        
        """
        #Check input
        datasets_bbox = list(bbox.keys())
        datasets_spots = list(spots.keys())
        if datasets_bbox != datasets_spots:
            raise Exception(f'Input keys of "bbox" and "spots" does not match: bbox:{datasets_bbox}, spots: {datasets_spots}')
        
        results = {d:{} for d in datasets_bbox}
        for d in datasets_bbox:
            labels = list(bbox[d].keys())
            spots_of_interest = spots[d][['x', 'y']].to_numpy()
            
            for l in labels:
                results[d][l] = []
                for corners in bbox[d][l]:
                    results[d][l].append(corners, spots_of_interest)
                    
        return results

    def point_in_region(spots, polygon_points, genes):
        """
        
        ## TODO: normalize by area. 


        """
        #Check input
        datasets_spots = list(spots.keys())
        datasets_polygons = list(polygon_points.keys())
        if datasets_polygons != datasets_spots:
            raise Exception(f'Input keys of "spots" and "polygons" does not match: spots:{datasets_spots}, polygons: {datasets_polygons}')    
        datasets = datasets_spots
        
        #Make geoseries out of the polygons
        polygons = to_polygons(polygon_points)
        gs = make_geoSeries(polygons)
        #Make bounding boxes for all polygons
        bbox = make_bounding_box(polygon_points)    
        
        results = {}
        sip= {}
        #Iterate over datasets
        for d in datasets:
            
            labels = list(polygon_points[d].keys())
            gdf = gp.GeoDataFrame(data=np.zeros((len(labels), len(genes)), dtype='int64'), columns=genes,  geometry=gs[d])
            spots_of_interest = spots[d][['x', 'y']].to_numpy()
            
            #Iterate over all (multi) polygons
            for l in labels:
                print(f'Binning points in regions. Processing dataset: {d}                  ', end='\r')
                
                point_inside = np.zeros(spots[d].shape[0]).astype('bool')
                
                #Iterate over every (sub) polygon of the (multi) polygon
                for p, bb in zip(polygon_points[d][l], bbox[d][l]):
                    #Filter points with the bounding box of the polygon
                    filt = bbox_filter_points(bb, spots_of_interest)
                    #Check which points are inside the polygon
                    is_inside = is_inside_sm_parallel(spots_of_interest[filt], p)
                    #For a point to be inside a multi polygon, it needs to be found inside the sub-polygons an uneven number of times.
                    point_inside[filt] = np.logical_xor(point_inside[filt], is_inside)
                
                #get the sum of every gene and asign to gen count for that area. 
                gdf.loc[l, genes] = spots[d][point_inside].groupby('gene').size() 
                
            results[d] = gdf
            
        return results

#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
class regionalize_multiple:

    def merge_norm(hex_bin, normalizer=None):
        """
        Merge multiple datasets and optonally normalize the data.
        Input:
        `hex_bin`(dict): Dictonray with hex bin results for every dataset.
            Output form the hex_bin.make_hexbin() function.
        `normalizer`(func): Function that can normalize the data.
            For example: "lambda x: np.log(x + 1)" for log normalization.
            Carfull, this normalizaiton will be applied on each dataset
            individually first, and then merged in one large dataset. 
        Returns:
        `df`(pd.Dataframe): Dataframe with all tiles in one table. Optionally
            with normalization applied
        `samples`(np.array): Array with original dataset names.
        
        """
        nrows = math.ceil(len(hex_bin.keys())/2)
        fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=(10,1.5*nrows), sharey=True, sharex=True)
        
        norm = normalizer != None
        
        for i, d in enumerate(list(hex_bin.keys())):
            print(f'Merging datasets. Processing sample: {i}                    ', end='\r')
            if i == 0:
                df_next = hex_bin[d]['df']
                df_all = df_next
                samples = [d for i in df_all.columns]
                if norm:
                    df_next_norm = normalizer(df_next)
                    df_norm = df_next_norm
                
            else:
                df_next = hex_bin[d]['df']
                df_all = pd.concat([df_all, df_next], axis=1, sort=False)
                for j in df_next.columns:
                    samples.append(d)
                if norm:
                    df_next_norm = normalizer(df_next)
                    df_norm = pd.concat([df_norm, df_next_norm], axis=1, sort=False)
                
            
            ax = axes[int(i/2), i%2]
            if norm:
                ax.hist(df_next_norm.sum(), bins=100)
                ax.set_title(f'{d} normalized')
            else:
                ax.hist(df_next.sum(), bins=100)
                ax.set_title(d)
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Sum molecule count')
        plt.tight_layout()
        
        if norm:
            return df_norm, np.array(samples)
        else:
            return df_all, np.array(samples)

    def make_hexbin_serial(spacing, spots, min_count):
        """
        Serial wrapper around make_hexbin()
        
        Use when running out of memory with the parallel function.
        
        Input:
        `spacing`(int): distance between tile centers, in same units as the data. 
            The actual spacing will have a verry small deviation (tipically lower 
            than 2e-9%) in the y axis, due to the matplotlib hexbin function.
            The function makes hexagons with the point up: ⬡
        `spots`(dictionary): Dictionary with for each dataset aDataFrame with 
            columns ['x', 'y', 'gene'] for the x and y coordinates and the gene 
            labels respectively.
        `min_count`(int): Minimal number of molecules in a tile to keep the tile in 
            the dataset. The algorithm will generate a lot of empty 
            tiles, which are later discarded using the min_count threshold.
            Suggested to be at least 1.
        
        Output:
        Dictionary with the following items:
        `gene` --> Dictionary with tile counts for each gene.
        `hexbin` --> matplotlib PolyCollection for hex bins.
        `coordinates` --> XY coordinates for all tiles.
        `coordinates_filt` --> XY coordinates for all tiles that have
            enough counts according to "min_count"
        `df` --> Pandas dataframe with counts for all genes in all 
            valid tiles.
        `spacing` --> Chosen spacing. Keep in mind that the distance between tile
            centers in different rows might deviate slightly. 
        
        """
        dataset_keys = list(spots.keys())
        results={}    
        
        
        spacing_list = [spacing for i in dataset_keys]
        datasets = [spots[k] for k in dataset_keys]
        min_count_list = [min_count for i in dataset_keys]
        
        for d in dataset_keys:
            results[d] = make_hexbin(spacing, spots[d], min_count)

        return results


    def make_hexbin_parallel(spacing, spots, min_count, n_jobs=None):
        """
        Parallel wrapper around make_hexbin()
        
        Can consume 
        
        Input:
        `spacing`(int): distance between tile centers, in same units as the data. 
            The actual spacing will have a verry small deviation (tipically lower 
            than 2e-9%) in the y axis, due to the matplotlib hexbin function.
            The function makes hexagons with the point up: ⬡
        `spots`(dictionary): Dictionary with for each dataset aDataFrame with 
            columns ['x', 'y', 'gene'] for the x and y coordinates and the gene 
            labels respectively.
        `min_count`(int): Minimal number of molecules in a tile to keep the tile in 
            the dataset. The algorithm will generate a lot of empty 
            tiles, which are later discarded using the min_count threshold.
            Suggested to be at least 1.
        
        Output:
        Dictionary with the following items:
        `gene` --> Dictionary with tile counts for each gene.
        `hexbin` --> matplotlib PolyCollection for hex bins.
        `coordinates` --> XY coordinates for all tiles.
        `coordinates_filt` --> XY coordinates for all tiles that have
            enough counts according to "min_count"
        `df` --> Pandas dataframe with counts for all genes in all 
            valid tiles.
        `spacing` --> Chosen spacing. Keep in mind that the distance between tile
            centers in different rows might deviate slightly. 
        
        """
        dataset_keys = list(spots.keys())
        spacing_list = [spacing for i in dataset_keys]
        datasets = [spots[k] for k in dataset_keys]
        min_count_list = [min_count for i in dataset_keys]

        #Paralel execution for all datasets
        with Pool(processes=n_jobs) as pool:
            result = pool.starmap(make_hexbin, zip(spacing_list, datasets, min_count_list), 1)
            
        pooled_results = {k:v for k,v in zip(dataset_keys, result)}

        return pooled_results

    def make_hexbin_joblib(spacing, spots, min_count, unique_genes, n_jobs=None):
        """
        Parallel wrapper around make_hexbin()
        
        Can consume quite some memory. If this is a problem use the serial 
        function.
        
        Input:
        `spacing`(int): distance between tile centers, in same units as the data. 
            The actual spacing will have a verry small deviation (tipically lower 
            than 2e-9%) in the y axis, due to the matplotlib hexbin function.
            The function makes hexagons with the point up: ⬡
        `spots`(dictionary): Dictionary with for each dataset aDataFrame with 
            columns ['x', 'y', 'gene'] for the x and y coordinates and the gene 
            labels respectively.
        `min_count`(int): Minimal number of molecules in a tile to keep the tile in 
            the dataset. The algorithm will generate a lot of empty 
            tiles, which are later discarded using the min_count threshold.
            Suggested to be at least 1.
        `n_jobs`(int): Number of jobs to use. If set to None, will use the number 
            of CPUs given by multiprocessing.cpu_count()
        
        Output:
        Dictionary with the following items:
        `gene` --> Dictionary with tile counts for each gene.
        `hexbin` --> matplotlib PolyCollection for hex bins.
        `coordinates` --> XY coordinates for all tiles.
        `coordinates_filt` --> XY coordinates for all tiles that have
            enough counts according to "min_count"
        `df` --> Pandas dataframe with counts for all genes in all 
            valid tiles.
        `spacing` --> Chosen spacing. Keep in mind that the distance between tile
            centers in different rows might deviate slightly. 
        
        """
        dataset_keys = list(spots.keys())
        
        if n_jobs == None:
            n_jobs = cpu_count()
        with Parallel(n_jobs=n_cores, backend='loky') as parallel:
            result = parallel(delayed(make_hexbin)(spacing, spots[k], min_count, unique_genes) for k in dataset_keys)
            
        pooled_results = {k:v for k,v in zip(dataset_keys, result)}
        del result
        gc.collect()

        return pooled_results

    #Just copied here as backup. This could be paralelized. 
    def clust_hex_connected(hex_bin, data, samples, distance_threshold=None, 
                        n_clusters=None, neighbor_rings=1, x_lim=None, 
                        y_lim=None, save=False, save_name=''):
        """
        Cluster hex-bin data, with a neighborhood embedding.
        
        Clusters with AggolmerativeClustering that uses a distance matrix 
        made from the tiles and their neighbours within the neighbour_radius.
        Can either cluster with a pre-defined number of resulting clusters
        by passing a number to "n_clusters", or clusters with a 
        "distance_threshold" that determines the cutoff. When passing
        multiple datasets that require a different number of clusters the
        "distance_threshold" will be more suitable.
        Input:
        `hex_bin`(dict): Dictonray with hex bin results for every dataset.
            Output form the hex_bin.make_hexbin() function.
        `data`(pd.DataFrame): Dataframe with data to cluster.
        `distance_threshold`(float): Distance threshold for Agglomerative
            Clustering.
        `n_clusters`(int): Number of desired resulting clusters.
        `neighbor_rings`(int): Number of rings around a central tile to make
            connections between tiles for AgglomerativeClustering with connectivity.
            1 means connections with the 6 imediate neighbors. 2 means the first and
            second ring, making 18 neigbors, etc.
        `x_lim`(tuple): Tuple with (x_min, x_max)
        `y_lim`(tuple): Tuple with (y_min, y_max)

        """
        n_rows = math.ceil(len(hex_bin.keys())/3)
        fig = plt.figure(constrained_layout=True, figsize=(20, n_rows*10))
        gs = fig.add_gridspec(n_rows, 6)
        
        labels_all = []
        datasets = list(hex_bin.keys())
        for i, d in enumerate(datasets):
            print(f'Connected clustering. Processing sample: {i}                    ', end='\r')
            ax = fig.add_subplot(gs[int(i/3), (i%3)*2:((i%3)*2)+2])
            
            n_neighbors = (neighbor_rings * 6)
            Kgraph = kneighbors_graph(hex_bin[d]['coordinates_filt'], n_neighbors, include_self=False)
            
            if distance_threshold!=None and n_clusters!=None:
                raise Exception('One of "distance_threshold" or "n_clusters" should be defined, not both.')
            elif distance_threshold != None:
                clust_result = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, 
                                                    compute_full_tree=True, connectivity=Kgraph).fit(data[samples == d])
            elif n_clusters != None:
                clust_result = AgglomerativeClustering(n_clusters=n_clusters, 
                                                    connectivity=Kgraph).fit(data[samples == d])
            else:
                raise Exception('One of "distance_threshold" or "n_clusters" should be defined.')

            labels = clust_result.labels_
            n_clust = len(np.unique(labels))
            labels_all.append(labels)
            
            colors = np.multiply(plt.cm.gist_ncar(labels/max(labels)), [0.9, 0.9, 0.9, 1])
            
            xdata = hex_bin[d]['coordinates_filt'][:,0]
            ydata = hex_bin[d]['coordinates_filt'][:,1]
            ax.scatter(xdata, ydata, s=4, marker=(6, 0, 0), c=colors)
            ax.set_aspect('equal')

            #set canvas size, assumes that the first section is the largest if not given.
            if x_lim == None or y_lim == None:
                if i == 0:
                    ylim = ax.get_ylim()
                    xlim = ax.get_xlim()
            else:
                ax.set_ylim(ylim)
                ax.set_xlim(xlim)

            #ax.invert_yaxis()
            ax.axis('off')
            ax.set_title(f'{d} {n_clust} clusters')
        plt.suptitle(f'Clustering with neighbour connectivity')
        
        if save == True:
            plt.savefig(f'{save_name}_clustering_connectivity_radius{neighbour_radius}.png', dpi=300)
        
        return labels_all

     def make_cluster_mean(hex_bin_signal, samples, labels, names):
        """

        THIS IS THE ORIGINAL FUNCTION RE-IMPLEMENTED ABOVE. MAKE A NEW IMPLEMENTATION THAT WORKS 

        Calculate mean for every cluster.
        
        Input:
        `hex_bin_signal`(np.array): Array with signal for every hexagonal tile.
            Tiles as columns and genes as rows. Data can be raw counts or 
            normalized counts.
        `samples`(np.array): Array with original dataset names to identify
            from with dataset the tiles are comming.
        `labels`(np.array): Array with the clustering labels for each dataset 
            as subarray. Example: "np.array([[1,1,2] [1,2,2]])" For 2 datasets 
            with each 3 samples.
        `names`(list): Names of the individual samples in the hex_bin dataset.
        
        Returns
        Dataframe with cluster names and mean.
        
        
        """
        cluster_names = []
        for d, label in zip(names, labels):
            for l in np.unique(label):
                cluster_names.append(f'{d}_{l}')
        
        n_index = hex_bin_signal.shape[1]
        cluster_mean = pd.DataFrame(data=np.zeros((n_index, len(cluster_names))), index=np.arange(0,n_index,1), columns=cluster_names)

        for d, label in zip(names, labels):
            for l in np.unique(label):
                name = f'{d}_{l}'
                filt1 = samples == d
                data = hex_bin_signal[filt1,:]  
                filt2 = label == l
                data_mean = np.mean(data[filt2,:], axis=0)
                cluster_mean.loc[:,name] = data_mean
                
        return cluster_mean

    
    def make_cluster_correlation(cluster_mean, method='pearson'):
        """
        Return a correlation matrix between cluster_mean expression.
        This is basically a wrapper for the Pandas .corr() function.
        Input:
        `cluster_mean`(pd.Dataframe): Dataframe with clusters as columns, and mean
            expression for each gene as rows. 
        `method`(str): Method for correlation: "pearson", "kendall", "spearman"
        Returns
        Pandas dataframe with correlation matrix.
        
        """
        return cluster_mean.corr(method=method)

    def make_similarity_network(correlation_matrix=None, cutoff=None, links=None):
        """
        Make links between correlating clusters and return a network.
        
        Makes links of the highest correlating clusters above the cutoff.
        
        Input:
        'correlation_matrix'(pd.Dataframe): Dataframe with a correlation
            matrix. Can be the result of the "make_cluster_mean()"
            function after calling .corr().
        `cutoff`(float): Cutoff correlation below which relations are 
            ignored.
        `links`(pd.Dataframe): Dataframe with edges between nodes. Contains at least
            two columns with "source" and "target" listing the IDs of the nodes to
            be connected. See networkx.from_pandas_edgelist() documentation for more
            details.
        Returns:
        Network with edges between nodes that are correlated with each other above
        the cutoff.
        
        """
        print(type(correlation_matrix))
        print(type(links))
        if type(correlation_matrix) == pd.core.frame.DataFrame and type(links) != pd.core.frame.DataFrame:
            links = correlation_matrix.stack().reset_index()
            links.columns = ['Var1', 'Var2', 'value']

            #Filter the links 
                #Filter on correlation value
            links = links.loc[(links['value']>cutoff) & (links['value']<1)]
                #filter on within sample correlation
            filt = [False if i[:8] == j[:8] else True for i,j in zip(links['Var1'], links['Var2'])]
            links = links.loc[filt]
            
        elif type(correlation_matrix) != pd.core.frame.DataFrame and type(links) == pd.core.frame.DataFrame:
            pass
        
        else:
            raise Exception('Input is not correct, specify correlation_matrix or links but not both or none, and they should be pandas dataframes')

        G = nx.from_pandas_edgelist(links, source='Var1', target='Var2', edge_attr='value')

        fig = plt.figure(figsize=(10,10))
        nx.draw(G, with_labels=True, node_color='orange', node_size=20, edge_color='gray', 
                linewidths=1, font_size=6,ax=plt.gca())
        
        return G

    def merge_labels(G, labels, names):
        """
        Merge labels based on a network that links labels toghether.
        
        Input:
        `G`(nx.network): Network with edges between cluster labels that need to be 
            merged.
        `labels`(np.array): Array with the clustering labels for each dataset 
            as subarray. Example: "np.array([[1,1,2] [1,2,2]])" For 2 datasets 
            with each 3 samples.
        `names`(list): Names of the individual samples in the hex_bin dataset.
        Returns:
        Array in the same shape as "labels" with the new merged labels.
        
        """
        merge_dict = {}
        for i, group in enumerate(nx.connected_components(G)):
            for g in group:
                merge_dict[g] = f'merge_{i}'

        str_labels = []
        for d, label in zip(names, labels):
            for l in label:
                str_labels.append(f'{d}_{l}')

        str_labels_merged = []
        for i in str_labels:
            if i in merge_dict.keys():
                i = merge_dict[i]
            str_labels_merged.append(i)

        merge_label_dict = {v: i for i, v in enumerate(np.unique(str_labels_merged))}
        int_labels_merged = np.array([merge_label_dict[i] for i in str_labels_merged])
        
        return np.array(int_labels_merged)
    
    def hex_region_boundaries_paralel(hex_bin, labels, decimals=7, n_jobs=None):
        """
        Paralel wrapper around hex_region_boundaries
        
        Input:
        `hex_bin`(dict): Dictonray with hex bin results for every dataset.
            Output form the hex_bin.make_hexbin() function.
        `labels`(np.array): Array of arrays with labels for each dataset in hex_bin. 
        `decimal`(int): Due to inaccuracies in the original hexagon tile generation 
            and rounding errors, the points need to be rounded to return all unique 
            points. If you experience errors with the generation of polygons 
            downstream, lower the number of decimals.
        `n_jobs`(int): Number of processes for multiprocessing.pool.
            If None it will use the max number of CPU cores.
        
        """
        dataset_keys = list(hex_bin.keys())
        #datasets = [hex_bin[k] for k in dataset_keys]
        #decimals_list = [decimals for i in dataset_keys]

        #Paralel execution for all datasets
        #with Pool(processes=n_jobs) as pool:
        #    result = pool.starmap(hex_region_boundaries, zip(datasets, labels, decimals_list), 1)
            
        n_cores = cpu_count()
        with Parallel(n_jobs=n_cores, backend='loky') as parallel:
            result = parallel(delayed(hex_region_boundaries)(hex_bin[k], l, decimals) for k,l in zip(dataset_keys, labels))

        pooled_results = {k:v for k,v in zip(dataset_keys, result)}

        return pooled_results



    def order_points_parallel(boundary_points, n_jobs=None):
        """
        Parallel wrapper around order_points()
        
        Input:
        `boundary_points`(dict): Dictionary with keys for each dataset with the 
            lables for each region as keys and a numpy array with xy coordinates
            for all boundary points.
        `n_jobs`(int): Number of processes for multiprocessing.pool.
            If None it will use the max number of CPU cores.
        Returns:
        Dictionary with for every dataset (same keys as "boundary_points" input) a 
        dictionary with for every label, a list of arrays with the border points of
        every seppearate sub-region in the correct order to make a polygon.   
        Will close the polygon, meaning that the first and last point are idential.
        
        
        """
        dataset_keys = list(boundary_points.keys())
        #datasets = [boundary_points[k] for k in dataset_keys]

        #Paralel execution for all datasets
        #with Pool(processes=n_jobs) as pool:
        #    result = pool.map(order_points, datasets, 1)
            
        n_cores = cpu_count()
        with Parallel(n_jobs=n_cores, backend='loky') as parallel:
            result = parallel(delayed(order_points)(boundary_points[k]) for k in dataset_keys)

        pooled_results = {k:v for k,v in zip(dataset_keys, result)}

        return pooled_results


#############################################################################
#############################################################################
#############################################################################



#############################################################################
#############################################################################
#############################################################################

def read_spot_data(base_folder, full_name=True, rename_columns = {'r_px_microscope_stitched': 'y', 
                                                                  'c_px_microscope_stitched': 'x',
                                                                  'below3Hdistance_genes': 'gene'}):
    """
    Read .parquet data and put dataframes in dictionary.
    
    Has the option to rename the columns with a dictionary.
    
    Input:
    `base_folder`(str): Path to folder with datasets. Will process all 
        ".parquet" files.
    `full_name`(bool): If True uses the full file name. Else it will take the 
        last part after a lower dash "_"
    `rename_columns`(dict): Dictionary with old and new column names. Default:
        {'r_px_microscope_stitched': 'y', 
        'c_px_microscope_stitched': 'x',
        'below3Hdistance_genes': 'gene'}
        Set to None if you want the default column names
    Output:
    Dictionary with file names as keys, with a pandas dataframe with the spot
    localizaion data.
    
    """
    data = {}
    for f in sorted(glob.glob(base_folder + '*.parquet')):
        if full_name:
            name = f.split('/')[-1]
        else:
            name = f.split('_')[-1]
        name = name.split('.')[0]
        df = pd.read_parquet(f)
        if isinstance(rename_columns, dict):
            df = df.rename(columns=rename_columns)
        data[name] = df
    
    return data

def convert_spot_numpy(spots, x_label='r_px_microscope_stitched', y_label='c_px_microscope_stitched', gene_label='below3Hdistance_genes'):
    """
    Convert spot localization data in a pandas dataframe to a standard file.
    
    Converts to a dictionary that for every dataset contains: 
    "xy": A (n, 2) numpy array with x y coordinates.
    "gene": A numpy array with the original gene names for every spot.
    Input:
    spots(dict): Dictionary with for every key a pandas dataframe with x and y 
        coordinates and a gene name column.
    x_label(str): Name of the column containing the x coordinates.
    y_label(str): Name of the column containing the y coordinates.
    gene_label(str): Name of the column containint the gene labels.
    Output:
    Dictionary in with the same keys as the input. Containing the xy 
    coordinates and the gene labels.  
    
    """
    datasets = list(spots.keys())
    converted = {d:{} for d in datasets}
    for d in datasets:
        converted[d]['xy'] = np.array(spots[d].loc[:, [x_label, y_label]])
        converted[d]['gene'] = spots[d].loc[:, gene_label].to_numpy()
    return converted







    
def clust_hex(hex_bin, manifold, n_clusters=None, data=None, clustering=None, 
              labels_input=None, x_lim=None, y_lim=None):
    """
    Cluster hex-bin data. There are multiple options for clustering.
    Either give as input:
    "n_clusters" and "data", to perform KMeans clustering.
    "clustering", for custom clustering.
    "labels_input", if you just want to plot results of existing
        clustering results.    
    Input:
    `hex_bin`(dict): Dictonray with hex bin results for every dataset.
        Output form the make_hexbin() function.
    `manifold`(pd.DataFrame): Dataframe with the manifold embedding
        Like UMAP or tSNE. Should have "c1" and "c2" as index.
    `n_clusters`(int): Number of clusters.
    `data`(pd.DataFrame): Dataframe with data to cluster.
    `clustering`(clustering object): Scikit learn clustering object. 
        Like: "AgglomerativeClustering(n_clusters=40).fit(pca)"
    `labels_input`(array, list): Cluster labels if dataset has been
        clusterd before.
    `x_lim`(tuple): Tuple with (x_min, x_max)
    `y_lim`(tuple): Tuple with (y_min, y_max)

    """
    if clustering == None and not isinstance(labels_input, (np.ndarray, list)):
        clust_result = KMeans(n_clusters=n_clusters).fit(data)
        labels = clust_result.labels_
    elif clustering != None and not isinstance(labels_input, (np.ndarray, list)):
        clust_result = clustering
        labels = clust_result.labels_
    elif clustering == None and isinstance(labels_input, (np.ndarray, list)):
        labels = labels_input
    
    n_rows = math.ceil(len(hex_bin.keys())/2)
    fig = plt.figure(constrained_layout=True, figsize=(10, n_rows*3))
    gs = fig.add_gridspec(n_rows, 4)

    datasets = list(hex_bin.keys())
    for i, d in enumerate(datasets):
        print(f'Clustering. Processing sample: {i}                    ', end='\r')
        ax = fig.add_subplot(gs[int(i/2), (i%2)*2:((i%2)*2)+2])

        ydata = hex_bin[d]['coordinates_filt'][:,0]
        xdata = hex_bin[d]['coordinates_filt'][:,1]

        labels_plot = labels[samples == d]
        
        colors = np.multiply(plt.cm.gist_ncar(labels_plot/max(labels_plot)), [0.9, 0.9, 0.9, 1])

        ax.scatter(xdata, ydata, s=0.1, marker=(6, 0, 30), c=colors)
        ax.set_aspect('equal')

        #set canvas size, assumes that the first section is the largest if not given.
        if x_lim == None or y_lim == None:
            if i == 0:
                ylim = ax.get_ylim()
                xlim = ax.get_xlim()
                print(xlim)
        else:
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)


        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(d)
    plt.suptitle(f'Clustering')

    ax = fig.add_subplot(gs[-1, 2])
    colors_manifold =  np.multiply(plt.cm.gist_ncar(labels/max(labels)), [0.9, 0.9, 0.9, 1])
    ax.scatter(manifold.loc['c1', :], manifold.loc['c2', :], s=0.5, c=colors_manifold)
    ax.set_aspect('equal')
    ax.set_axis_off()

    ax = fig.add_subplot(gs[-1, 3])
    unique = np.unique(labels, return_counts=True)
    colors_unique = np.multiply(plt.cm.gist_ncar(unique[0]/max(unique[0])), [0.9, 0.9, 0.9, 1])
    ax.pie(unique[1], colors=colors_unique)
    
    return labels

#TODO: make paralel, make plotting a seperate function






