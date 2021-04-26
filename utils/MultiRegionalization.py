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
from numba import jit, njit
import numba
from pint import UnitRegistry
#Mypy types

class MultiRegionalize:

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