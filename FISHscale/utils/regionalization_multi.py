from numpy.linalg.linalg import norm
from FISHscale.utils.decomposition import Decomposition
from FISHscale.utils.density_1D import Density1D
import gc
import glob
import math
from itertools import combinations, permutations
import geopandas as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
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
from tqdm import tqdm
from typing import Any
import copy
from sklearn.manifold import TSNE, SpectralEmbedding
import dask
import colorsys
from dask.diagnostics import ProgressBar
import itertools

from collections import Counter
import logging

class RegionalizeMulti(Decomposition):
    
    #### HEXAGONAL BINNING ####    
    def hexbin_multi(self, spacing: float, min_count: int) -> dict:
        """Make hexagonal bining of all datasets

        Args:
            spacing (float): Center to center spacing between hexagons
            min_count (int): Minimum count to keep tile i nthe dataset.

        Returns:
            dict: Dictionary with dataset names as keys 
        """
        results = {}
         
        for d in self.datasets:
            r = dask.delayed(d.hexbin_make)(spacing, min_count, n_jobs=1)
            results[d.dataset_name] = {'df_hex': r[0],
                                       'coordinates': r[1]}
        
        with ProgressBar():
            results = dask.compute(results) #It is not paralel because d can not be pickled, but there is still some speed up
        
        return results[0]
    
    #### PLOTTING ####        
    def hexbin_plot(self, c:list, cm=None, gridspec=None, figsize=None, 
                    show_sample_title:bool = True, vmin:float = None,
                     vmax:float = None, save:bool=False, savename:str='',
                     save_format:str='pdf'):
        """Plot spatial multidataset hexbin results

        Args:
            c (list): List of color values to plot for each dataset.
                The length of the list should equal to the number of datasets
                in self.datasets. Each object should either contain an  numpy 
                array with color values as a float between 0 an 1, or a list of
                RGB color values. The length of each sub- list or array should
                match the number of hexagonal tiles.
            cm (plt colormap, optional): The color map to use when the items in
                c are arrays. Defaults to plt.cm.viridis.
            gridspec (plt gridspec, optional): To incorporate this plot into
                another figure, pass a gridspec location. This will be
                subdivided to fit each dataset. If none a new figure will be 
                made. Defaults to None.
            figsize (tuple): Size of figure if not defined by gridspec. 
                If None uses: (sqrt(n_datasets) * 5, sqrt(n_datasets) * 5).
                Defaults to None.
            show_sampel_title (bool): Show titles of individual samples.
                Defaults to True.
            save (bool, optional): Save the plot as .pdf. Defaults to False.
            savename (str, optional): Name of the plot. Defaults to ''.
        """
        #calculate grid
        n_datasets = len(c)
        n_grid = math.ceil(np.sqrt(n_datasets))
        
        #Make figure if gridspec is not provided
        if gridspec == None:
            if figsize == None:
                figsize = (5*n_grid, 5*n_grid)
            fig = plt.figure(figsize=(figsize))
            gridspec = fig.add_gridspec(1,1)[0]
        
        #Subdivide space into required number of axes    
        inner_grid = gridspec.subgridspec(n_grid, n_grid)
        axes = inner_grid.subplots()
        
        #plot data
        for d, col, ((i,j), ax) in zip(self.datasets, c, np.ndenumerate(axes)):
            ax = axes[i,j]
            d.hexbin_plot(c = col, cm=cm, ax=ax, vmin=vmin, vmax=vmax)
            if show_sample_title:
                ax.set_title(d.dataset_name, fontsize=6)
            ax.set_aspect('equal')
            ax.set_axis_off()
            
        #Hide unused axes   
        n_res = (n_grid **2) - n_datasets
        for i in range(n_res):
            i += n_datasets
            ax = axes[int(i/n_grid), i%n_grid]
            ax.set_axis_off()
            
        plt.tight_layout()
        
        #Save figure
        if save:
            plt.savefig(f'{savename}_hexbinmulti.{save_format}', facecolor='white', bbox_inches='tight')
            
    def hexbin_tsne_plot(self, data = None, samples=None, tsne:np.ndarray = None, components: int = 2, 
                         save:bool=False, savename:str=''):
        """Calculate tSNE on hexbin and plot spatial identities.
        
        Calculates a tSNE and makes a color scheme based on the tSNE 
        coordinates, which is then used to color the spatial hexbin data.
        With 3 components the 3 tSNE dimenstions are used for RGB colors.
        With 2 components the angle and distance are used for HSV colors.

        Args:
            data ([pd.DataFrame]): DataFrame with features in rows and samples
                in columns, like from the self.hexbin_make() function. To merge
                multiple datasets use the "self.merge_norm()" function. Best to
                normalize the data and/or apply PCA first.
                Defaults to None.
            samples (np.ndarray): Array with sample labels for each column in
                data. The "self.merge_norm()" function returns this array.
                Defautls to None.                
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
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)
        
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
            ax0 = fig.add_subplot(gs[:, :1])
            
            #Use rotation and distance from centroid to calculate HSV colors. 
            #Calculate rotation
            rotation = np.array([Density1D.get_rotation_rad('self', 0,0, i[0], i[1]) for i in tsne])
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
            ax0.scatter(tsne[:,0], tsne[:,1], s=0.5, c=c)
            ax0.set_aspect('equal')
            ax0.set_axis_off()
            ax0.set_title('tSNE', fontsize=14)
            
            #plot spatial
            c = np.array(c)
            d_c = [list(c[samples == d, :]) for d in self.datasets_names]
            self.hexbin_plot(d_c, gridspec=gs[:, 1:])

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
    
    #### DATA MANIPULATION ####
    def get_dict_item(self, results:dict, item:str)-> list:
        """Return list of items from a nested dictionary.
        
        Assumes the keys are the dataset names

        Args:
            results (dict): Nested dictionary
            item (str): Key to retrieve for each dataset

        Returns:
            list: List with items in the same order as self.datasets_names
        """

        c = []
        for k in self.datasets_names:
            c.append(results[k][item])
        return c
            
    def merge_norm(self, data:list, mode:str=None, plot:bool = False, alternative_dimensions=False, **kwargs):
        """Merge multiple datasets and optionaly normalize before merging.

        Args:
            data (list): List of pandas dataframes in the same order as 
                self.datasets_names, unless "alternative_dimensions" is True.
                In which case each dataset will be named after its index in the
                data list. 
            mode (str, optional):Normalization method. Choose from: "log",
                "sqrt",  "z", "APR" or None. for log +1 transform, square root 
                transform, z scores or Analytic Pearson residuals respectively.
                When the mode is None, no normalization will be performed and
                the input is the output. Defaults to None.
            plot (bool, optional): Plot a histogram of the data. Usefull to
                evaluate normalization performace. Defaults to False.
            alternative_dimensions (bool, optional): Set to True if "data" is 
                not in the same order or dimensions as self.datasets.
                Defaults to False.

        Kwargs:
            Will be passed to the APR() normalization function.
        
        Returns:
            Tuple with:
            [pd.DataFrame]: Merged dataframe.
            [np.ndarray]: Array with sample labels to match columns in the
                merged dataframe with the original dataset name. 
        """
        nrows = math.ceil(len(data)/2)
        norm = mode != None
        if plot:
            fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=(10,1.5*nrows), sharey=True, sharex=True)
        
        if alternative_dimensions == True:
            names = [str(i) for i in range(len(data))]
        else:
            names = self.datasets_names
            
        df_all = []
        samples = []
        for i, (df_next, name) in tqdm(enumerate(zip(data, names))):
            samples.append([name] * df_next.shape[1])
            if norm: 
                 df_next = self.normalize(df_next, mode=mode, **kwargs)
            df_all.append(df_next)
            
            if plot:    
                ax = axes[int(i/2), i%2]
                ax.hist(df_next.sum(), bins=100)
                title = name
                if norm: 
                    title += ' normalized'
                ax.set_title(title)
                ax.set_ylabel('Frequency')
                ax.set_xlabel('Sum molecule count')
                
        if plot:
            plt.tight_layout()
                
        df_all = pd.concat(df_all, axis=1, sort=False)
        samples = np.array(list(itertools.chain.from_iterable(samples)))
        return df_all, samples
    
    def cluster_mean(self, data: Any, labels: np.ndarray) -> Any:
        """Calculate cluster mean.

        For a DataFrame with samples in columns, calculate the mean expression
            values for each unique label in labels.

        Args:
            data (pd.DataFrame): Pandas DataFrame with samples in columns.
            labels (np.ndarray): Numpy array with cluster labels

        Returns:
            [pd.DataFrame]: Pandas Dataframe with mean values for each label.

        """
        unique_labels = np.unique(labels)
        cluster_mean = pd.DataFrame(data=np.zeros((data.shape[0], len(unique_labels))), index = data.index,
                                    columns=unique_labels)

        #Loop over clusters
        for l in unique_labels:
            filt = labels == l
            #Get mean expression of cluster
            cluster_mean.loc[:, l] = data.loc[:, filt].mean(axis=1)

        return cluster_mean
    
    def make_cluster_correlation(self, data:dict, normalized:bool = True, method:str = 'pearson'):
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
        if normalized:
            target = 'df_norm'
        else:
            target = 'df_mean'
        
        #Get datasets
        dfs = self.get_dict_item(data, target)
        
        #make dataset specific labels
        dataset_labels = []
        for i, d in enumerate(dfs):
            for l in d.columns:
                dataset_labels.append(f'{i}_{l}')
        
        #Merge datasets
        df = pd.concat(dfs, axis=1)
        df.columns = dataset_labels
        
        #Make correlation matrix
        return df.corr(method=method)
        
    #### REGIONALIZATION ####
    def similarity_network_correlation(self, data:dict, normalized:bool=True, cutoff:float=0.7, 
                                       method:str='pearson', plot:bool=False):
        """
        Make links between correlating clusters and return a network.
        
        Makes links of the highest correlating clusters above the cutoff.
        
        TODO: NOW IT ONLY CORRELATES WITH THE NEXT DATASET BUT HAVE IT SKIP ONE SO THAT A SINGLE BAD DATASET SHOULD NOT 
        MESS UP THE LINKAGE
        
        Input:
        data (dictionary): Dictionary with the "regionalize_cluster()" results.
        normalized (bool, optional): If True uses the normalized data that was
            made by the "regionalize_cluster()" function. Defaults to True.
        cutoff (float, optional): Cutoff correlation coefficient above which
            clusters will be linked. Defaults to 0.7.
        method (str, optional): Correlation method to use. Choose from: 
            "pearson", "kendall", "spearman". Defaults to "pearson" 
        plot (bool, optional): If true plots the generated network.
            Defaults to False.

        Returns:
            Networkx Network with edges between nodes that are correlated with each 
            other above the cutoff.
        
        """
        if normalized:
            target = 'df_norm'
        else:
            target = 'df_mean'
        
        edge_list = []
        datasets = list(data.keys())
        dix = lambda x: int(x.split('_')[0])
        
        for i, (d0, d1) in enumerate(zip(self.datasets_names, self.datasets_names[1:])):
            df0 = data[d0][target]
            df1 = data[d1][target]
            
            #make dataset specific labels
            dataset_labels = []
            for j, d in enumerate([df0, df1]):
                for l in d.columns:
                    dataset_labels.append(f'{i + j}_{l}')
            
            #Merge datasets
            df = pd.concat([df0, df1], axis=1)
            df.columns = dataset_labels
            
            #Make correlation matrix
            links =  df.corr(method=method).stack().reset_index()
            links.columns = ['Var1', 'Var2', 'value']
            #Filter the links
            links = links.loc[links['Var1'].str.startswith(str(i))]
            links = links.loc[links['Var2'].str.startswith(str(i+1))]
            links = links.loc[(links['value'] < 1) & (links['value'] > cutoff)]
            links = links.sort_values('value', ascending=False)
            #Select highest correlating links, one per cluster
            links_edges = []
            used_var1 = []
            used_var2 = []
            for i, [v1, v2, val] in links.iterrows():
                if v1 not in used_var1 and v2 not in used_var2:
                    links_edges.append([v1, v2, val])
                    used_var2.append(v2)
                    used_var1.append(v1)
            
            edges = pd.DataFrame.from_records(links_edges, columns =['Var1', 'Var2', 'value'])
            edge_list.append(edges)
        
        #Merge all links    
        edges_all = pd.concat(edge_list, axis=0)
        #return edges_all
    
        #Network
        G = nx.from_pandas_edgelist(edges_all, source='Var1', target='Var2', edge_attr='value')

        #Plot
        if plot:
            fig = plt.figure(figsize=(10,10))
            nx.draw(G, with_labels=True, node_color='orange', node_size=20, edge_color='gray', 
                    linewidths=1, font_size=10,ax=plt.gca())
        
        return G

    def merge_labels(self, G, labels:list, reorder_labels:bool = True, data:list=None,
                     mode:str='APR') -> list:
        """Merge cluster labels based on network with linked clusters.
        
        Args:
            G ([networkx]): Network with links between cluster labels that 
                need to be merged. 
            labels (list): List of numpy arrays with the cluster labels for 
                each dataset. Labels should have the format 
                '<dataset index>_<cluster label>' like '1_3' for dataset 1 
                cluster 3. 
            reorder_labels (bool, optional): If True the clsuter labels will be
                reordered based on similarity. Uses (merged)cluster mean 
                expression as input for SpectralEmbedding to order labels.
                When True the parameters "data" and "mode" need to be defined.
                Defaults to True.
            data (list, optional): List of hexbin dataframes where the order
                matches self.datasets_names. Defaults to None.
            mode (str, optional): Normalization mode to use for merging 
                datasets. Defaults to 'APR'.

        Returns:
            list: List of numpy arrays with new cluster labels for each 
            dataset. Order is the same as self.datasets_names. 
        """
        
        #make dictionary with new label for a group
        merge_dict = {}
        merged_clusters = []
        for i, group in enumerate(nx.connected_components(G)):
            for g in group:
                merge_dict[g] = f'merge_{i}'
                merged_clusters.append(g)
        
         
        #Make labels to have a dataset index. Like '1_3' for dataset 1 cluster 3       
        dataset_label = []
        for i, l in enumerate(labels):
            dataset_label.append(np.array([f'{i}_{j}' for j in l]))
            
        #Replace original labels with merged label. 
        merged_label = []
        for dl in dataset_label:
            dl_list = []
            for l in dl:
                if l in merged_clusters:
                    dl_list.append(merge_dict[l])
                else:
                    dl_list.append(l)
            merged_label.append(np.array(dl_list))
        
        #Change labels to integers
        unique_labels = np.unique(np.concatenate(merged_label))
        int_label_dict = dict(zip(unique_labels, range(unique_labels.shape[0])))
        int_labels = []
        for l in merged_label:
            int_labels.append(np.array([int_label_dict[i] for i in l]))
            
        if reorder_labels:
            #Merge datasets
            merged_data, samples = self.merge_norm(data, mode=mode)
            #Merge labels
            all_labels = np.concatenate(int_labels)
            
            #Calculate cluster mean
            cluster_mean = self.cluster_mean(merged_data, all_labels)
            #Order clusters
            manifold = SpectralEmbedding(n_components=1).fit_transform(cluster_mean.T)
            #even_spaced = np.linspace(0, 1, manifold.shape[0])
            even_spaced = np.arange(manifold.shape[0])
            even_spaced_dict = dict(zip(np.sort(manifold.ravel()), even_spaced))
            manifold_even = np.array([even_spaced_dict[i] for i in manifold.ravel()])
            manifold_even_dict = dict(zip(cluster_mean.columns, manifold_even))
            #Reassign labels
            final_labels = []
            for l in int_labels:
                final_labels.append(np.array([manifold_even_dict[i] for i in l]))
            
        else:
            final_labels = int_labels

        return final_labels
        
    def regionalize(self,
                    spacing: float, 
                    min_count: int,
                    feature_selection: np.ndarray = None,
                    normalization_mode: str = 'APR',
                    dimensionality_reduction: str = 'PCA', 
                    n_components: list = [0,100],
                    clust_dist_threshold: float = 70,
                    n_clusters: int=None,
                    clust_neighbor_rings: int = 1,
                    smooth: bool = False,
                    smooth_neighbor_rings: int = 1, 
                    post_merge: bool = False,
                    post_merge_t: float = 0.05,
                    smooth_cycles: int = 1,
                    merge_labels: bool = True,
                    merge_cutoff: float = 0.7,
                    correlation_method = 'pearson',
                    reorder_labels: bool = True) -> dict:
        """Regionalize and cluster individual datasets.

       Args:
             spacing (float): distance between tile centers, in same units as 
                the data. The function makes hexagons with the point up: ⬡
            min_count (int):  Minimal number of molecules in a tile to keep the 
                tile in the dataset. The algorithm will generate a lot of empty 
                tiles, which are later discarded using the min_count threshold.
                Suggested to be at least 1.
            feature_selection (np.ndarray, optional): Array of genes to use.
                If none is provided will run on all genes. Defaults to None.
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
            n_clusters (int, optional): Number of desired clusters. Either this
                or clust_dist_threshold should be provided. Defaults to None.
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
            merge_labels (bool, optional): If True, the cluster labels of the
                regionalization of the individual sections are merged based on
                correlation to link one dataset to the next. Defaults to True.
            merge_cutoff (float, optional): Cutoff correlation coefficient 
                above which clusters will be linked. Defaults to 0.7.
            correlation_method (str, optional): Correlation method to use for
                mergin. Choose from: "pearson", "kendall", "spearman".
                Defaults to "pearson".
            reorder_labels (bool, optional): Reorder the labels so that similar
                clusters get a label number that is close. Only works when 
                "merge_labels" is set to True. Defaults to True.

        Returns:
            Dict containing:
                - df_hex: Dataframe with counts for each hexagonal tile.
                - labels: Numpy array with cluster labels for each tile.
                - hex_coord: XY coordinates for each hexagonal tile.
                - df_mean: Dataframe with mean count per region.
                - df_norm: Dataframe with mean normalized count per region.
                Optional:
                - labels_merged: Merged labels when "merge_labels" is True.
        """
        #regionalize individual datasets
        results = {}
        for d in self.datasets:
            r = dask.delayed(d.regionalize)(spacing, min_count, feature_selection, normalization_mode, 
                                            dimensionality_reduction, n_components, clust_dist_threshold, 
                                            n_clusters, clust_neighbor_rings, smooth, smooth_neighbor_rings, 
                                            smooth_cycles, post_merge, post_merge_t, order_labels=False, n_jobs=1)
            results[d.dataset_name] = ({'df_hex': r[0],
                                       'labels': r[1],
                                       'coordinates': r[2],
                                       'df_mean': r[3],
                                       'df_norm': r[4]})
        
        with ProgressBar(): 
            collection = dask.compute(results)
        collection = collection[0]

        
        if merge_labels:
            #make similarity network based on correlation
            G = self.similarity_network_correlation(collection, normalized=True, cutoff=merge_cutoff, 
                                        method=correlation_method, plot=False)
            
            #Merge labels
            merged_labels = self.merge_labels(G, self.get_dict_item(collection, 'labels'),
                                              reorder_labels=reorder_labels, 
                                              data=self.get_dict_item(collection, 'df_hex'),
                                              mode=normalization_mode)
            
            for i, d in enumerate(self.datasets_names):
                collection[d]['labels_merged'] = merged_labels[i]

        return collection
    
    def merged_cluster_mean(self, reg:dict, mode:str=None) -> Any:
        """Calculate cluster  after label merging.

        For a DataFrame with samples in columns, calculate the mean expression
            values for each unique label in labels.

        Args:
            reg (dict): Dictionary with results from the regionalize() 
                function.
            mode (str): Normalization mode. Data can be normalized before
                merging. 

        Returns:
            [pd.DataFrame]: Pandas Dataframe with mean values for each label.

        """
        labels_merged = self.get_dict_item(reg, 'labels_merged')
        labels_merged_concat = np.concatenate(labels_merged)
        unique_labels = np.unique(labels_merged_concat)
        
        data, samples = self.merge_norm(self.get_dict_item(reg, 'df_hex'), mode=mode)
        
        cluster_mean = pd.DataFrame(data=np.zeros((data.shape[0], len(unique_labels))), index = data.index,
                                    columns=unique_labels)
        #return cluster_mean, data
        #Loop over clusters
        for l in unique_labels:
            filt = labels_merged_concat == l
            #Get mean expression of cluster
            cluster_mean.loc[:, l] = data.loc[:, filt].mean(axis=1)

        return cluster_mean
    
    
        
        

        
        
        
        

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
            logging.info(f'Merging datasets. Processing sample: {i}                    ')
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
            n_jobs = self.cpu_count()
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
            logging.info(f'Connected clustering. Processing sample: {i}                    ',)
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
        logging.info(type(correlation_matrix))
        logging.info(type(links))
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
        logging.info(f'Clustering. Processing sample: {i}                    ', )
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
                logging.info(xlim)
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