import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

class ClusterCleaner:
    """
    Class for storing clustering information.
    """
    def __init__(self, genes, clusters):
        self.barcodes_df = pd.DataFrame({'Gene': genes, 'cluster': clusters})
        # Create cluster gene expression matrix
        

    def merge(self):
        hm = self.barcodes_df.groupby(['Gene','cluster']).size().unstack(fill_value=0)
        logging.info(hm.shape)
        # Z-score normalization

        scaler = StandardScaler()
        hm = pd.DataFrame(scaler.fit_transform(hm.values), columns=hm.columns, index=hm.index)
        hm.head()

        hm_merge = self.post_merge(hm, hm.columns, 0.01, linkage_metric='correlation', linkage_method='average', name='SupFig3Dend', save=True)
        logging.info(hm_merge)

        hm_merge = np.array(hm_merge)
        unique_clusters= np.unique(hm_merge)
        dic = dict(zip(unique_clusters, np.arange(unique_clusters.shape[0])))
        hm_merge = np.array([dic[i] for i in hm_merge])

        dic = dict(zip(np.arange(np.unique(self.barcodes_df['cluster']).shape[0]), hm_merge))

        clusters = np.array([dic[i] for i in self.barcodes_df['cluster']])
        logging.info('Number of clusters after merging: {}'.format(unique_clusters.shape[0]))
        return clusters

    # Auxiliary functions for merging clusters
    def post_merge(self,df, labels, post_merge_cutoff, linkage_method='single', 
                linkage_metric='correlation', fcluster_criterion='distance', name='', save=True):
        """
        Merge clusters based on likage and a cutoff. The mean expression levels of 
        the clusters are linked and them merged based on the cutoff provided.
        Input:
        `df`(Pandas dataframe): df with expression matrix. row-genes, col-cells.
        `labels`(list/array): Labels of the cells.
        `post_merge_cutoff`(float): Merge clusters that have a distance from each 
            other below the cutoff.
        `linkage_method`(string): Scipy linkage methods. Default = 'single'
        `linkage_metric`(string): Scipy lingae metric. Default = 'correlation'
        `fcluster_criterion`(string): Scipy fcluster criterion. Default = 'distance'

        Returns:
        `new_labels`(list): List of new cell labels after merging. 
        Additionally it plots the dendrogram showing which clusters are merged.

        """
        Z = scipy.cluster.hierarchy.linkage(df.T, method=linkage_method, metric=linkage_metric)
        merged_labels_short = scipy.cluster.hierarchy.fcluster(Z, post_merge_cutoff, criterion=fcluster_criterion)

        #Update labels  
        label_conversion = dict(zip(df.columns, merged_labels_short))
        label_conversion_r = dict(zip(merged_labels_short, df.columns))
        new_labels = [label_conversion[i] for i in labels] 

        #Plot the dendrogram to visualize the merging
        fig, ax = plt.subplots(figsize=(20,10))
        scipy.cluster.hierarchy.dendrogram(Z, labels=df.columns ,color_threshold=post_merge_cutoff)
        ax.hlines(post_merge_cutoff, 0, ax.get_xlim()[1])
        ax.set_title('Merged clusters')
        ax.set_ylabel(linkage_metric, fontsize=20)
        ax.set_xlabel('pre-merge cluster labels', fontsize=20)
        ax.tick_params(labelsize=10)
        
        if save == True:
            fig.savefig('ClusterCorrelation.png'.format(name), dpi=500)

        return new_labels



    def gen_labels(self,df, model):
        """
        Generate cell labels from model.
        Input:
        `df`: Panda's dataframe that has been used for the clustering. (used to 
        get the names of colums and rows)
        `model`(obj OR array): Clustering object. OR numpy array with cell labels.
        Returns (in this order):
        `cell_labels` = Dictionary coupling cellID with cluster label
        `label_cells` = Dictionary coupling cluster labels with cellID
        `cellID` = List of cellID in same order as labels
        `labels` = List of cluster labels in same order as cells
        `labels_a` = Same as "labels" but in numpy array
        
        """
        if str(type(model)).startswith("<class 'sklearn.cluster"):
            cell_labels = dict(zip(df.columns, model.labels_))
            label_cells = {}
            for l in np.unique(model.labels_):
                label_cells[l] = []
            for i, label in enumerate(model.labels_):
                label_cells[label].append(df.columns[i])
            cellID = list(df.columns)
            labels = list(model.labels_)
            labels_a = model.labels_
        elif type(model) == np.ndarray:
            cell_labels = dict(zip(df.columns, model))
            label_cells = {}
            for l in np.unique(model):
                label_cells[l] = []
            for i, label in enumerate(model):
                label_cells[label].append(df.columns[i])
            cellID = list(df.columns)
            labels = list(model)
            labels_a = model
        else:
            logging.info('Error wrong input type')
        
        return cell_labels, label_cells, cellID, labels, labels_a

