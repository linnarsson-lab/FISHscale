import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from diameter_clustering import QTClustering, MaxDiameterClustering
#from diameter_clustering.dist_matrix import compute_sparse_dist_matrix
from sklearn.cluster import DBSCAN, MiniBatchKMeans , AgglomerativeClustering#, OPTICS
from scipy.spatial import distance
import logging

def _get_counts(cell_i_g,dblabel, unique_genes):
    gene, cell = np.unique(cell_i_g,return_counts=True)
    d = pd.DataFrame({dblabel:cell},index=gene)
    g= pd.DataFrame(index=unique_genes)
    data = pd.concat([g,d],join='outer',axis=1).fillna(0)
    return data.values.astype('int16')

def _cell_extract(cell, unique_genes):
    dblabel = cell['Segmentation'][0]
    mat = _get_counts(cell['g'],dblabel, unique_genes)
    centroid = cell['x'],cell['y']
    centroid = sum(centroid[0])/len(centroid[0]), sum(centroid[1])/len(centroid[1])
    return dblabel, centroid, mat

def _distance(data, dist):
    p = data.loc[:,['x','y']].values
    A= p.max(axis=0) - p.min(axis=0)
    A = np.abs(A)
    max_dist =  np.max(A)
    if max_dist <= dist: #*self.pixel_size.magnitude
        return True
    else:
        return False

def _resegmentation_dots(data):
    p = data.loc[:,['x','y']].values
    A= p.max(axis=0) - p.min(axis=0)
    A = np.abs(A)

    if np.max(A) > 50 and data.shape[0] >= 10:#*self.pixel_size.magnitud
        npoints = int(len(p)/800)
        if npoints == 0:
            npoints = 1
        segmentation2 = MiniBatchKMeans(n_clusters=npoints).fit_predict(p).astype(np.int64)
        #logging.info('MiniBatchKMeans Done.')
        sub_max = segmentation2.max()
        segmentation_ = []
        for x in np.unique(segmentation2):
            distd = data[segmentation2 ==x]
            if (segmentation2 == x).sum() >= 10 and x > -1 and _distance(distd, 65):
                pass
                #segmentation_.append(x)
            elif (segmentation2 == x).sum() >= 10 and x > -1 and _distance(distd, 65) == False:
                p2 = p[segmentation2 ==x,:]
                #logging.info('QTC was required on sample size: {}'.format(p2.shape))
                segmentation3= QTClustering(max_radius=25,min_cluster_size=12,metric='euclidean',verbose=False).fit_predict(p2).astype(np.int64) #*self.pixel_size.magnitude
                #segmentation3= MaxDiameterClustering(max_distance=35,metric='euclidean',verbose=False).fit_predict(p2).astype(np.int64) #*self.pixel_size.magnitude
                #segmentation3 = AgglomerativeClustering(n_clusters=None,affinity='euclidean',linkage='ward',distance_threshold=50).fit_predict(p2).astype(np.int64) #*self.pixel_size.magnitude
                segmentation3 = np.array([s3+sub_max if s3 >=0 else -1 for s3 in segmentation3])
                segmentation2[np.where(segmentation2 == x)] = segmentation3

            else:
                segmentation2[np.where(segmentation2 == x)] = -1
            sub_max = segmentation2.max()+1

    else:
        if data.shape[0] >=10:
            segmentation2 = np.array([1]*data.shape[0])
        else:
            segmentation2 = np.array([-1]*data.shape[0])

    #segmentation2 = np.array([x+count if x >= 0 else -1 for x in segmentation2])
    segmentation2 = np.array([x if x >= 0 else -1 for x in segmentation2])
    data['tmp_segment'] = segmentation2

    #resegmentation += segmentation2.tolist()
    #count = np.max(np.array(resegmentation)) + 2
    #resegmentation_data.append(data)
    return data

def _segmentation_dots(partition, func, resegmentation_function):
    cl_molecules_xy = partition.loc[:,['x','y']].values
    segmentation = func.fit_predict(cl_molecules_xy)
    partition['tmp_sement'] = segmentation.astype(np.int64)
    indexes, resegmentation = [],[]
    resegmentation_data = []
    results_resegmentation = Parallel(n_jobs=multiprocessing.cpu_count(),backend="threading")(delayed(resegmentation_function)(part) for _, part in partition.groupby('tmp_sement'))
    resegmentation = []
    new_results_resegmentation = []
    count = 0
    for i in results_resegmentation:
        resegmentation += i['tmp_segment'].tolist()
        segmentation2 = np.array([x+count if x >= 0 else -1 for x in i.tmp_sement])
        i['tmp_segment'] = segmentation2
        new_results_resegmentation.append(i)
        count = np.max(np.array(resegmentation)) + 2
    segmentation = pd.concat(new_results_resegmentation)
    segmentation['Segmentation'] = segmentation['tmp_segment']
    return segmentation
