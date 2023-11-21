from multiprocessing import cpu_count
from os import path, makedirs, environ
from re import L, X
#environ['NUMEXPR_MAX_THREADS'] = str(cpu_count())
from typing import Union, Optional
import pandas as pd
from FISHscale.utils.inside_polygon import close_polygon 
from FISHscale.utils.hex_regionalization import Regionalize
from FISHscale.utils.fast_iteration import Iteration, MultiIteration
from FISHscale.utils.colors import ManyColors
from FISHscale.utils.gene_correlation import GeneCorr
from FISHscale.utils.spatial_metrics import SpatialMetrics
from FISHscale.utils.density_1D import Density1D
from FISHscale.utils.normalization import Normalization
from FISHscale.visualization.gene_scatter import GeneScatter, MultiGeneScatter, AttributeScatter
from FISHscale.utils.data_handling import DataLoader, DataLoader_base
from FISHscale.utils.bonefight import BoneFight, BoneFightMulti
from FISHscale.utils.regionalization_multi import RegionalizeMulti
from FISHscale.utils.decomposition import Decomposition
from FISHscale.spatial.boundaries import Boundaries, Boundaries_Multi
from FISHscale.spatial.gene_order import Gene_order
from FISHscale.segmentation.cellpose import Cellpose
from FISHscale.utils.regionalization_gradient import Regionalization_Gradient, Regionalization_Gradient_Multi
from FISHscale.utils.volume_align import Volume_Align
from FISHscale.utils.binning import Binning, Binning_multi
import sys
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import numpy as np
import loompy
from pint import UnitRegistry
from os import name as os_name
from glob import glob
from time import strftime
from math import ceil
from dask import dataframe as dd
import dask
from dask.distributed import Client, LocalCluster
try:
    from pyarrow.parquet import ParquetFile
except ModuleNotFoundError as e:
    logging.info(f'Please install "pyarrow" to load ".parquet" files. Without only .csv files are supported which are memory inefficient. Error: {e}')
from tqdm import tqdm
from difflib import get_close_matches
import logging
from joblib import Parallel, delayed
import multiprocessing
from diameter_clustering import QTClustering #, MaxDiameterClustering
from sklearn.cluster import DBSCAN, MiniBatchKMeans #, AgglomerativeClustering#, OPTICS
from scipy.spatial import distance
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',stream=sys.stdout, level=logging.INFO,force=True,)


class Dataset(Regionalize, Iteration, ManyColors, GeneCorr, GeneScatter, AttributeScatter, SpatialMetrics, DataLoader, Normalization, 
              Density1D, BoneFight, Decomposition, Boundaries, Gene_order, Cellpose, 
              Regionalization_Gradient, Binning):
    """
    Base Class for FISHscale, still under development

    Add methods to the class to run segmentation, make loom files, visualize (spatial localization, UMAP, TSNE).
    """

    def __init__(self,
        filename: str,
        x_label: str = 'r_transformed',
        y_label: str = 'c_transformed',
        z_label: str = None,
        z: float = 0,
        gene_label: str = 'decoded_genes',
        other_columns: Optional[list] = [],
        unique_genes: Optional[np.ndarray] = None,
        exclude_genes: list = None,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        polygon: np.ndarray = None,
        select_valid: Union[bool, str] = False,
        reparse: bool = False,
        color_input: Optional[Union[str, dict]] = None,
        working_selection: str = None,
        verbose: bool = True,
        part_of_multidataset: bool = False,
        image=None,
        ):
        """initiate Dataset

        Args:
            filename (str): Name (and  optionally path) of the saved Pandas 
                DataFrame to load.
            x_label (str, optional): Name of the column of the Pandas dataframe
                that contains the X coordinates of the points. Defaults to 
                'r_px_microscope_stitched'.
            y_label (str, optional): Name of the column of the Pandas dataframe
                that contains the Y coordinates of the points. Defaults to 
                'c_px_microscope_stitched'.
            z_label (str, optional): Name of the column of the datafile
                that contains the Z coordinates of the points. If None, the Z
                coordinate will default to the z value (see below).
                Defaults to None.
            z (float, optional): Z coordinate of the dataset. The z coordinate
                is also a way to order MultiDatasets. So if your dataset 
                already contains z values you can still set this. 
                Defaults to zero.
            gene_label (str, optional):  Name of the column of the Pandas 
                dataframe that contains the gene labels of the points. 
                Defaults to 'decoded_genes'.
            other_columns (list, optional): List with labels of other columns 
                that need to be loaded. Defaults to [].
            unique_genes (np.ndarray, optional): Array with unique gene names.
                This can also be a selection of genes to load. After a
                selection is made data needs to be re-parsed to include all
                genes.                
                If not provided it will find the unique genes from the 
                gene_column. This is slow for > 10e6 rows. 
                Defaults to None.
            exclude_genes (list, optional): List with genes to exclude from
                dataset. Defaults to None.     
            pixel_size (str, optional): Unit size of the data. Is used to 
                convert data to micrometer scale. Uses Pint's UnitRegistry.
                Example: "0.1 micrometer", means that each pixel or unit has 
                a real size of 0.1 micrometer, and thus the data will be
                multiplied by 0.1 to make micrometer the unit of the data.
                Defaults to "1 micrometer".
            x_offset (float, optional): Offset in X axis. Defaults to 0.
            y_offset (float, optional): Offset in Y axis. Defaults to 0.
            z_offset (float, optional): Offset in Z axis. Defaults to 0.
            polygon (np.ndarray, optional): A numpy array with shape (X,2) with
                a polygon that can be used to select points. If the polygon is
                changed the dataset needs to be re-parsed. If multiple regions 
                need to be selected, a single array containing the points of
                all polygons can be passed as long as each one is closed (First
                and last point are identical). Defaults to None.
            select_valid ([bool, str], optional): If the datafile already 
                contains information which datapoints to include this can be
                used to trim the dataset. The column should contain a boolean
                or binary array where "True" or "1" means that the datapoint
                should be included. 
                A string can be passed with the column name to use. If True is
                passed it will look for the default column name "Valid".
            reparse (bool, optional): True if you want to reparse the data,
                if False, it will repeat the parsing. Parsing will apply the
                offset. Defaults to False.
            color_input (Optional[str, dict], optional): If a filename is 
                specifiedthat endswith "_color_dictionary.pkl" the function 
                will try to load that dictionary. If "auto" is provided it will
                try to load an previously generated color dictionary for this 
                dataset. If "make", is provided it will make a new color 
                dictionary for this dataset and save it. If a dictionary is 
                proivided it will use that dictionary. If None is provided the 
                function will first try to load a previously generated color 
                dictionary and make a new one if this fails. Defaults to None.
            working_selection (str, optional): Datasets can contain multiple 
                boolean selectors, for instance for different anatomical
                regions of the sample like; brain, eye, jaw. If you want to
                perform analysis only on a selection, you can set the
                "working_selection" to that name. It is required that the 
                column was initially passed to "other_columns" to be selected.
                Defaults to None.
            verbose (bool, optional): If True prints additional output.
            part_of_multidataset (bool, optional): True if dataset is part of
                a multidataset. 

        """
        #Parameters
        self.verbose = verbose
        self.part_of_multidataset = part_of_multidataset
        self.cpu_count = cpu_count()
        self.z = z
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.polygon = close_polygon(polygon) if isinstance(polygon, np.ndarray) else polygon
        self.select_valid = select_valid
        if not isinstance(other_columns, list):
            other_columns = [other_columns]
        self.other_columns = other_columns
        
        #Dask
        #if not self.part_of_multidataset:
        #    self.cluster = LocalCluster()
        #    self.client = Client(self.cluster)
        #    logging.info(f'Dask dashboard link: {self.client.dashboard_link}')

        #Files and folders
        self.filename = filename
        self.dataset_name = path.splitext(path.basename(self.filename))[0]
        self.dataset_folder = path.dirname(path.abspath(filename))
        self.FISHscale_data_folder = path.join(self.dataset_folder, f'{self.dataset_name}_FISHscale_Data')
        makedirs(self.FISHscale_data_folder, exist_ok=True)
        
        #Handle scale
        self.ureg = UnitRegistry()
        self.pixel_size = self.ureg(pixel_size)
        self.pixel_size = self.pixel_size.to('micrometer')
        self.pixel_area = self.pixel_size ** 2
        self.unit_scale = self.ureg('1 micrometer')
        self.area_scale = self.unit_scale ** 2
        self.x_offsets = [x_offset]
        self.y_offsets = [y_offset]
        self.z_offsets = [z_offset]
        
        #Load data
        self.load_data(self.filename, x_label, y_label, gene_label, self.other_columns, x_offset, y_offset, z_offset, 
                       self.pixel_size.magnitude, unique_genes, exclude_genes, self.polygon, self.select_valid, 
                       reparse, z_label)

        #Gene metadata
        self.gene_index = dict(zip(self.unique_genes, range(self.unique_genes.shape[0])))
        self.gene_n_points = self._get_gene_n_points()

        #Set up divisions of dataframe so that rows of partitions can be indexed
        div = np.cumsum([self.gene_n_points[g] for g in self.unique_genes])
        div = np.insert(div, 0, 0)
        self.df.divisions = tuple(div)

        #Handle colors
        self.auto_handle_color_dict(color_input)
        
        #Working selection
        self._working_selection = working_selection
        self._working_selection_options = other_columns
        
        #Verbosity
        self.verbose = verbose
        self.vp(f'Loaded: {self.dataset_name}')
        if image is None:
            self.image = None
        else:
            self.image = self.read_image(image)
            
    def vp(self, *args):
        """Function to print output if verbose mode is True
        """
        if self.verbose:
            for arg in args:
                logging.info('    ' + str(arg))
                
    def check_gene_input(self, gene):
        """Check if gene is in dataset. If not give suggestion.
        """            
        if not gene in self.unique_genes:
            raise Exception(f'Given gene: "{gene}" can not be found in dataset. Did you maybe mean: {get_close_matches(gene, self.unique_genes, cutoff=0.4)}?')
                
    def offset_data_temp(self, x_offset: float = 0, y_offset: float = 0, z_offset: float = 0):
        """Offset the data with the given offset values.
        
        This will not permanently affect the parsed data. 
        
        Args:
            x_offset (float): Offset in X axis. 
            y_offset (float): Offset in Y axis.
            z_offset (float): Offset in Z axis.
        """
        if x_offset != 0:
            self.x_offset += x_offset
            self.df.x += x_offset
            self.x_min += x_offset
            self.x_max += x_offset
        if y_offset != 0:
            self.y_offset += y_offset
            self.df.y += y_offset
            self.y_min += y_offset
            self.y_max += y_offset
        if z_offset != 0:
            self.z_offset += z_offset
            self.df.z += z_offset
            self.z_min += z_offset
            self.z_max += z_offset

        self.x_extent = self.x_max - self.x_min
        self.y_extent = self.y_max - self.y_min 
        self.z_extent = self.z_max - self.z_min 
        self.xyz_center = (self.x_max - 0.5*self.x_extent, self.y_max - 0.5*self.y_extent, self.z_max - 0.5*self.z_extent)

    def read_image(self, filename):
        #Read image
        from skimage import img_as_bool
        import cv2
        logging.info(f'Reading image: {filename}')
        
        if filename.count('.zarr'):
            import zarr
            print(filename)
            img = zarr.load(filename)
            img[img >0] = 1
            logging.info(f'Rescaling image')
            width = int(img.shape[1] * .27)
            height = int(img.shape[0] * .27)
            dim = (width, height)
            # resize image
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img[img >0] = 1
            img = img_as_bool(img)*255
            img = img.astype('uint8')
            img[img > 0] = 1
            img = img_as_bool(img)
            mx = np.max(img.shape)
            square_image = np.zeros([mx,mx])
            square_image[:img.shape[0],:img.shape[1]] = img
        elif filename.count('.tif'):
            import tifffile
            img = tifffile.imread(filename)
            img[img >0] = 1
            img_d = rescale(img, self.pixel_size.magnitude, anti_aliasing=False)
            img_d[img_d >0] = 1
            img_d = img_as_bool(img_d)
        return square_image

    def visualize(self,
                remote=False,
                columns:list=[],
                color_dic=None,
                x=None,
                y=None,
                c={},
                ):
        """
        Run open3d visualization on self.data
        Pass to columns a list of strings, each string will be the class of the dots to be colored by. example: color by gene

        Args:
            columns (list, optional): List of columns to be plotted with different colors by visualizer. Defaults to [].
            width (int, optional): Frame width. Defaults to 2000.
            height (int, optional): Frame height. Defaults to 2000.
        """
        from FISHscale.visualization.vis_macos import Window
        from open3d.visualization import gui
        if remote:
            import open3d as o3d
            o3d.visualization.webrtc_server.enable_webrtc()

        
        gui.Application.instance.initialize()
        if self.color_dict:
            color_dic = self.color_dict

        self.window = Window(self,
                        columns,
                        color_dic,
                        x_alt=x,
                        y_alt=y,
                        c_alt=c)
        
        gui.Application.instance.run()
            
    def set_working_selection(self, level: Union[None, str] = None):
        """Set the working selection on which to work.
        
        If your dataset contains columns with boolean filters for certain 
        subsets of the data, you can select on which sub-selection you work 
        by setting the working_selection. This can for instance be different
        anatomical regions in your dataset.
        Setting the level to "None" resets the working selection. 

        Args:
            level (Union[None, str], optional): _description_. Defaults to None.
        """
        if level == None or level in self._working_selection_options:
            self._working_selection = level
            self.vp(f'Working selection set to: {self._working_selection}')
        else:
            raise Exception(f'Selection: "{level}" is not found in dataset, choose from: {self.other_columns}')
    
    def reset_working_selection(self):
        """Reset the working selection to include all datapoints.
        """
        self.set_working_selection(level = None)
        
    def segment(self,
                label_column,
                save_to=None,
                segmentation_function=None,
                ):
                    
        from tqdm import trange
        from scipy import sparse
        from dask.diagnostics import ProgressBar
        from dask import dataframe as dd
        import shutil
        #import modin.pandas as pd

        """
        Run DBscan segmentation on self.data, this will reassign a column on self.data with column_name

        Args:
            eps (int, optional): [description]. Defaults to 25.
            min_samples (int, optional): [description]. Defaults to 5.
            column_name (str, optional): [description]. Defaults to 'cell'.
            cutoff (int,optional): cells with number of dots above this threshold will be removed and -1 passed to background dots
        """        
            
        logging.info('Running segmentation by: {}'.format(label_column))
        if path.exists(path.join(save_to,'Segmentation')):
            shutil.rmtree(path.join(save_to,'Segmentation'))
            makedirs(path.join(save_to,'Segmentation'))
        else:
            makedirs(path.join(save_to,'Segmentation'))

        count = 0
        #partition_count = 0
        matrices, labels_list, centroids, polygons, clusters = [], [], [], [], []
        segmentation_results = []
        labels_segmentation = []
        logging.info('Segmentation V2')
        for nx in trange(self.dask_attrs[label_column].npartitions - 1):
            partition = self.dask_attrs[label_column].get_partition(nx).compute()
            logging.info('Initiation segmentation of cluster: {}'.format(nx))
            partition = _segmentation_dots(partition, segmentation_function)
            logging.info('Segmentation done.')
            s = partition.Segmentation.values
            
            unique_counts, vals = np.unique(np.array([-1]+s.tolist()), return_counts=True)
            dic_vals = dict(zip(unique_counts, vals))
            dic = dict(
                        zip(
                            unique_counts,
                            [-1]+np.arange(unique_counts.shape[0]).tolist(), 
                            )
                        )
            s = np.array([dic[x] if x >= 0 and dic_vals[x] >= 10  else -1 for x in s]) 
            unique_counts, vals = np.unique(np.array([-1]+s.tolist()), return_counts=True)
            dic_vals = dict(zip(unique_counts, vals))
            dic = dict(
                        zip(
                            unique_counts,
                            [-1]+np.arange(unique_counts.shape[0]).tolist(), 
                            )
                        )
            labels = np.array([dic[x]+count if x >= 0 and dic_vals[x] >= 10  else -1 for x in s]) 
            logging.info('Segmentation of label {}. Min label: {} and max label: {}'.format(nx, labels.min(), labels.max()))
            partition['Segmentation'] = labels

            if labels.max() >= 0:
                labels_segmentation += labels.tolist()
                count =  np.max(np.array(labels_segmentation)) +1
                logging.info('Saving partition')
                #partition_grp = partition.groupby('Segmentation'
                clusterN = partition[label_column].values[0]
                partition.to_parquet(path.join(save_to,'Segmentation','{}.parquet'.format(clusterN)))
                logging.info('Filter -1 out of partition')
                partition_filt = partition[partition.Segmentation != -1]
                #partition_filt = dd.from_pandas(partition_filt, npartitions=len(partition_filt.Segmentation.unique()))
                #result_grp = partition_filt.groupby('Segmentation').apply(_cell_extract, self.unique_genes).compute().values

                #result_grp = Parallel(
                #    n_jobs=multiprocessing.cpu_count(), backend='threading',max_nbytes=None)(delayed(_cell_extract)([part, self.unique_genes]) for _, part in partition_filt.groupby('Segmentation'))
                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    task = [(part, self.unique_genes) for _, part in partition_filt.groupby('Segmentation')]
                    results_grp = executor.map(_cell_extract, task)

                    for dbl, centroid, mat in results_grp:
                        if type(mat) != type(None):
                            labels_list.append(dbl)
                            centroids.append(centroid)
                            matrices.append(mat)
                            clusters.append(clusterN)
                            polygons.append(centroid)
                
                #partition[label_column] = np.ones_like(partition['Segmentation'].values)*nx
            else:
                logging.info('GScluster did not produce any cells, removing number {} from the list'.format(nx))

        matrices = np.concatenate(matrices,axis=1)
        logging.info('Shape of gene X cell matrix: {}'.format(matrices.shape))
        if type(save_to) == type(None):
            file = path.join(self.dataset_folder,self.filename.split('.')[0]+'_cells.loom')
        else:
            file = path.join(save_to,self.filename.split('.')[0]+'_cells.loom')
        row_attrs = {'Gene':self.unique_genes}
        col_attrs = {'Segmentation':labels_list, 'Centroid':centroids,label_column:clusters, 'Polygons':polygons}# 'Polygon':polygons
        matrices = sparse.csr_matrix(matrices,dtype=np.int16)
        logging.info('Saving polygons')
        np.save(path.join(save_to,self.filename.split('.')[0]+'_polygons'),polygons)
        np.save(path.join(save_to,self.filename.split('.')[0]+'_molecule_labels'),segmentation_results)
        loompy.create(file,matrices,row_attrs,col_attrs)
        logging.info('Number of cells found: {}. Loompy written.'.format(count))

#@numba.njit(parallel=True)
def _get_counts(cell_i_g,dblabel, unique_genes):
    gene, cell = np.unique(cell_i_g,return_counts=True)
    d = pd.DataFrame({dblabel:cell},index=gene)
    g= pd.DataFrame(index=unique_genes)
    data = pd.concat([g,d],join='outer',axis=1).fillna(0)
    return data.values.astype('int16')

#@numba.njit(parallel=True)
def _cell_extract(cell_unique_genes):
    cell, unique_genes = cell_unique_genes
    #dblabel = cell['Segmentation'][0]
    #try:
    dblabel = cell.Segmentation.values[0]
    #mat = _get_counts(cell.g.values,dblabel,unique_genes)
    gene, cell_counts = np.unique(cell.g.values,return_counts=True)
    data = np.zeros(len(unique_genes))
    data[np.where(np.isin(unique_genes, gene))[0]] = cell_counts
    mat = data.reshape([len(data),1])
    #mat = _get_counts(cell['g'],dblabel, unique_genes)
    centroid = cell.x.values.mean(),cell.y.values.mean()
    #centroid = sum(centroid[0])/len(centroid[0]), sum(centroid[1])/len(centroid[1])
    return dblabel, centroid, mat
    #except:
    #    return None, None, None

#@numba.njit(parallel=True)
def _distance(data, dist):
    p = data.loc[:,['x','y']].values
    A= p.max(axis=0) - p.min(axis=0)
    A = np.abs(A)
    max_dist =  np.max(A)
    if max_dist <= dist: #*self.pixel_size.magnitude
        return True
    else:
        return False

#@numba.njit(parallel=True)
def _resegmentation_dots(data):
    if len(data) == 0:
        return None
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
            if (segmentation2 == x).sum() >= 10 and x > -1 and _distance(distd, 45):
                pass
                #segmentation_.append(x)
            elif (segmentation2 == x).sum() >= 10 and x > -1 and _distance(distd, 45) == False:
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

    segmentation2 = np.array([x if x >= 0 else -1 for x in segmentation2])
    data['tmp_segment'] = segmentation2
    return data

#@numba.njit(parallel=True)
def _segmentation_dots(partition, func):
    cl_molecules_xy = partition.loc[:,['x','y']].values
    segmentation = func.fit_predict(cl_molecules_xy)
    partition['tmp_segment'] = segmentation.astype(np.int64)
    indexes, resegmentation = [],[]
    resegmentation_data = []

    #results_resegmentation = Parallel(n_jobs=multiprocessing.cpu_count(),backend="loky",max_nbytes=None)(delayed(_resegmentation_dots)(part) for _, part in partition.groupby('tmp_segment'))

    resegmentation = []
    new_results_resegmentation = []
    count = 0

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        logging.info('Starting resegmentation, grouping by tmp_segment...')
        task = [part for _, part in partition.groupby('tmp_segment')]
        logging.info('Starting resegmentation, grouping done.')
        results_resegmentation = executor.map(_resegmentation_dots, task)

        for i in results_resegmentation:
            if i is not None:

                segmentation2 = np.array([x+count if x >= 0 else -1 for x in i.tmp_segment])
                resegmentation += segmentation2.tolist()
                i['tmp_segment'] = segmentation2
                new_results_resegmentation.append(i)
                count = np.max(np.array(resegmentation)) + 2
    segmentation = pd.concat(new_results_resegmentation)
    segmentation['Segmentation'] = segmentation['tmp_segment']
    return segmentation



class MultiDataset(ManyColors, MultiIteration, MultiGeneScatter, DataLoader_base, Normalization, RegionalizeMulti,
                   Decomposition, BoneFightMulti, Regionalization_Gradient_Multi, Boundaries_Multi, Volume_Align,
                   Binning_multi):
    """Load multiple datasets as Dataset objects.
    """

    def __init__(self,
        data: Union[list, str],
        data_folder: str = '',
        unique_genes: Optional[np.ndarray] = None,
        MultiDataset_name: Optional[str] = None,
        color_input: Optional[Union[str, dict]] = None,
        verbose: bool = False,
        grid_layout: bool = False,
        columns_layout: int = 5,
        #If loading from files define:
        x_label: str = 'r_px_microscope_stitched',
        y_label: str = 'c_px_microscope_stitched',
        z_label: str = None,
        gene_label: str = 'decoded_genes',
        other_columns: Optional[list] = [],
        exclude_genes: list = None,
        z: float = 0,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        polygon: Union[np.ndarray, list] = None,
        select_valid: Union[bool, str] = False,
        reparse: bool = False,
        parse_num_threads: int = -1):
        """initiate PandasDataset

        Args:
            data (Union[list, str]): List with initiated Dataset objects, or 
                path to folder with files to load. If a path is given the 
                found files will be processed in alphanumerically sorted order.
                Unique genes must match for all Datasets.
            data_folder (Optional, str): Path to folder with data when "data" 
                is a list of already initiated Datasets. This folder will be 
                used to save MultiDataset metadata. If not provided it will
                save in the current working directory. Defaults to ''.                
            unique_genes (np.ndarray, optional): Array with unique gene names.
                If not provided it will find the unique genes from the 
                gene_column. This is slow for > 10e6 rows. 
                This can also be a selection of genes to load. After a
                selection is made data needs to be re-parsed to make a new
                selection. Defaults to None.
            MultiDataset_name (Optional[str], optional): Name of multi-dataset.
                This is used to store multi-dataset specific parameters, such
                as gene colors. If `None` will use a timestamp.
                Defaults to None.
            color_input (Optional[str, dict], optional): If a filename is 
                specified that endswith `_color_dictionary.pkl` the function 
                will try to load that dictionary. If "auto" is provided it will
                try to load an previously generated color dictionary for this 
                MultiDataset, identified by MultiDataset_name. 
                If "make", is provided it will make a new color dictionary for 
                this dataset and save it. If a dictionary is proivided it will 
                use that dictionary. If None is provided the function will 
                first try to load a previously generated color dictionary and 
                make a new one if this fails. Defaults to None.
            verbose (bool, optional): If True prints additional output.

            #Below input only needed if `data` is a path to files. 
            x_label (str, optional): Name of the column of the Pandas dataframe
                that contains the X coordinates of the points. Defaults to 
                'r_px_microscope_stitched'.
            y_label (str, optional): Name of the column of the Pandas dataframe
                that contains the Y coordinates of the points. Defaults to 
                'c_px_microscope_stitched'.
            z_label (str, optional): Name of the column of the datafile
                that contains the Z coordinates of the points. If None, the Z
                coordinate will default to the z value (see below).
                Defaults to None.
            gene_label (str, optional):  Name of the column of the Pandas 
                dataframe that contains the gene labels of the points. 
                Defaults to 'decoded_genes'.
            other_columns (list, optional): List with labels of other columns 
                that need to be loaded. Data will stored under "self.other"
                as Pandas Dataframe. Defaults to [].
            exclude_genes (list, optional): List with genes to exclude from
                dataset. Defaults to None.
            z (float, optional): Z coordinate of the dataset. Defaults to zero.
            pixel_size (str, optional): Unit size of the data. Is used to 
                convert data to micrometer scale. Uses Pint's UnitRegistry.
                Example: "0.1 micrometer", means that each pixel or unit has 
                a real size of 0.1 micrometer, and thus the data will be
                multiplied by 0.1 to make micrometer the unit of the data.
                Defaults to "1 micrometer".
            x_offset (float, optional): Offset in X axis. Defaults to 0.
            y_offset (float, optional): Offset in Y axis. Defaults to 0.
            z_offset (float, optional): Offset in Z axis. Defaults to 0.
            polygon ([np.ndarray, list], optional): Array or list of numpy
                arrays with shape (X,2) to select points. If a single polygon 
                is given this is used for all datasets. If the polygon is
                changed the dataset needs to be re-parsed. If multiple regions 
                in a single dataset need to be selected, a single array 
                containing the points of all polygons can be passed as long as
                each one is closed (First and last point are identical). 
                Defaults to None.
            parse_num_threads (int, optional): Number of workers for opening
                and parsing the datafiles. Datafiles need to be loaded in 
                memory to be parsed, which could cause problems with RAM. Use
                less workers if this happends. Set to 1, to process the files 
                sequentially. 
        """
        #Parameters
        self.gene_label, self.x_label, self.y_label= gene_label,x_label,y_label
        self.verbose =verbose
        self.index=0
        self.cpu_count = cpu_count()
        self.ureg = UnitRegistry()
        self.unique_genes = unique_genes
        self.grid_layout = grid_layout
        self.columns_layout = columns_layout
        
        #Dask
        #self.cluster = LocalCluster()
        #self.client = Client(self.cluster)
        #logging.info(f'Dask dashboard link: {self.client.dashboard_link}')
        
        #Name and folders
        if not MultiDataset_name:
            MultiDataset_name = 'MultiDataset_' + strftime("%Y-%m-%d_%H-%M-%S")
        self.dataset_name = MultiDataset_name
        self.dataset_folder = data_folder
        self.FISHscale_data_folder = path.join(self.dataset_folder, 'MultiDataset', f'{self.dataset_name}_FISHscale_MultiData')
        makedirs(self.FISHscale_data_folder, exist_ok=True)
        
        #Load data
        if type(data) == list:
            self.load_Datasets(data)
        elif type(data) == str:
            if parse_num_threads == -1 or parse_num_threads > self.cpu_count:
                parse_num_threads = self.cpu_count
            self.load_from_files(data, x_label, y_label, z_label, z, gene_label, other_columns, unique_genes, exclude_genes,
                                 pixel_size, x_offset, y_offset, z_offset, polygon, select_valid, reparse, color_input, 
                                 parse_num_threads)
        else:
            raise Exception(f'Input for "data" not understood. Should be list with initiated Datasets or valid path to files.')
        
        #Handle units
        self.check_unit()

        #Handle colors
        self.auto_handle_color_dict(color_input)
        self.overwrite_color_dict()
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.datasets):
            self.index = 0
            raise StopIteration
            
        index = self.index
        self.index += 1
        pd = self.datasets[index]
        return pd

    def vp(self, *args):
            """Function to print output if verbose mode is True
            """
            if self.verbose:
                for arg in args:
                    logging.info('    ' + str(arg))
                    
    def check_gene_input(self, gene):
        """Check if gene is in dataset. If not give suggestion.
        """            
        if not gene in self.unique_genes:
            raise Exception(f'Given gene: "{gene}" can not be found in dataset. Did you maybe mean: {get_close_matches(gene, self.unique_genes, cutoff=0.4)}?')

    def load_from_files(self, 
        filepath: str, 
        x_label: str = 'r_px_microscope_stitched',
        y_label: str = 'c_px_microscope_stitched',
        z_label: str = None,
        z: float = 0,
        gene_label: str = 'decoded_genes',
        other_columns: Optional[list] = None,
        unique_genes: Optional[np.ndarray] = None,
        exclude_genes: list = None,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        polygon: Union[np.ndarray, list] = None,
        select_valid: Union[bool, str] = False,
        reparse: bool = False,
        color_input: dict = None,
        num_threads: int = -1):
        """Load files from folder.

        Output can be found in self.dataset.
        Which is a list of initiated Dataset objects.

        Args:
            filepath (str): folder to look for parquet files. Files will be 
                processed in alphanumerically sorted order.
            x_label (str, optional): Name of the column of the Pandas dataframe
                that contains the X coordinates of the points. Defaults to 
                'r_px_microscope_stitched'.
            y_label (str, optional): Name of the column of the Pandas dataframe
                that contains the Y coordinates of the points. Defaults to 
                'c_px_microscope_stitched'.
            z_label (str, optional): Name of the column of the datafile
                that contains the Z coordinates of the points. If None, the Z
                coordinate will default to the z value (see below).
                Defaults to None.
            z (float, optional): Z coordinate of the dataset. Defaults to zero.
            gene_label (str, optional):  Name of the column of the Pandas 
                dataframe that contains the gene labels of the points. 
                Defaults to 'decoded_genes'.
            other_columns (list, optional): List with labels of other columns 
                that need to be loaded. Data will stored under "self.other"
                as Pandas Dataframe. Defaults to None.
            unique_genes (np.ndarray, optional): Array with unique gene names.
                This can also be a selection of genes to load. After a
                selection is made data needs to be re-parsed to include all
                genes.                
                If not provided it will find the unique genes from the 
                gene_column. This is slow for > 10e6 rows. 
                Defaults to None.
            exclude_genes (list, optional): List with genes to exclude from
                dataset. Defaults to None.     
            pixel_size (str, optional): Unit size of the data. Is used to 
                convert data to micrometer scale. Uses Pint's UnitRegistry.
                Example: "0.1 micrometer", means that each pixel or unit has 
                a real size of 0.1 micrometer, and thus the data will be
                multiplied by 0.1 to make micrometer the unit of the data.
                Defaults to "1 micrometer".
            x_offset (float, optional): Offset in X axis. Defaults to 0.
            y_offset (float, optional): Offset in Y axis. Defaults to 0.
            z_offset (float, optional): Offset in Z axis. Defaults to 0.
            polygon ([np.ndarray, list], optional): Array or list of numpy
                arrays with shape (X,2) to select points. If a single polygon 
                is given this is used for all datasets. If the polygon is
                changed the dataset needs to be re-parsed. Defaults to None.
            select_valid ([bool, str], optional): If the datafile already 
                contains information which datapoints to include this can be
                used to trim the dataset. The column should contain a boolean
                or binary array where "True" or "1" means that the datapoint
                should be included. 
                A string can be passed with the column name to use. If True is
                passed it will look for the default column name "Valid".
            unique_genes (np.ndarray, optional): Array with unique genes for
                dataset. If not given can take some type to compute for large
                datasets.
            reparse (bool, optional): True if you want to reparse the data,
                if False, it will repeat the parsing. Parsing will apply the
                offset. Defaults to False.
            num_threads (int, optional): Number of workers for opening and
                parsing the datafiles. Datafiles need to be loaded in memory to
                be parsed, which could cause problems with RAM. Use less
                workers if this happends. Set to 1, to process the files 
                sequentially. 
        """      

        #Correct slashes in path
        #if os_name == 'nt': #I guess this would work
        #    if not filepath.endswith('\\'):
        #        filepath = filepath + '\\'
        #if os_name == 'posix':
        #    if not filepath.endswith('/'):
        #        filepath = filepath + '/'

        #Load data
        files = glob(path.join(filepath, '*.parquet'))
        if len(files) == 0:
            files = glob(path.join(filepath, '*.csv'))
        if len(files) == 0:
            raise Exception(f'No .parquet or .csv files found in {filepath}')
        files = sorted(files)
        
        n_files = len(files)
        if not isinstance(z, (list, np.ndarray)):
            z = [z] * n_files
        if not isinstance(x_offset, (list, np.ndarray)):
            x_offset = [x_offset*c for row in range(ceil(n_files/self.columns_layout)) for c in range(self.columns_layout)]
            self.x_offsets = x_offset
        if not isinstance(y_offset, (list, np.ndarray)):
            y_offset_tmp = [y_offset for c in range(self.columns_layout)]
            y_offset = [np.array(y_offset_tmp)+(y*y_offset) for y in range(ceil(n_files/self.columns_layout))]
            y_offset = np.concatenate(y_offset).tolist()
            self.y_offsets = y_offset
            #y_offset = [y_offset] * n_files
        if not isinstance(z_offset, (list, np.ndarray)):
            z_offset = [z_offset] * n_files
            self.z_offsets = z_offset
        if not isinstance(pixel_size, (list, np.ndarray)):
            pixel_size = [pixel_size] * n_files
        if not isinstance(polygon, list):
            polygon = [polygon] * n_files
        
        #Get the unique genes
        if isinstance(unique_genes, np.ndarray):
            self.unique_genes = unique_genes
        
        #unique genes not provided
        else:
            if not isinstance(self.unique_genes, np.ndarray): 
                #Check if data has already been parsed to get the unique genes
                ug_success = False
                if self._check_parsed(files[0].split('.')[0] + '_FISHscale_Data')[0] and reparse == False:
                    try:
                        self.unique_genes = self._metadatafile_get_bypass(files[0], 'unique_genes')
                        ug_success = True
                    except Exception as e:
                        self.vp(f'Failed to fetch unique genes from metadata of previously parsed files. Recalculating. Exception: {e}')
                
                if not ug_success: 
                    open_f = self._open_data_function(files[0])
                    all_genes = open_f(files[0], [gene_label])
                    self.unique_genes = np.unique(all_genes)
        
        #Open the files with the option to do this in paralell.
        lazy_result = []

        for f, zz, pxs, xo, yo, zo, pol in tqdm(zip(files, z, pixel_size, x_offset, y_offset, z_offset, polygon)):
            lr = dask.delayed(Dataset) (f, x_label, y_label, z_label, zz, gene_label, other_columns, self.unique_genes, exclude_genes, 
                                        pxs, xo, yo, zo, pol, select_valid, reparse, color_input, verbose = self.verbose, 
                                        part_of_multidataset=True)
            lazy_result.append(lr)
        futures = dask.persist(*lazy_result, num_workers=1, num_threads = num_threads)
        self.datasets = dask.compute(*futures)
        self.datasets_names = [d.dataset_name for d in self.datasets]
        
    def load_Datasets(self, Dataset_list:list):
        """
        Load Datasets

        Args:
            Dataset_list (list): list of Dataset objects.
            overwrite_color_dict (bool, optional): If True sets the color_dicts
                of all individal Datasets to the MultiDataset color_dict.

        """        
        self.datasets = Dataset_list
        self.datasets_names = [d.dataset_name for d in self.datasets]

        #Set unique genes
        self.unique_genes = self.datasets[0].unique_genes
        self.check_unique_genes() 
        self.set_multidataset_true()

    def overwrite_color_dict(self) -> None:
        """Set the color_dict of the sub-datasets the same as the MultiDataset.
        """
        for d in self.datasets:
                d.color_dict = self.color_dict
                d._metadatafile_add({'color_dict': self.color_dict})

    def check_unit(self) -> None:
        """Check if all datasets have the same scale unit. 

        Raises:
            Exception: If `unit_scale` is not identical.
            Exception: If `unit_area` is not identical.
        """
        all_unit = [d.unit_scale for d in self.datasets]
        all_area = [d.area_scale for d in self.datasets]
        if not np.all([i == all_unit[0] for i in all_unit]):
            logging.info(all_unit)
            logging.info(all_area)
            raise Exception('Unit is not identical for all datasets.')
        #if not np.all(all_area == all_area[0]):
        if not np.all(i == all_area[0] for i in all_area):
            raise Exception('Area unit is not identical for all datasets.')
        self.unit_scale = all_unit[0]
        self.unit_area = all_area[0]

    def check_unique_genes(self) -> None:
        """Check if all datasets have same unique genes.

        Raises:
            Exception: Raises exception if not.
        """
        all_ug = [d.unique_genes for d in self.datasets]
        if not np.all(all_ug == all_ug[0]):
            raise Exception('Gene lists are not identical for all datasets.')
        
    def set_multidataset_true(self):
        """Set self.part_of_mutidataset to True.
        """
        for d in self.datasets:
                d.part_of_multidataset = True
                
    def order_datasets(self, orderby: str='z', order:list=None):
        """Order self.datasets.

        Args:
            orderby (str, optional): Sort by parameter:
                'z' : Sort by Z coordinate.
                'x' : Sort by width in X.
                'y' : Sort by width in Y.
                'name' : Sort by dataset name in alphanumerical order.                
                Defaults to 'z'.
            order (list, optional): List of indexes to use for sorting. If 
                "order" is provided, "orderby" is ignored. Defaults to None.
        """
        
        if order == None:
            if orderby == 'z':
                sorted = np.argsort([d.z for d in self.datasets])
            elif orderby == 'name':
                sorted = np.argsort([d.dataset_name for d in self.datasets])
            elif orderby == 'x':
                sorted = np.argsort([d.x_extent for d in self.datasets])
            elif orderby == 'y':
                sorted = np.argsort([d.y_extent for d in self.datasets])
            else:
                raise Exception(f'"Orderby" key not understood: {orderby}')
        else:
            sorted = order
        
        self.datasets = [self.datasets[i] for i in sorted]
        self.datasets_names = [self.datasets_names[i] for i in sorted]            
        

    def arange_grid_offset(self, orderby: str='order', ncol:int = None, spacing_fraction:float = 0.03):
        """Set offset of datasets so that they are in a XY grid side by side.

        Changes the X and Y coordinates so that all datasets are positioned in
        a grid for side by side plotting. Option to sort by various parameters.

        Use `reset_offset()` to bring centers back to (0,0).

        Args:
            orderby (str, optional): Sort by parameter:
                'order' : Sort by order in self.datasets
                'z' : Sort by Z coordinate.
                'x' : Sort by width in X.
                'y' : Sort by width in Y.
                'name' : Sort by dataset name in alphanumerical order.                
                    Defaults to 'order'.
            ncol (int, optional): Number of columns. If None is given, the 
                datasets will be put in a square grid. Defaults to None.
            specing_percentage (float, optional): Fraction of the largest 
                dataset that is used to make the spacing between datasets.
                Defults to 0.03. 

        Raises:
            Exception: If `orderby` is not properly defined.
        """

        max_x_extent = max([d.x_extent for d in self.datasets])
        max_y_extent = max([d.y_extent for d in self.datasets])
        max_x_extent = max_x_extent + (max_x_extent * spacing_fraction)
        max_y_extent = max_y_extent + (max_y_extent * spacing_fraction)
        n_datasets = len(self.datasets)
        if ncol == None:
            ncol = ceil(np.sqrt(n_datasets))
        nrow = ceil(n_datasets / ncol)
        x_extent = (ncol - 1) * max_x_extent
        y_extent = (nrow - 1) * max_y_extent
        x_spacing = np.linspace(-0.5 * x_extent, 0.5* x_extent, ncol)
        y_spacing = np.linspace(-0.5 * y_extent, 0.5* y_extent, nrow)
        x, y = np.meshgrid(x_spacing, y_spacing)
        x, y = x.ravel(), y.ravel()
        y = np.flip(y)

        if orderby == 'order':
            sorted = np.arange(0, len(self.datasets))
        elif orderby == 'z':
            sorted = np.argsort([d.z for d in self.datasets])
        elif orderby == 'name':
            sorted = np.argsort([d.dataset_name for d in self.datasets])
        elif orderby == 'x':
            sorted = np.argsort([d.x_extent for d in self.datasets])
        elif orderby == 'y':
            sorted = np.argsort([d.y_extent for d in self.datasets])
        else:
            raise Exception(f'"Orderby" key not understood: {orderby}')
        
        for i, s in enumerate(sorted):
            offset_x = x[i]
            offset_y = y[i]
            dataset_center = self.datasets[s].xyz_center
            offset_x = offset_x - dataset_center[0]
            offset_y = offset_y - dataset_center[1]
            self.datasets[s].offset_data_temp(offset_x, offset_y, 0)
        

    def reset_offset(self, z: bool=False) -> None: 
        """Reset the offset so that the center is at (0,0) or (0,0,0).

        Args:
            z (bool, optional): If True also the Z coordinate is reset to 0.
                Defaults to False.
        """

        for d in self.datasets:
            x_offset = -d.xyz_center[0]
            y_offset = -d.xyz_center[1]
            if z:
                z_offset = d.z
            else:
                z_offset = 0
            d.offset_data_temp(x_offset, y_offset, z_offset)
        
    def set_working_selection(self, level: Union[None, str] = None):
        """Set the working selection on which to work.
        
        If your dataset contains columns with boolean filters for certain 
        subsets of the data, you can select on which sub-selection you work 
        by setting the working_selection. This can for instance be different
        anatomical regions in your dataset.
        Setting the level to "None" resets the working selection. 

        Args:
            level (Union[None, str], optional): _description_. Defaults to None.
        """
        for d in self.datasets:
            d.set_working_selection(level)
            
    def reset_working_selection(self):
        """Reset the working selection to include all datapoints.
        """
        for d in self.datasets:
            d.set_working_selection(level = None)

    def visualize(
                self,
                remote=False,
                columns:list=[],
                color_dic=None,
                x=None,
                y=None,
                c={}):
            """
            Run open3d visualization on self.data
            Pass to columns a list of strings, each string will be the class of the dots to be colored by. example: color by gene

            Args:
                columns (list, optional): List of columns to be plotted with different colors by visualizer. Defaults to [].
                width (int, optional): Frame width. Defaults to 2000.
                height (int, optional): Frame height. Defaults to 2000.
            """
            from FISHscale.visualization.vis_macos import Window
            from open3d.visualization import gui
            if remote:
                import open3d as o3d
                o3d.visualization.webrtc_server.enable_webrtc()

            
            gui.Application.instance.initialize()
            if self.color_dict:
                color_dic = self.color_dict

            self.window = Window(self,
                            columns,
                            color_dic,
                            x_alt=x,
                            y_alt=y,
                            c_alt=c)
            
            gui.Application.instance.run()