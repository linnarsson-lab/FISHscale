from multiprocessing import cpu_count
from os import path, makedirs, environ
environ['NUMEXPR_MAX_THREADS'] = str(cpu_count())
from typing import Union, Optional
import pandas as pd
try:
    from FISHscale.visualization.primitiveVis_open3dv2 import Window
except ModuleNotFoundError as e:
    print(f'Please install "PyQt5" for data visualization. {e}') 
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
from FISHscale.utils.clustering import Clustering
from FISHscale.utils.bonefight import BoneFight, BoneFightMulti
from FISHscale.utils.regionalization_multi import RegionalizeMulti
from FISHscale.utils.decomposition import Decomposition
from FISHscale.spatial.boundaries import Boundaries, Boundaries_Multi
from FISHscale.spatial.gene_order import Gene_order
from FISHscale.segmentation.cellpose import Cellpose
from FISHscale.utils.regionalization_gradient import Regionalization_Gradient, Regionalization_Gradient_Multi
import sys
from datetime import datetime
from sklearn.cluster import DBSCAN
import pandas as pd
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
    print(f'Please install "pyarrow" to load ".parquet" files. Without only .csv files are supported which are memory inefficient. Error: {e}')
from tqdm import tqdm

class Dataset(Regionalize, Iteration, ManyColors, GeneCorr, GeneScatter, AttributeScatter, SpatialMetrics, DataLoader, Normalization, 
              Density1D, Clustering, BoneFight, Decomposition, Boundaries, Gene_order, Cellpose, 
              Regionalization_Gradient):
    """
    Base Class for FISHscale, still under development

    Add methods to the class to run segmentation, make loom files, visualize (spatial localization, UMAP, TSNE).
    """

    def __init__(self,
        filename: str,
        x_label: str = 'r_px_microscope_stitched',
        y_label: str ='c_px_microscope_stitched',
        gene_label: str = 'below3Hdistance_genes',
        other_columns: Optional[list] = [],
        unique_genes: Optional[np.ndarray] = None,
        exclude_genes: list = None,
        z: float = 0,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        polygon: np.ndarray = None,
        reparse: bool = False,
        color_input: Optional[Union[str, dict]] = None,
        verbose: bool = False,
        part_of_multidataset: bool = False):
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
            gene_label (str, optional):  Name of the column of the Pandas 
                dataframe that contains the gene labels of the points. 
                Defaults to 'below3Hdistance_genes'.
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
            polygon (np.ndarray, optional): A numpy array with shape (X,2) with
                a polygon that can be used to select points. If the polygon is
                changed the dataset needs to be re-parsed. If multiple regions 
                need to be selected, a single array containing the points of
                all polygons can be passed as long as each one is closed (First
                and last point are identical). Defaults to None.
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
        if not isinstance(other_columns, list):
            other_columns = [other_columns]
        self.other_columns = other_columns
        
        #Dask
        #if not self.part_of_multidataset:
        #    self.cluster = LocalCluster()
        #    self.client = Client(self.cluster)
        #    print(f'Dask dashboard link: {self.client.dashboard_link}')

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
        
        #Load data
        self.load_data(self.filename, x_label, y_label, gene_label, self.other_columns, x_offset, y_offset, z_offset, 
                       self.pixel_size.magnitude, unique_genes, exclude_genes, self.polygon, reparse=reparse)

        #Gene metadata
        self.gene_index = dict(zip(self.unique_genes, range(self.unique_genes.shape[0])))
        self.gene_n_points = self._get_gene_n_points()

        #Handle colors
        self.auto_handle_color_dict(color_input)
        
        #Verbosity
        self.verbose = verbose
        self.vp(f'Loaded: {self.dataset_name}')
            
    def vp(self, *args):
        """Function to print output if verbose mode is True
        """
        if self.verbose:
            for arg in args:
                print('    ' + arg)
                
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

        self.x_extent = self.x_max - self.x_min
        self.y_extent = self.y_max - self.y_min 
        self.xy_center = (self.x_max - 0.5*self.x_extent, self.y_max - 0.5*self.y_extent)

    def visualize(self,
                columns:list=[],
                width=2000,
                height=2000,
                show_axis=False,
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
        if self.color_dict:
            color_dic = self.color_dict

        window = Window(self,
                        columns,
                        width,
                        height,
                        color_dic,
                        x_alt=x,
                        y_alt=y,
                        c_alt=c) 
        
    def DBsegment(self,
                    label_column,
                    eps=50,
                    min_samples=10,
                    cutoff=4,
                    save_to=None):
        from tqdm import trange
        """
        Run DBscan segmentation on self.data, this will reassign a column on self.data with column_name

        Args:
            eps (int, optional): [description]. Defaults to 25.
            min_samples (int, optional): [description]. Defaults to 5.
            column_name (str, optional): [description]. Defaults to 'cell'.
            cutoff (int,optional): cells with number of dots above this threshold will be removed and -1 passed to background dots
        """        
        def segmentation(partition):
            cl_molecules_xy = partition.loc[:,['x','y']].values
            segmentation = DBSCAN(eps,min_samples=min_samples).fit(cl_molecules_xy)
            return segmentation.labels_

        def get_counts(cell_i):
            cell_i,dblabel, centroid = cell_i[1], cell_i[0],(cell_i[1].x.mean(),cell_i[1].y.mean())
            if dblabel != -1:
                cell_i_g = cell_i['g']
                centroid = (cell_i.x.mean(),cell_i.y.mean())
                gene,cell =  np.unique(cell_i_g,return_counts=True)
                d = pd.DataFrame({dblabel:cell},index=gene)
                g= pd.DataFrame(index=self.unique_genes)
                data = pd.concat([g,d],join='outer',axis=1).fillna(0)
                return data, dblabel, centroid

        def get_cells(partition):
            cl_molecules_xy = partition.loc[:,['x','y','g','DBscan',label_column]]
            clr= cl_molecules_xy.groupby('DBscan')#.applymap(get_counts)
            dblabel, centroids, data = [],[],[]
            try:
                cl = cl_molecules_xy[label_column].values[0]
                for cell in clr:
                    try:
                        d, label, centroid = get_counts(cell)
                        dblabel.append(label)
                        centroids.append(centroid)
                        data.append(d)
                    except:
                        pass
                data = pd.concat(data,axis=1)
                return data, dblabel, centroids
            except:
                return None, [], []

        def gene_by_cell_loom(dask_attrs):
            matrices, labels, centroids, clusters = [],[],[], []
            for p in trange(self.dask_attrs[label_column].npartitions):
                matrix, label, centroid = get_cells(dask_attrs.partitions[p].compute())
                if type(matrix) != type(None):
                    matrices.append(matrix)

                try:
                    clusters += [self.dask_attrs[label_column].partitions[p][label_column].values.compute()[0]]*len(label)
                except:
                    pass

                labels += label
                centroids += centroid
            matrices = pd.concat(matrices,axis=1)

            if type(save_to) == type(None):
                file = path.join(self.dataset_folder,self.filename.split('.')[0]+'_DBcells.loom')
            else:
                file = path.join(save_to+'DBcells.loom')
            row_attrs = {'Gene':matrices.index.values}
            col_attrs = {'DBlabel':matrices.columns.values, 'Centroid':centroids, label_column:clusters}
            loompy.create(file,matrices.values,row_attrs,col_attrs)

        print('Running DBscan by: {}'.format(label_column))
        r = self.dask_attrs[label_column].groupby(label_column).apply(segmentation, meta=object).compute()
        result = [r[self.dask_attrs[label_column].partitions[x][label_column].compute().values[0]] for x in range(self.dask_attrs[label_column].npartitions)]
        self.dask_attrs[label_column] = self.dask_attrs[label_column].merge(pd.DataFrame(np.concatenate(result),index=self.dask_attrs[label_column].index,columns=['DBscan']))
        print('DBscan results added to dask attributes. Generating gene by cell matrix as loom file.')
        gene_by_cell_loom(self.dask_attrs[label_column])
        
                    

class MultiDataset(ManyColors, MultiIteration, MultiGeneScatter, DataLoader_base, Normalization, RegionalizeMulti,
                   Decomposition, BoneFightMulti, Regionalization_Gradient_Multi, Boundaries_Multi):
    """Load multiple datasets as Dataset objects.
    """

    def __init__(self,
        data: Union[list, str],
        data_folder: str = '',
        unique_genes: Optional[np.ndarray] = None,
        MultiDataset_name: Optional[str] = None,
        color_input: Optional[Union[str, dict]] = None,
        verbose: bool = False,

        #If loading from files define:
        x_label: str = 'r_px_microscope_stitched',
        y_label: str ='c_px_microscope_stitched',
        gene_label: str = 'below3Hdistance_genes',
        other_columns: Optional[list] = [],
        exclude_genes: list = None,
        z: float = 0,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        polygon: Union[np.ndarray, list] = None,
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
            gene_label (str, optional):  Name of the column of the Pandas 
                dataframe that contains the gene labels of the points. 
                Defaults to 'below3Hdistance_genes'.
            other_columns (list, optional): List with labels of other columns 
                that need to be loaded. Data will stored under "self.other"
                as Pandas Dataframe. Defaults to None.
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
        
        #Dask
        #self.cluster = LocalCluster()
        #self.client = Client(self.cluster)
        #print(f'Dask dashboard link: {self.client.dashboard_link}')
        
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
            self.load_from_files(data, x_label, y_label, gene_label, other_columns, unique_genes, exclude_genes, z, 
                                 pixel_size, x_offset, y_offset, z_offset, polygon, reparse, color_input, 
                                 num_threads=parse_num_threads)
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
                    print('    ' + arg)

    def load_from_files(self, 
        filepath: str, 
        x_label: str = 'r_px_microscope_stitched',
        y_label: str ='c_px_microscope_stitched',
        gene_label: str = 'below3Hdistance_genes',
        other_columns: Optional[list] = None,
        unique_genes: Optional[np.ndarray] = None,
        exclude_genes: list = None,
        z: float = 0,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        polygon: Union[np.ndarray, list] = None,
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
            gene_label (str, optional):  Name of the column of the Pandas 
                dataframe that contains the gene labels of the points. 
                Defaults to 'below3Hdistance_genes'.
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
                changed the dataset needs to be re-parsed. Defaults to None.
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
        if os_name == 'nt': #I guess this would work
            if not filepath.endswith('\\'):
                filepath = filepath + '\\'
        if os_name == 'posix':
            if not filepath.endswith('/'):
                filepath = filepath + '/'

        #Load data
        files = glob(filepath + '*' + '.parquet')
        if len(files) == 0:
            files = glob(filepath + '*' + '.csv')
        if len(files) == 0:
            raise Exception(f'No .parquet or .csv files found in {filepath}')
        files = sorted(files)
        
        n_files = len(files)
        if not isinstance(z, (list, np.ndarray)):
            z = [z] * n_files
        if not isinstance(x_offset, (list, np.ndarray)):
            x_offset = [x_offset] * n_files
        if not isinstance(y_offset, (list, np.ndarray)):
            y_offset = [y_offset] * n_files
        if not isinstance(z_offset, (list, np.ndarray)):
            z_offset = [z_offset] * n_files
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
            lr = dask.delayed(Dataset) (f, x_label, y_label, gene_label, other_columns, self.unique_genes, exclude_genes, 
                                        zz, pxs, xo, yo, zo, pol, reparse, color_input, verbose = self.verbose, 
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
            print(all_unit)
            print(all_area)
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
        

    def arange_grid_offset(self, orderby: str='order'):
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

        Raises:
            Exception: If `orderby` is not properly defined.
        """

        max_x_extent = max([d.x_extent for d in self.datasets])
        max_y_extent = max([d.y_extent for d in self.datasets])
        n_datasets = len(self.datasets)
        grid_size = ceil(np.sqrt(n_datasets))
        x_extent = (grid_size - 1) * max_x_extent
        y_extent = (grid_size - 1) * max_y_extent
        x_spacing = np.linspace(-0.5 * x_extent, 0.5* x_extent, grid_size)
        y_spacing = np.linspace(-0.5 * y_extent, 0.5* y_extent, grid_size)
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
            dataset_center = self.datasets[s].xy_center
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
            x_offset = -d.xy_center[0]
            y_offset = -d.xy_center[1]
            if z:
                z_offset = d.z
            else:
                z_offset = 0
            d.offset_data_temp(x_offset, y_offset, z_offset)
        

    def visualize(self,
                columns:list=[],
                width=2000,
                height=2000,
                show_axis=False,
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

        if self.color_dict:
            color_dic = self.color_dict

        window = Window(self,
                        columns,
                        width,
                        height,
                        color_dic,
                        x_alt=x,
                        y_alt=y,
                        c_alt=c) 
        
        self.App.exec_()
        