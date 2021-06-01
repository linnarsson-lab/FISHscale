
from typing import Union, Optional
import pandas as pd
from FISHscale import Window
from FISHscale.utils.hex_regionalization import Regionalize
from FISHscale.utils.fast_iteration import Iteration, MultiIteration
from FISHscale.utils.colors import ManyColors
from FISHscale.utils.gene_correlation import GeneCorr
from FISHscale.utils.spatial_metrics import SpatialMetrics
from FISHscale.utils.normalization import Normalization
from FISHscale.visualization.gene_scatter import GeneScatter, MultiGeneScatter
from FISHscale.utils.data_handling import DataLoader, DataLoader_base
from PyQt5 import QtWidgets
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
from os import path, makedirs
from glob import glob
from time import strftime
from math import ceil
from multiprocessing import cpu_count
from dask import dataframe as dd
import dask
try:
    from pyarrow.parquet import ParquetFile
except ModuleNotFoundError as e:
    print(f'Please install "pyarrow" to load ".parquet" files. Without only .csv files are supported which are memory inefficient. Error: {e}')

class Dataset(Regionalize, Iteration, ManyColors, GeneCorr, GeneScatter, SpatialMetrics, DataLoader, Normalization):
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
        z: float = 0,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        reparse: bool = False,
        color_input: Optional[Union[str, dict]] = None,
        verbose: bool = False,
        part_of_multidataset: bool = False):
        """initiate PandasDataset

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
                If not provided it will find the unique genes from the 
                gene_column. This is slow for > 10e6 rows. 
                Defaults to None.
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
            reparse (bool, optional): True if you want to reparse the data,
                if False, it will repeat the parsing. Parsing will apply the
                offset. Defaults to False.
            color_input (Optional[st`r, dict], optional): If a filename is 
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
        self.z += z_offset
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.other_columns = other_columns

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
                       self.pixel_size.magnitude, unique_genes, reparse=reparse)

        self.dask_attrs = dd.from_pandas(pd.DataFrame(index=self.df.index),npartitions=self.df.npartitions,sort=False)

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
            self.df.x += self.x_offset
        if y_offset != 0:
            self.df.y += self.y_offset
        if z_offset != 0:
            self.df.z += self.z_offset
            
            
        self.x_min = prop['x_min']
        self.x_max = prop['x_max']
        self.y_min = prop['y_min']
        self.y_max = prop['y_max']
        self.x_extend = self.x_max - self.x_min
        self.y_extend = self.y_max - self.y_min 
        self.xy_center = (self.x_max - 0.5*self.x_extend, self.y_max - 0.5*self.y_extend)


    def visualize(self,columns:list=[],width=2000,height=2000,show_axis=False):
        """
        Run open3d visualization on self.data
        Pass to columns a list of strings, each string will be the class of the dots to be colored by. example: color by gene

        Args:
            columns (list, optional): List of columns to be plotted with different colors by visualizer. Defaults to [].
            width (int, optional): Frame width. Defaults to 2000.
            height (int, optional): Frame height. Defaults to 2000.
            color_dic ([type], optional): Dictionary of colors if None it will assign random colors. Defaults to None.
        """        

        QtWidgets.QApplication.setStyle('Fusion')
        App = QtWidgets.QApplication.instance()
        if App is None:
            App = QtWidgets.QApplication(sys.argv)
        else:
            print('QApplication instance already exists: %s' % str(App))

        window = Window(self,columns,width,height)
        App.exec_()
        App.quit()

    def DBsegment(self,eps=25,min_samples=5,cutoff=250):
        """
        Run DBscan segmentation on self.data, this will reassign a column on self.data with column_name

        Args:
            eps (int, optional): [description]. Defaults to 25.
            min_samples (int, optional): [description]. Defaults to 5.
            column_name (str, optional): [description]. Defaults to 'cell'.
            cutoff (int,optional): cells with number of dots above this threshold will be removed and -1 passed to background dots
        """        

        print('Running DBscan segmentation')
        X = np.array([self.x,self.y]).T
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

        print('Assigning background dots whose cluster has more than {}'.format(cutoff))
        c =Counter(clustering.labels_)
        bg = [x for x in c if c[x] > cutoff]
        self.labels =  np.array([-1 if x in bg else x for x in clustering.labels_]) 
        print('Background molecules: {}'.format((self.labels == -1).sum()))
        print('DBscan found {} clusters'.format(self.labels.max()))
        

class MultiDataset(ManyColors, MultiIteration, MultiGeneScatter, DataLoader_base):
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
        z: float = 0,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        reparse: bool = False,
        parse_num_workers: int = -1):
        """initiate PandasDataset

        Args:
            data (Union[list, str]): List with initiated Dataset objects, or 
                path to folder with files to load. Unique genes must be 
                identical for all Datasets.
            data_folder (Optional, str): Path to folder with data when "data" 
                is a list of already initiated Datasets. This folder will be 
                used to save MultiDataset metadata. If not provided it will
                save in the current working directory. Defaults to ''.                
            unique_genes (np.ndarray, optional): Array with unique gene names.
                If not provided it will find the unique genes from the 
                gene_column. This is slow for > 10e6 rows. 
                Defaults to None.
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
            
            parse_num_workers (int, optional): Number of workers for opening
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
            if parse_num_workers == -1 or parse_num_workers > self.cpu_count:
                parse_num_workers = self.cpu_count
            self.load_from_files(data, x_label, y_label, gene_label, other_columns, z, pixel_size, 
                                x_offset, y_offset, z_offset, reparse, num_workers=parse_num_workers)
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
        z: float = 0,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        reparse: bool = False,
        num_workers: int = -1):
        """Load files from folder.

        Output can be found in self.dataset.
        Which is a list of initiated Dataset objects.

        Args:
            filepath (str): folder to look for parquet files.
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
            unique_genes (np.ndarray, optional): Array with unique genes for
                dataset. If not given can take some type to compute for large
                datasets.
            reparse (bool, optional): True if you want to reparse the data,
                if False, it will repeat the parsing. Parsing will apply the
                offset. Defaults to False.
            num_workers (int, optional): Number of workers for opening and
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
        
        #Get unique genes of first dataset if not defined
        if not isinstance(self.unique_genes, np.ndarray):
            print('making unique')       
            open_f = self._open_data_function(files[0])
            all_genes = open_f(files[0], [gene_label])
            self.unique_genes = np.unique(all_genes)                
        
        #Open the files with the option to do this in paralell.
        lazy_result = []
        for f, zz, pxs, xo, yo, zo in zip(files, z, pixel_size, x_offset, y_offset, z_offset):
            lr = dask.delayed(Dataset) (f, x_label, y_label, gene_label, other_columns, self.unique_genes, zz, pxs,
                                    xo, yo, zo, reparse, verbose = self.verbose, part_of_multidataset=True)
            lazy_result.append(lr)
        futures = dask.persist(*lazy_result, num_workers=num_workers)
        self.datasets =  dask.compute(*futures)
        
    def load_Datasets(self, Dataset_list:list):
        """
        Load Datasets

        Args:
            Dataset_list (list): list of Dataset objects.
            overwrite_color_dict (bool, optional): If True sets the color_dicts
                of all individal Datasets to the MultiDataset color_dict.

        """        
        self.datasets = Dataset_list

        #Set unique genes
        self.unique_genes = self.datasets[0].unique_genes
        self.check_unique_genes() 
        self.set_multidataset_true()

    def overwrite_color_dict(self) -> None:
        """Set the color_dict of the sub-datasets the same as the MultiDataset.
        """
        for d in self.datasets:
                d.color_dict = self.color_dict

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

    def arange_grid_offset(self, orderby: str='z'):
        """Set offset of datasets so that they are in a XY grid side by side.

        Changes the X and Y coordinates so that all datasets are positioned in
        a grid for side by side plotting. Option to sort by various parameters.

        Use `reset_offset()` to bring centers back to (0,0).

        Args:
            orderby (str, optional): Sort by parameter:
                'z' : Sort by Z coordinate.
                'x' : Sort by width in X.
                'y' : Sort by width in Y.
                'name' : Sort by dataset name in alphanumerical order.                
                 Defaults to 'z'.

        Raises:
            Exception: If `orderby` is not properly defined.
        """

        max_x_extend = max([d.x_extend for d in self.datasets])
        max_y_extend = max([d.y_extend for d in self.datasets])
        n_datasets = len(self.datasets)
        grid_size = ceil(np.sqrt(n_datasets))
        x_extend = (grid_size - 1) * max_x_extend
        y_extend = (grid_size - 1) * max_y_extend
        x_spacing = np.linspace(-0.5 * x_extend, 0.5* x_extend, grid_size)
        y_spacing = np.linspace(-0.5 * y_extend, 0.5* y_extend, grid_size)
        x, y = np.meshgrid(x_spacing, y_spacing)
        x, y = x.ravel(), y.ravel()

        if orderby == 'z':
            sorted = np.argsort([d.z for d in self.datasets])
        elif orderby == 'name':
            sorted = np.argsort([d.dataset_name for d in self.datasets])
        elif orderby == 'x':
            sorted = np.argsort([d.x_extend for d in self.datasets])
        elif orderby == 'y':
            sorted = np.argsort([d.y_extend for d in self.datasets])
        else:
            raise Exception(f'"Orderby" key not understood: {orderby}')

        for i, s in enumerate(sorted):
            offset_x = x[i]
            offset_y = y[i]
            dataset_center = self.datasets[s].xy_center
            offset_x = offset_x - dataset_center[0]
            offset_y = offset_y - dataset_center[1]
            self.datasets[s].offset_data(offset_x, offset_y, 0, apply=True)

    def reset_offset(self, z: bool=False) -> None: 
        """Reset the offset so that the center is at (0,0) or (0,0,0).

        Args:
            z (bool, optional): If True also the Z coordinate is reset to 0.
                Defaults to False.
        """

        for d in self.datasets:
            center_x = -d.xy_center[0]
            center_y = -d.xy_center[1]
            if z:
                offset_z = -d.z
            else:
                offset_z = 0
            d.offset_data(center_x, center_y, offset_z, apply=True)
        
    def visualize(self,columns:list=[],width=2000,height=2000,show_axis=False,color_dic=None):
        """
        Run open3d visualization on self.data
        Pass to columns a list of strings, each string will be the class of the dots to be colored by. example: color by gene

        Args:
            columns (list, optional): List of columns to be plotted with different colors by visualizer. Defaults to [].
            width (int, optional): Frame width. Defaults to 2000.
            height (int, optional): Frame height. Defaults to 2000.
        """        

        QtWidgets.QApplication.setStyle('Fusion')
        App = QtWidgets.QApplication.instance()
        if App is None:
            App = QtWidgets.QApplication(sys.argv)
        else:
            print('QApplication instance already exists: %s' % str(App))
        if self.color_dict:
            color_dic = self.color_dict
        window = Window(self,columns,width,height,color_dic) 
        
        App.exec_()
        App.quit()