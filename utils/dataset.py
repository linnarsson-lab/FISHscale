import re
from typing import Union, Any, Optional
import pandas as pd
from FISHscale import Window
#from FISHscale.utils.hex_bin import HexBin
from FISHscale.utils.hex_regionalization import Regionalize
from FISHscale.utils.fast_iteration import Iteration, MultiIteration
from FISHscale.utils.colors import ManyColors
from FISHscale.utils.gene_correlation import GeneCorr
from FISHscale.utils.spatial_metrics import SpatialMetrics
from FISHscale.visualization.gene_scatter import GeneScatter, MultiGeneScatter
from PyQt5 import QtWidgets
import sys
from datetime import datetime
from sklearn.cluster import DBSCAN, MiniBatchKMeans, Birch, SpectralClustering
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np
import loompy
from pint import UnitRegistry
import os
from glob import glob
from time import strftime
from math import ceil
from multiprocessing import cpu_count
import itertools
import tempfile
try:
    from pyarrow.parquet import ParquetFile
except ModuleNotFoundError as e:
    print(f'Please install "pyarrow" to load ".parquet" files. Without only .csv files are supported which are memory inefficient. Error: {e}')
from dask import dataframe as dd
import dask

class Dataset(Regionalize, Iteration, ManyColors, GeneCorr, GeneScatter, SpatialMetrics):
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
        apply_offset: bool = False,
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
            apply_offset (bool, optional): Offsets the coordinates of the 
                points with the provided x and y offsets. Defaults to False.
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

        #Open file
        self.filename = filename
        self.dataset_name = self.filename.split('/')[-1].split('.')[0]
        self.other_columns = other_columns
        self.df = self.load_data(self.filename, x_label, y_label, gene_label, self.other_columns)
        self.z = z

        #Get gene list
        if not isinstance(unique_genes, np.ndarray):
            self.unique_genes = self.df.g.drop_duplicates().compute().to_numpy()
        else:
            self.unique_genes = unique_genes

        #Handle scale
        self.ureg = UnitRegistry()
        self.pixel_size = self.ureg(pixel_size)
        self.pixel_size = self.pixel_size.to('micrometer')
        self.pixel_area = self.pixel_size ** 2
        self.df.x = self.df.x * self.pixel_size.magnitude
        self.df.y = self.df.y * self.pixel_size.magnitude
        self.unit_scale = self.ureg('1 micrometer')
        self.area_scale = self.unit_scale ** 2

        #Handle offset
        self._offset_flag = False
        self.offset_data(x_offset, y_offset, z_offset, apply = apply_offset)

        #Handle colors
        self.auto_handle_color_dict(color_input)
        #Verbosity
        self.verbose = verbose
        self.vp(f'Loaded: {self.dataset_name}')      

    def load_data(self, filename: str, x_label: str, y_label: str, gene_label: str, 
        other_columns: Optional[list]) -> Union[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """[summary]

        Args:
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

        Raises:
            IOError: When filname can not be opened

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, pd.DataFrame or None]: Arrays
                with X coordinates, Y coordinates, gene labels and other data.
                If no other data specified will retun None.

        """
        
        if filename.endswith('.parquet') or filename.endswith('.csv'):
            
            if filename.endswith('.parquet'):
                open_f = lambda f, c: dd.read_parquet(f, columns = c)
            else:
                open_f = lambda f, c: dd.read_csv(f, usecols = c)
            
            col_to_open = [[x_label, y_label, gene_label], other_columns]
            col_to_open = list(itertools.chain.from_iterable(col_to_open))
            rename_col = dict(zip([x_label, y_label, gene_label], ['x', 'y', 'g']))
            
            data = open_f(filename, col_to_open)
            data = data.rename(columns = rename_col)
            
        else:
            raise IOError (f'Invalid file type: {filename}, should be in ".parquet" or ".csv" format.') 
        
        return data
            
    def vp(self, *args):
        """Function to print output if verbose mode is True
        """
        if self.verbose:
            for arg in args:
                print('    ' + arg)
    
    def offset_data(self, x_offset: float, y_offset: float, z_offset: float, apply:bool = True):
        """Offset the data with the given offset values.

        Args:
            x_offset (float): Offset in X axis. 
            y_offset (float): Offset in Y axis.
            z_offset (float): Offset in Z axis.
            apply (bool, optional): If True applies the offset to the data. If
                False only the self.x_offset/y_offset/z_offset gets changed. 
                Defaults to True.

        """
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        if apply:
            self.df.x += self.x_offset
            self.df.y += self.y_offset
            self.z += self.z_offset
        self._set_coordinate_properties()
        self._offset_flag = True

    def _set_coordinate_properties(self):
        """Calculate properties of coordinates.

        Calculates XY min, max, extend and center of the points. 
        """
        self.x_min, self.y_min, self.x_max, self.y_max = dask.compute(self.df.x.min(), self.df.y.max(), self.df.x.max(), self.df.y.max())
        self.x_extend = self.x_max - self.x_min
        self.y_extend = self.y_max - self.y_min 
        self.xy_center = (self.x_max - 0.5*self.x_extend, self.y_max - 0.5*self.y_extend)

    def transpose(self):
        """Transpose data. Switches X and Y.
        """
        rename_col = {'x': 'y', 'y': 'x'}
        self.df = self.df.rename(columns = rename_col)
        self.x_min, self.y_min, self.x_max, self.y_max = self.y_min, self.x_min, self.y_max, self.x_max
        self.x_extend = self.x_max - self.x_min
        self.y_extend = self.y_max - self.y_min 
        self.xy_center = (self.x_max - 0.5*self.x_extend, self.y_max - 0.5*self.y_extend)
        self._offset_flag = True

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
        
    def make_loom(self,
        filename:str,
        cell_column:str,
        with_background:bool=False,
        save=True):
        """
        Will generate the gene expression matrix and save it as loom file
        """

        cell = {x[0]:x[1] for x in self.data.groupby(cell_column)}
        del cell[-1]
        genes = np.unique(self.data[self.gene_column])

        c0 = Counter({x:0 for x in genes})
        gene_cell = []

        with tqdm(total=self.labels.max()) as pbar:
            for x in cell:
                generow = cell[x][self.gene_column]
                #print(generow)
                c1 = Counter(generow)
                v = []
                for x in c0:  
                    if x in c1:
                        v.append(c1[x])
                    else:
                        v.append(0)
                gene_cell.append(v)
                pbar.update(1)
                # Updates in increments of 10 stops at 100

        print('Assembling Gene by Cell Matrix')     
        df = np.stack(gene_cell).T
        cells_id = np.arange(df.shape[1])
        print('Cells_id_shape',cells_id.shape,df.shape)
        if genes.tolist() == self.hex_binned.df.index.tolist():
            print(df.shape)
            print(self.hex_binned.df.values.shape)
            df = np.concatenate([df, self.hex_binned.df.values],axis=1)

        print(df.shape)
        y = df.sum(axis=0)
        data = pd.DataFrame(data=df,index=genes)

        print('Creating Loom File')
        grpcell  = self.data[self.data[cell_column] > -1].groupby(cell_column)
        centroids = np.array([[(cell[1][self.y].min()+ cell[1][self.y].max())/2,(cell[1][self.x].min()+ cell[1][self.x].max())/2] for cell in grpcell])

        if with_background:
            cells_id = np.concatenate([cells_id ,np.array([-x for x in range(1,self.hex_binned.df.shape[1] +1)])])
            print('cells_id',cells_id.shape)
            centroids = np.concatenate([centroids,self.hex_binned.coordinates_filt])
            print('centroids',centroids.shape)
        
        colattrs = {'cell':cells_id,'cell_xy':centroids}
        rowattrs = {'gene':data.index.values}
        
        if save:
            loompy.create(filename+datetime.now().strftime("%d-%m-%Y%H:%M:%S"), data.values, rowattrs, colattrs)
            print('Loom File Created')
    

class MultiDataset(ManyColors, MultiIteration, MultiGeneScatter):
    """Load multiple datasets as Dataset objects.
    """

    def __init__(self,
        data: Union[list, str],
        unique_genes: Optional[np.ndarray] = None,
        MultiDataset_name: Optional[str] = None,
        color_input: Optional[Union[str, dict]] = None,
        verbose: bool = False,
        n_cores: int = None,

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
        apply_offset: bool = False):
        """initiate PandasDataset

        Args:
            data (Union[list, str]): List with initiated Dataset objects, or 
                path to folder with files to load. Unique genes must be 
                identical for all Datasets.
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
            n_cores (int, optional): Number of cores to use by default for 
                parallel jobs. If `None` defaults to all cpu cores. Defaults to
                None.

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
            apply_offset (bool, optional): Offsets the coordinates of the 
                points with the provided x and y offsets. Defaults to False.
        """
        #Parameters
        self.gene_label, self.x_label, self.y_label= gene_label,x_label,y_label
        self.verbose =verbose
        self.index=0
        self.cpu_count = cpu_count()
        self.ureg = UnitRegistry()

        #Name
        if not MultiDataset_name:
            MultiDataset_name = 'MultiDataset_' + strftime("%Y-%m-%d_%H-%M-%S")
        self.dataset_name = MultiDataset_name

        #Load data
        self.unique_genes = unique_genes
        if type(data) == list:
            self.load_Datasets(data)
        elif type(data) == str:
            self.load_from_files(data, x_label, y_label, gene_label, other_columns, z, pixel_size, 
                                x_offset, y_offset, z_offset, apply_offset)
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
        apply_offset: bool = False):
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
            apply_offset (bool, optional): Offsets the coordinates of the 
                points with the provided x and y offsets. Defaults to False.
        """      

        #Correct slashes in path
        if os.name == 'nt': #I guess this would work
            if not filepath.endswith('\\'):
                filepath = filepath + '\\'
        if os.name == 'posix':
            if not filepath.endswith('/'):
                filepath = filepath + '/'

        #Load data
        files = glob(filepath + '*' + '.parquet')
        results = []
        with tqdm(total=len(files)) as pbar:
            for i, f in enumerate(files):
                results.append(Dataset(f, x_label, y_label, gene_label, other_columns, self.unique_genes, z, pixel_size, 
                                       x_offset, y_offset, z_offset, apply_offset, color_input=None, 
                                       verbose=self.verbose, part_of_multidataset = True))
                #Get unique genes of first dataset if not defined
                if not isinstance(self.unique_genes, np.ndarray):
                    self.unique_genes = results[0].unique_genes
                pbar.update(1)
            
        self.datasets = results

    def load_Datasets(self, Dataset_list:list):
        """
        Load Datasets

        Args:
            Dataset_list (list): list of Dataset objects.
            overwrite_color_dict (bool, optional): If True sets the color_dicts
                of all individaal Datasets to the MultiDataset color_dict.

        """        
        self.datasets = Dataset_list

        #Set unique genes
        self.unique_genes = self.datasets[0].unique_genes
        self.check_unique_genes()    

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
        #if not np.all(all_unit == all_unit[0]):
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

        
  








