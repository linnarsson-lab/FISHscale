from typing import Union, Any, Optional
import pandas as pd
from FISHscale.visualization import Window
#from FISHscale.utils.hex_bin import HexBin
from FISHscale.utils.hex_regionalization import regionalize
from FISHscale.utils.fast_iteration import Iteration
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

class Dataset(regionalize, Iteration):
    """
    Base Class for FISHscale, still under development

    Add methods to the class to run segmentation, make loom files, visualize (spatial localization, UMAP, TSNE).
    """

    def __init__(self,
        filename: str,
        x_label: str = 'r_px_microscope_stitched',
        y_label: str ='c_px_microscope_stitched',
        gene_label: str = 'below3Hdistance_genes',
        other_columns: Optional[list] = None,
        unique_genes: Optional[np.ndarray] = None,
        z: float = 0,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        apply_offset: bool = False,
        verbose: bool = False):
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
            verbose (bool, optional): If True prints additional output.

        """
        #Open file
        self.filename = filename
        self.dataset_name = self.filename.split('/')[-1].split('.')[0]
        self.x_label = x_label
        self.y_label = y_label
        self.gene_label = gene_label
        self.other_columns = other_columns
        self.x, self.y, self.gene, self.other = self.load_data(self.filename, self.x_label, self.y_label, self.gene_label, self.other_columns)
        self.z = z
        #Get gene list
        if not isinstance(unique_genes, np.ndarray):
            self.unique_genes = np.unique(self.gene)
        else:
            self.unique_genes = unique_genes
        
        #gene_dict = dict(zip(self.unique_genes, ))
        

        #Handle scale
        self.ureg = UnitRegistry()
        self.pixel_size = self.ureg(pixel_size)
        self.pixel_area = self.pixel_size ** 2
        self.x = self.x * self.pixel_size
        self.x = self.x.to('micrometer').magnitude
        self.y = self.y * self.pixel_size
        self.y = self.y.to('micrometer').magnitude
        self.unit_scale = self.ureg('1 micrometer')
        self.area_scale = self.unit_scale ** 2

        #Handle offset
        self.offset_data(x_offset, y_offset, z_offset, apply = apply_offset)

        #Verbosity
        self.verbose = verbose

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
        #Maybe add more file types?
        if filename.endswith('.parquet'):
            data = pd.read_parquet(filename)
        else:
            raise IOError (f'Invalid file type: {filename}, should be in ".parquet" format.') 

        x = data.loc[:, x_label].to_numpy()
        y = data.loc[:, y_label].to_numpy()
        genes = data.loc[:, gene_label].to_numpy()
        if other_columns != None:
            other = data.loc[:, other_columns]
        else:
            other = None

        return x, y, genes, other

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
            self.x += self.x_offset
            self.y += self.y_offset
            self.z += self.z_offset


    def visualize(self,columns=[],width=2000,height=2000,color_dic=None):
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

        window = Window(self.data,[self.gene_column]+columns,width,height,color_dic) 
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
        X = self.data.loc[:,[self.x,self.y]].values
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

        print('Assigning background dots whose cluster has more than {}'.format(cutoff))
        c =Counter(clustering.labels_)
        bg = [x for x in c if c[x] > cutoff]
        self.labels =  np.array([-1 if x in bg else x for x in clustering.labels_]) 
        print('Bockground molecules: {}'.format((self.labels == -1).sum()))
        #self.labels = np.array([-1 if (clustering.labels_ == x).sum() > 500 else x for x in clustering.labels_])
        self.data['DBscan'] = self.labels
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
    

class multi_Dataset():
    """Load multiple datasets as PandasDataset object.
    """


    def __init__(self, filepath: str,
        x: str = 'r_px_microscope_stitched',
        y: str ='c_px_microscope_stitched',
        gene_column: str = 'below3Hdistance_genes',
        other_columns: list = [],
        unique_genes: np.ndarray = None,
        pixel_size: str = '1 micrometer'):
        """Load multiple datasets as PandasDataset object. 

        Args:
            filepath (str): Path to files. Files must be pandas dataframes in
                .parquet format. 
            x (str, optional): [description]. Defaults to 
                'r_px_microscope_stitched'.
            y (str, optional): [description]. Defaults to 
                'c_px_microscope_stitched'.
            gene_column (str, optional): [description]. Defaults to 
                'below3Hdistance_genes'.
            other_columns (list, optional): [description]. Defaults to [].
            unique_genes (np.ndarray, optional): Array with unique gene names.
                If not provided it will find the unique genes from the 
                gene_column. This can however take some time for > 10e6 rows. 
                Defaults to None.
            pixel_size (str, optional): Unit size of the data. Is used to 
                convert data to micrometer scale. Uses Pint's UnitRegistry.
                Example: "0.1 micrometer", means that each pixel or unit has 
                a real size of 0.1 micrometer, and thus the data will be
                multiplied by 0.1 to make micrometer the unit of the data.
                Defaults to "1 micrometer".

        """
        self.datasets = self.load_data(filepath, x, y, gene_column, other_columns, unique_genes, pixel_size)

    def load_multi_data(self, filepath: str, x: str, y: str, gene_column: str, 
        other_columns: list, unique_genes: np.ndarray, pixel_size: str):

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
                #if self.verbose:
                #    print(f'Loading dataset ({i}/{len(files)})', end='\r')
                results.append(Dataset(f, x, y, gene_column, other_columns, unique_genes=unique_genes, pixel_size=pixel_size, verbose=False))
                #Get unique genes of first dataset if not defined
                if not isinstance(unique_genes, np.ndarray):
                    unique_genes = results[0].unique_genes
                pbar.update(1)
            
        return results
        







