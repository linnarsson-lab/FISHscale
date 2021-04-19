import pandas as pd
from FISHscale import Window
#from FISHscale.utils.hex_bin import HexBin
from FISHscale.utils.hex_regionalization import regionalize
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

class PandasDataset(regionalize):
    """
    Base Class for FISHscale, still under development

    Add methods to the class to run segmentation, make loom files, visualize (spatial localization, UMAP, TSNE).
    """

    def __init__(self,
        filename: str,
        x: str = 'r_px_microscope_stitched',
        y: str ='c_px_microscope_stitched',
        gene_column: str = 'below3Hdistance_genes',
        other_columns: list = [],
        unique_genes: np.ndarray = None,
        pixel_size: str = '1 micrometer',
        x_offset: float = 0,
        y_offset: float = 0,
        z_offset: float = 0,
        apply_offset: bool = False,
        verbose: bool = False):
        """initiate PandasDataset

        Args:
            filename (str): [description]
            x (str, optional): [description]. Defaults to 'r_px_microscope_stitched'.
            y (str, optional): [description]. Defaults to 'c_px_microscope_stitched'.
            gene_column (str, optional): [description]. Defaults to 'below3Hdistance_genes'.
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
        self.x, self.y = x, y
        self.data = pd.read_parquet(filename) #maybe make a data loading function for other formats? 
        self.gene_column = gene_column
        self.other_columns = other_columns

        #Get gene list
        if not isinstance(unique_genes, np.ndarray):
            self.unique_genes = np.unique(self.data[self.gene_column])
        else:
            self.unique_genes = unique_genes

        #Handle scale
        self.ureg = UnitRegistry()
        self.pixel_size = self.ureg(pixel_size)
        self.pixel_area = self.pixel_size ** 2
        scale_corrected = self.data.loc[:, [self.x, self.y]].to_numpy() * self.pixel_size
        scale_corrected = scale_corrected.to('micrometer')
        self.data.loc[:, [self.x, self.y]] = scale_corrected.magnitude
        self.unit_scale = self.ureg('1 micrometer')
        self.area_scale = self.unit_scale ** 2

        #Handle offset
        self.offset_data(x_offset, y_offset, z_offset, apply = apply_offset)

        #Verbosity
        self.verbose = verbose


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
            y_offset (float): Offset in X axis.
            z_offset (float): Offset in X axis.
            apply (bool, optional): If True applies the offset to the data. If
                False only the self.x/y/z_offset gets changed. 
                Defaults to True.

        """
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        if apply:
            self.data = self.data + np.array([self.x_offset, self.y_offset])


    def visualize(self,columns=[],width=2000,height=2000,show_axis=False,color_dic=None):
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

        window = Window(self,[self.gene_column]+columns,width,height,color_dic) 
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
        
    def make_molecules_df(self):
        molecules_df = []
        genes = np.unique(self.data[self.gene_column])
        for g in self.data[self.gene_column]:
            e = np.where(genes != g, np.zeros(genes.shape[0]),1)
            molecules_df.append(e)
        self.molecules_df = np.stack(molecules_df).T

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
    

class multi_dataset():
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
        self.index=0
        self.datasets = self.load_data(filepath, x, y, gene_column, other_columns, unique_genes, pixel_size)
        self.x,self.y,self.gene_column = x,y,gene_column

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

    def load_data(self, filepath: str, x: str, y: str, gene_column: str, 
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
                results.append(PandasDataset(f, x, y, gene_column, other_columns, unique_genes=unique_genes, pixel_size=pixel_size, z_offset=1000*i,verbose=False))
                #Get unique genes of first dataset if not defined
                if not isinstance(unique_genes, np.ndarray):
                    unique_genes = results[0].unique_genes
                pbar.update(1)
            
        return results
        
    def visualize(self,columns=[],width=2000,height=2000,show_axis=False,color_dic=None):
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

        window = Window(self,[self.gene_column]+columns,width,height,color_dic) 
        App.exec_()
        App.quit()








