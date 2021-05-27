from os import path, makedirs
from glob import glob
from dask import dataframe as dd
from typing import Optional, Union
import numpy as np
import itertools
import dask
from pyarrow.parquet import ParquetFile
from itertools import repeat
import fastparquet

import pandas as pd


def _gb_dump(x, name, folder_name:str, z: float):
    """Save groupby results as .parquet files.

    Args:
        x ([type]): groupby result.
        name ([type]): Dataset name.
        folder_name (str): Folder path.
        z (float): Z coordinate.
    """
    makedirs(folder_name, exist_ok=True)
    fn_out = path.join(folder_name, f'{name}_{x.name}.parquet')
    x['z'] = z
    x.to_parquet(fn_out)

class data_loader():
           
    def _check_parsed(self, filename: str, reparse: bool) -> bool:
        """Check if data has already been parsed

        Args:
            filename (str): path to data file.
            reparse (bool): True if data needs to be reparsed

        Returns:
            bool: True if folder with name <dataset_name>_FISHscale_Data is 
                present and contains at least one ".parquet" file.
        """
        if path.exists(self.FISHscale_data_folder):
            fn = path.join(self.FISHscale_data_folder, '*.parquet')
            file_list = glob(fn)
            if len(file_list) > 0:
                if not reparse:
                    self.vp(f'Found {len(file_list)} already parsed files. Skipping parsing.')
                return True
            else:
                return False          

        else:
            return False

    def _set_coordinate_properties(self, data):
        """Calculate properties of coordinates.

        Calculates XY min, max, extend and center of the points. 
        """
        self.x_min, self.y_min, self.x_max, self.y_max = data.x.min(), data.y.max(), data.x.max(), data.y.max()
        self.x_extend = self.x_max - self.x_min
        self.y_extend = self.y_max - self.y_min 
        self.xy_center = (self.x_max - 0.5*self.x_extend, self.y_max - 0.5*self.y_extend)

    def _gb_dump(self, data, name, folder_name:str):
        """Save groupby results as .parquet files.

        Args:
            x ([type]): groupby result.
            name ([type]): Dataset name.
            folder_name (str): Folder path.
            z (float): Z coordinate.
        """
        fn_out = path.join(folder_name, f'{name}_{data.name}.parquet')
        #write data
        data.to_parquet(fn_out)

    def load_data(self, filename: str, x_label: str, y_label: str, gene_label: str, other_columns: Optional[list], 
                  x_offset: float, y_offset: float, z_offset: float, pixel_size: str, reparse: bool = False) -> Union[
                      np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        
        
        #Check if data has already been parsed
        if not self._check_parsed(filename, reparse) or reparse:
            
            print('parsing data')
            
            #Data parsing
            if filename.endswith('.parquet') or filename.endswith('.csv'):
                
                # .parquet files
                if filename.endswith('.parquet'):
                    #Dask Dataframe
                    #open_f = lambda f, c: dd.read_parquet(f, columns = c)
                    #Pandas Dataframe, This turned out to be faster and more RAM effcient.
                    open_f = lambda f, c: pd.read_parquet(f, columns = c)
                # .csv files
                else:
                    open_f = lambda f, c: dd.read_csv(f, usecols = c)
                
                #Get columns to open
                col_to_open = [[gene_label, x_label, y_label], other_columns]
                col_to_open = list(itertools.chain.from_iterable(col_to_open))
                rename_col = dict(zip([gene_label, x_label, y_label], ['g', 'x', 'y']))
                
                #Read the data file
                data = open_f(filename, col_to_open)
                data = data.rename(columns = rename_col)
                
                #Offset data
                if x_offset !=0 and y_offset != 0:
                    data.loc[:, ['x', 'y']] += [x_offset, y_offset]
                #Add z_offset
                data['z'] = self.z + z_offset
                #Scale the data
                if pixel_size != 1:
                    data.loc[:, ['x', 'y']] = data.loc[:, ['x', 'y']] * pixel_size
                #Find data extend
                self._set_coordinate_properties(data)
                
                #Group the data by gene, rescale, offset and save
                data.groupby('g').apply(lambda x: self._gb_dump(x, self.dataset_name, self.FISHscale_data_folder))#, meta=('float64')).compute()
                
            else:
                raise IOError (f'Invalid file type: {filename}, should be in ".parquet" or ".csv" format.') 
            
        #Load Dask Dataframe from the parsed gene dataframes
        makedirs(self.FISHscale_data_folder, exist_ok=True)
        return dd.read_parquet(path.join(self.FISHscale_data_folder, '*.parquet'))
    
    def _get_gene_n_points(self):
        
        gene_n_points = {}
        for g in self.unique_genes:
            f = glob(path.join(self.FISHscale_data_folder, f'*{g}.parquet'))
            pfile = ParquetFile(f[0])
            gene_n_points[g] = pfile.metadata.num_rows
            
        return gene_n_points
    
    def transpose(self):
        """Transpose data. Switches X and Y.
        """
        rename_col = {'x': 'y', 'y': 'x'}
        self.df = self.df.rename(columns = rename_col)
        self.x_min, self.y_min, self.x_max, self.y_max = self.y_min, self.x_min, self.y_max, self.x_max
        self.x_extend = self.x_max - self.x_min
        self.y_extend = self.y_max - self.y_min 
        self.xy_center = (self.x_max - 0.5*self.x_extend, self.y_max - 0.5*self.y_extend)

    def get_rows(self,l:list):
        """
        Get rows by index

        Args:
            l (list): list of indexes to get from self.df

        Returns:
            dask.dataframe: filtered dask dataframe
        """        
        return self.df.map_partitions(lambda x: x[x.index.isin(l)])

    def add_dask_attribute(self,name:str,l:list):
        self.dask_attrs = self.dask_attrs.merge(pd.DataFrame({name:l}))
        
        
        
    
    
