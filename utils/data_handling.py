from os import path, makedirs
from glob import glob
from dask import dataframe as dd
from typing import Optional, Union
import numpy as np
import itertools

from pyarrow.parquet import ParquetFile

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
    
    def _check_parsed(self, filename: str) -> bool:
        """Check if data has already been parsed

        Args:
            filename (str): path to data file

        Returns:
            bool: True if folder with name <dataset_name>_FISHscale_Data is 
                present and contains at least one ".parquet" file.
        """
        if path.exists(self.FISHscale_data_folder):
            fn = path.join(self.FISHscale_data_folder, '*.parquet')
            file_list = glob(fn)
            if len(file_list) > 0:
                self.vp(f'Found {len(file_list)} already parsed files. Skipping parsing.')
                return True
            else:
                return False          
            
        else:
            return False
        
    
    def load_data(self, filename: str, x_label: str, y_label: str, gene_label: str, 
        other_columns: Optional[list], reparse: bool = False) -> Union[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        
        
        #Check if data has already been parsed
        if not self._check_parsed(filename) or reparse:
            
            print('parsing data')
            
            #Data parsing
            if filename.endswith('.parquet') or filename.endswith('.csv'):
                
                # .parquet files
                if filename.endswith('.parquet'):
                    open_f = lambda f, c: dd.read_parquet(f, columns = c)
                # .csv files
                else:
                    open_f = lambda f, c: dd.read_csv(f, usecols = c)
                
                #Get columns to open
                col_to_open = [[x_label, y_label, gene_label], other_columns]
                col_to_open = list(itertools.chain.from_iterable(col_to_open))
                rename_col = dict(zip([x_label, y_label, gene_label], ['x', 'y', 'g']))
                
                #Read the data file
                data = open_f(filename, col_to_open)
                data = data.rename(columns = rename_col)
                
                #Group the data by gene and save
                
                data.groupby('g').apply(lambda x: _gb_dump(x, self.dataset_name, self.FISHscale_data_folder, z=self.z),  meta=('float64')).compute()
                
            else:
                raise IOError (f'Invalid file type: {filename}, should be in ".parquet" or ".csv" format.') 
            
        #Load Dask Dataframe from the parsed gene dataframes
        return dd.read_parquet(path.join(self.FISHscale_data_folder, '*.parquet'))
    
    def _get_gene_n_points(self):
        
        gene_n_points = {}
        for g in self.unique_genes:
            f = glob(path.join(self.FISHscale_data_folder, f'*{g}.parquet'))
            pfile = ParquetFile(f[0])
            gene_n_points[g] = pfile.metadata.num_rows
            
        return gene_n_points
        
        
        
    
    
