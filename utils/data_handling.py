from os import path, makedirs,listdir
from glob import glob
import shutil
import re
from dask import dataframe as dd
from typing import Optional, Dict, Any, Callable
import numpy as np
import itertools
import pandas as pd
import pickle
from tqdm import tqdm
from FISHscale.utils.inside_polygon import is_inside_sm_parallel
from pyarrow.parquet import ParquetFile
from pyarrow import ArrowInvalid
import warnings

class DataLoader_base():
      
    def _open_data_function(self, filename: str) -> Callable:
        """Returns lambda function that can open a specific datafile.
        
        The returned function will take 2 arguments: filename and columns.
        filename = Full name of file.
        columns = List of columns to open
        
        Currently supports: .parquet and .csv

        Args:
            filename (str): Full name of file. 

        Returns:
            [Callable]: lambda fucntion to open target file.
        """
        # .parquet files
        if filename.endswith('.parquet'):
            #Dask Dataframe
            #open_f = lambda f, c: dd.read_parquet(f, columns = c)
            #Pandas Dataframe, This turned out to be faster and more RAM effcient.
            open_f = lambda f, c: pd.read_parquet(f, columns = c)
            
            def open_f(f, columns):
                try:
                    return pd.read_parquet(f, columns = columns)
                except ArrowInvalid as e:
                    p = ParquetFile(f)
                    raise Exception(f'Columns not found, choose from: {p.schema.names}. Error message: {e}')
                    
        # .csv files
        else:
            open_f = lambda f, c: dd.read_csv(f, usecols = c)
            
        return open_f
    
    def _metadatafile_make(self, data_dict: Dict):
        """Make a metadata file. This is a pickled dictionary.

        Args:
            data_dict (Dict): Dictionary with metadata

        Raises:
            Exception: If input is not a dictionary.
        """
        if not isinstance(data_dict, dict):
            raise Exception(f'Input should be a dictionary, not {type(data_dict)}.')
        
        file_name = path.join(self.FISHscale_data_folder, f'{self.dataset_name}_metadata.pkl')
        with open(file_name, 'wb') as pf:
            pickle.dump(data_dict, pf)
        
    def _metadatafile_read(self, file_name=None) -> Dict:
        """Read the full metadata file and return the dictionary.

        Raises:
            Exception: If metadata file was not found.
            Exception: If metadata file could not be opened.
            Exception: If the metadata is not a dictionary.

        Returns:
            Dict: Metadata dictionary
        """
        if file_name == None:
            file_name = path.join(self.FISHscale_data_folder, f'{self.dataset_name}_metadata.pkl')
        
        try:
            with open(file_name, 'rb') as pf:
                prop = pickle.load(pf)
        except FileNotFoundError as e:
            print('Metadata file was not found, please reparse.')
            raise e
        except Exception as e:
            print('Could not open metadata file please reparse or remake.')
            raise e
        
        if not isinstance(prop, dict):
            raise Exception(f'Metadata file should be a dictionary, not {type(prop)}.')
        
        return prop
        
    def _metadatafile_add(self, data_dict: Dict):
        """Add data to the metadata file.
        
        If key already exists it will be overwritten.

        Args:
            data_dict (Dict): Data to add.

        Raises:
            Exception: If "data_dict" is not a dictionary.
        """
        if not isinstance(data_dict, dict):
            raise Exception(f'Input should be a dictionary, not {type(data_dict)}.')
        
        existing_dict = self._metadatafile_read()
        merged_dict = {**existing_dict, **data_dict}
        
        self._metadatafile_make(merged_dict)
        
    def _metadatafile_get(self, item: str, verbose=False) -> Any:
        """Get a single item from the metadata.
        
        Retruns False when key is not present.

        Args:
            item (str): Key of the item to return
            verbose (bool, optional): If True, print the error messages if a
                key can not be found. Defaults to False.

        Returns:
            Any: The item if the key was present, or False if it was not.
        """
        
        existing_dict = self._metadatafile_read()
        try:
            result = existing_dict[item]
            return result
        except KeyError:
            if verbose:
                print(f'Key {item} not present in metadata.')
                print(f'Existing keys: {existing_dict.keys()}')
            return False
        
    def _metadatafile_get_bypass(self, file : str, item: str) -> Any:
        """Access metadata file of a parsed dataset wihout loading the dataset.

        Args:
            file (str): Full path to the original dataset file.
            item (str): Item to retrieve from the metadata

        Returns:
            Any: Resulting values fetched from the metadata file.
        """
        
        file_path = file.split('.')[0] + '_FISHscale_Data'
        file_name = path.splitext(path.basename(file))[0]
        metadata_file = path.join(file_path, (file_name + '_metadata.pkl'))
        
        #open file
        existing_dict = self._metadatafile_read(metadata_file)    
        
        return existing_dict[item]
    
    def _metatdata_set(self, obj: object, exclude: list=[]):
        """Transfer metadata from file to self.

        Args:
            obj (object): Object to add to. Should be `self`
            exclude (list, optional): List with keys to ignore. Use this for
                metadata items that are already handled elsewhere.
                Defaults to [].
        """
        
        metadata = self._metadatafile_read()
        
        for k,v in metadata.items():
            if k not in exclude:
                if hasattr(obj, k):
                    warnings.warn(f'Object already has an attribute: "{k}". Overwriting "{k}" with stored data from metadata file.')
                setattr(obj, k, v)
        
    def _dump_to_parquet(self, data, name, folder_name:str):
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
        
    def _check_parsed(self, folder: str) -> bool:
        """Check if data has already been parsed.

        Args:
            folder (str): folder with FISHscale data.

        Returns:
            [bool, int]: True if folder with name <dataset_name>_FISHscale_Data
                is present and contains at least one ".parquet" file. And the
                number of files found.
        """
        if path.exists(folder):
            fn = path.join(folder, '*.parquet')
            len_file_list = len(glob(fn))
            if len_file_list > 0:
                return True, len_file_list
            else:
                return False, 0

        else:
            return False, 0

    def get_dask_attrs_rows(self,l:list):
        """
        Get rows by index

        Args:
            l (list): list of indexes to get from self.df

        Returns:
            dask.dataframe: filtered dask dataframe
        """        
        return self.df.map_partitions(lambda x: x[x.index.isin(l)])

    def add_dask_attribute(self,
                            attribute_name:str,
                            l:np.array, 
                            include_genes=False):
        """
        [summary]

        Args:
            attribute_name (str): Name of the attribute that the dask dataframe will be partitioned by
            l: list or dictionary with columns and list of attributes to add.
            include_genes: whether the gene column will be reincluded .
            (the entire dataset will be saved twice). Defaults to False.
            
        """
        if type(l) != dict:
            d ={'x':self.df.x.compute(),
                'y':self.df.y.compute(),
                'z':self.df.z.compute(),                 
                attribute_name:l.astype('str')}
        else:
            d ={'x':self.df.x.compute(),
                'y':self.df.y.compute(),
                'z':self.df.z.compute()} 
            for x in l:
                d[x] = l[x]          
                        
        if include_genes:
            d['g'] = self.df.g.compute()

        da = pd.DataFrame(d)        
        if path.exists(path.join(self.dataset_folder, self.FISHscale_data_folder, 'attributes',attribute_name)):
            shutil.rmtree(path.join(self.dataset_folder, self.FISHscale_data_folder, 'attributes',attribute_name))

        makedirs(path.join(self.dataset_folder, self.FISHscale_data_folder, 'attributes',attribute_name),exist_ok=True)
        da.groupby(attribute_name).apply(lambda x: self._dump_to_parquet(x, self.dataset_name, self.FISHscale_data_folder+'/attributes/{}'.format(attribute_name)))#, meta=('float64')).compute()
        self.dask_attrs[attribute_name] = dd.read_parquet(path.join(self.dataset_folder, self.FISHscale_data_folder, 'attributes/{}'.format(attribute_name), '*.parquet'))   
        #self.dask_attrs.to_parquet(path.join(self.dataset_folder,self.FISHscale_data_folder,'attributes'))    

class DataLoader(DataLoader_base):
    
    def _coordinate_properties(self, data):
        """Calculate properties of coordinates.

        Calculates XY min, max, extent and center of the points. 
        """
        self.x_min, self.x_max = data.x.min(), data.x.max()
        self.y_min, self.y_max = data.y.min(), data.y.max()
        self.x_extent = self.x_max - self.x_min
        self.y_extent = self.y_max - self.y_min 
        self.xy_center = (self.x_max - 0.5*self.x_extent, self.y_max - 0.5*self.y_extent)
        
        prop = {'x_min': self.x_min,
                'x_max': self.x_max,
                'y_min': self.y_min,
                'y_max': self.y_max}
        self._metadatafile_make(prop)

    def _get_coordinate_properties(self):
        """Get the properties of coordinates.
        """
        #Load metadata
        prop = self._metadatafile_read()
        #Set properties
        self.x_min = prop['x_min']
        self.x_max = prop['x_max']
        self.y_min = prop['y_min']
        self.y_max = prop['y_max']
        self.x_extent = self.x_max - self.x_min
        self.y_extent = self.y_max - self.y_min 
        self.xy_center = (self.x_max - 0.5*self.x_extent, self.y_max - 0.5*self.y_extent)
        self.shape = prop['shape']
        
    def _numberstring_sort(self, l): 
        """Sort where full numbers are considered.""" 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return np.array(sorted(l, key = alphanum_key))
    
    def _exclude_genes(self, ug, eg):
        """Filter out excluded genes from unique gene array.
        """
        if type(eg) != type(None):
            ug = np.array([g for g in ug if g not in eg])
        return ug

    def load_data(self, filename: str, x_label: str, y_label: str, gene_label: str, other_columns: Optional[list], 
                  x_offset: float, y_offset: float, z_offset: float, pixel_size: str, unique_genes: Optional[np.ndarray],
                  exclude_genes: list = None, polygon: np.ndarray = None, reparse: bool = False) -> Any:             
        """Load data from data file.
        
        Opens a file containing XY coordinates of points with a gene label.
        The data will be parsed by grouping the data by the gene label and 
        storing the results in individual .parquet files for each gene. These
        files then get read as a single Dask Dataframe, to be memory efficient.
        The parsed data will be saved in a folder next to the original data,
        and once parsed, reopening the file will skip the parsing. If you want
        to explicity reparse, use the "reparse" option.
        
        Note: Parsing will save the offsets. To change the offset the data
            needs to be reparsed unfortunately. 
        Note: The original data file needs to fit into ram RAM.

        Args:
            filename (str): Path to the datafile to load. Should be in .parquet
                or .csv format.
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
            x_offset (float, optional): Offset in X axis.
            y_offset (float, optional): Offset in Y axis.
            z_offset (float, optional): Offset in Z axis.
            pixel_size (str, optional): Size of the pixels in micrometer.
            unique_genes (np.ndarray, optional): Array with unique genes for
                dataset. Genes present in data but not in 'unique_genes' will
                not be included. If not given, can take some type to compute for large
                datasets.
            exclude_genes (list, optional): List with genes to exclude from
                dataset. Defaults to None. 
            polygon (np.ndarray, optional): Array with shape (X,2) with closed
                polygon to select points.
            reparse (bool, optional): True if you want to reparse the data,
                if False, it will repeat the parsing. Parsing will apply the
                offset. Defaults to False.

        Raises:
            IOError: If file can not be opened.

        Returns:
            Dask Dataframe: With all data and partitioned by gene.
        """
        new_parse = False
        #Check if data has already been parsed
        already_parsed = self._check_parsed(filename.split('.')[0] + '_FISHscale_Data') 
        if not already_parsed[0] or reparse:
            if already_parsed[0] and not reparse:
                self.vp(f'Found {already_parsed[1]} already parsed files. Skipping parsing.')
            new_parse = True
            
            #Data parsing
            if filename.endswith(('.parquet', '.csv')):
                
                #Get function to open file
                open_f = self._open_data_function(filename)
                
                #Get columns to open              
                col_to_open = [[gene_label, x_label, y_label], other_columns]
                col_to_open = list(itertools.chain.from_iterable(col_to_open))
                rename_col = dict(zip([gene_label, x_label, y_label], ['g', 'x', 'y']))
                
                #Read the data file
                data = open_f(filename, col_to_open) 
                data = data.rename(columns = rename_col)
                
                #Offset data
                if x_offset !=0 or y_offset != 0:
                    data.loc[:, ['x', 'y']] += [x_offset, y_offset]
                    self.x_offset = 0
                    self.y_offset = 0
                    
                #Add z_offset
                self.z += z_offset
                data['z'] = self.z
                self.z_offset = 0
                
                #Scale the data
                if pixel_size != 1:
                    data.loc[:, ['x', 'y']] = data.loc[:, ['x', 'y']] * pixel_size
                
                #Find data extent and make metadata file
                self._coordinate_properties(data)

                #unique genes
                if not isinstance(unique_genes, (np.ndarray, list)):
                    ug = np.unique(data.g)
                    #Get the order the same as how Pandas would sort.
                    ug = self._numberstring_sort(ug)
                    #Make unique genes
                    ug = self._exclude_genes(ug, exclude_genes)
                    self.unique_genes = ug
                    #Select requested genes
                    data = data.loc[data.g.isin(self.unique_genes)]
                else:
                    ug = np.asarray(unique_genes)
                    #Get the order the same as how Pandas would sort.
                    ug = self._numberstring_sort(ug)
                    #Make unique genes
                    ug = self._exclude_genes(ug, exclude_genes)    
                    self.unique_genes = ug                  
                    #Select requested genes
                    data = data.loc[data.g.isin(self.unique_genes)]
                self._metadatafile_add({'unique_genes': self.unique_genes})    
                
                #Filter dots with polygon
                if type(polygon) != type(None):
                    filt = is_inside_sm_parallel(polygon, data.loc[:,['x', 'y']].to_numpy())
                    data = data.loc[filt,:]
                    print('adding to metadata')
                    self._metadatafile_add({'polygon': polygon})
                
                #Get data shape
                self.shape = data.shape
                self._metadatafile_add({'shape': self.shape})
                
                #Group the data by gene and save
                tqdm.pandas()
                data.groupby('g').progress_apply(lambda x: self._dump_to_parquet(x, self.dataset_name, self.FISHscale_data_folder))#, meta=('float64')).compute()
                if path.exists(path.join(self.dataset_folder, self.FISHscale_data_folder, 'attributes')):
                    shutil.rmtree(path.join(self.dataset_folder, self.FISHscale_data_folder, 'attributes'))
 
            else:
                raise IOError (f'Invalid file type: {filename}, should be in ".parquet" or ".csv" format.') 
        
        #Load Dask Dataframe from the parsed gene dataframes
        makedirs(self.FISHscale_data_folder, exist_ok=True)
        if isinstance(unique_genes, (np.ndarray, list)):
            #Make selected genes file list         
            p = path.join(self.FISHscale_data_folder, self.dataset_name)
            ug = self._exclude_genes(unique_genes, exclude_genes)
            filter_filelist = [f'{p}_{g}.parquet' for g in ug]

            #Load selected genes        
            self.df = dd.read_parquet(filter_filelist)
            self.shape = (self.df.shape[0].compute(), self.df.shape[1])
        else:
            #Load all genes
            self.df = dd.read_parquet(path.join(self.FISHscale_data_folder, '*.parquet'))

        if new_parse == False:
            #Get coordinate properties from metadata
            self._get_coordinate_properties()
            #Unique genes
            unique_genes_metadata = self._metadatafile_get('unique_genes')
            #Attributes
            p = path.join(self.dataset_folder, self.FISHscale_data_folder, 'attributes')
            self.dask_attrs = {x :dd.read_parquet(path.join(p,x,'*.parquet'), )  for x in listdir(p)}
            
            #Check if unique_genes are given by user
            if isinstance(unique_genes, (np.ndarray, list)):
                ug = self._exclude_genes(unique_genes, exclude_genes)
                self.unique_genes = np.asarray(self._numberstring_sort(ug))
                self._metadatafile_add({'unique_genes': self.unique_genes})
            #Check if unique genes could be found in metadata
            elif isinstance(unique_genes_metadata, (np.ndarray, list)): 
                self.unique_genes = self._numberstring_sort(unique_genes_metadata)
            #Calcualte the unique genes, slow
            else:
                self.unique_genes = self.df.g.drop_duplicates().compute().to_numpy()
                self._metadatafile_add({'unique_genes': self.unique_genes})
                
            #Handle all other metadata, (Excluding all attributes that are already handled somewhere else)
            self._metatdata_set(self, exclude=['unique_genes', 'x_min', 'x_max', 'y_min', 'y_max', 'shape', 'color_dict'])

        #Handle metadata
        else: 
            makedirs(path.join(self.dataset_folder, self.FISHscale_data_folder, 'attributes'), exist_ok=True)
            self.dask_attrs = {}
            '''self.dask_attrs = dd.from_pandas(pd.DataFrame(index=self.df.index), npartitions=self.df.npartitions, sort=False)
            for c in other_columns:
                self.add_dask_attribute(c, self.df[c])
            self.dask_attrs.to_parquet(path.join(self.dataset_folder, self.FISHscale_data_folder, 'attributes'))'''


    def _get_gene_n_points(self) -> Dict:
        """Get number of points per gene.
        
        Uses the metadata of the .parquet files for each gene.

        Returns:
            List: Number of points per gene in the order of 
                "self.unique_genes".
        """       
        gene_n_points = {}
        for g in self.unique_genes:
            f = glob(path.join(self.FISHscale_data_folder, f'*{g}.parquet'))
            pfile = ParquetFile(f[0])
            gene_n_points[g] = pfile.metadata.num_rows
            
        return gene_n_points
    
    def transpose(self):
        """Transpose data. Switches X and Y.
        
        This operation does NOT survive reloading the data.
        """
        rename_col = {'x': 'y', 'y': 'x'}
        self.df = self.df.rename(columns = rename_col)
        self.x_min, self.y_min, self.x_max, self.y_max = self.y_min, self.x_min, self.y_max, self.x_max
        self.x_extent = self.x_max - self.x_min
        self.y_extent = self.y_max - self.y_min 
        self.xy_center = (self.x_max - 0.5*self.x_extent, self.y_max - 0.5*self.y_extent)
    
    def flip_x(self):
        """Flips the X coordinates around the X center.
        
        This operation does NOT survive reloading the data.
        """
        self.df.x = -(self.df.x - self.xy_center[0]) + self.xy_center[0]
    
    def flip_y(self):
        """Flips the Y coordinates around the Y center.
        
        This operation does NOT survive reloading the data.
        """
        self.df.y = -(self.df.y - self.xy_center[1]) + self.xy_center[1]


