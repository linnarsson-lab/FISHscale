import pandas as pd
import numpy as np
from typing import Generator, Tuple
from functools import lru_cache

class Iteration:

    def get_gene(self, gene: str, include_z = False):
        """Get the xyz coordinates of points of a queried gene.
        
        This causes the data to get loaded in RAM.

        Args:
            gene (str): Name of gene.

        Returns:
            [pd.DataFrame]: Pandas Dataframe with coordinates.
        """
        gene_i= self.gene_index[gene]
        if include_z:
            columns = ['x', 'y', 'z']
        else:
            columns = ['x', 'y']
        
        return self.df.get_partition(gene_i).loc[:, columns].compute()
    
    def get_gene_sample(self, gene: str, include_z = False, frac: float=0.1, minimum: int=None, 
                        random_state: int=None):
        """Get the xyz coordinates of a sample of points of a queried gene.
        
        This causes the data to get loaded in RAM.

        Args:
            gene (str): Name of gene.
            frac (float, optional): Fraction of the points to load. 
                Defaults to 0.1 which is 10% of the data.
            minimum (int, optional): If minimum is given the fraction will be 
                adapted to return at least the minimum number of points. if 
                there are less points than the minimum it returns all. 
                Defaults to None.
            random_state (int, optional): Random state for the sampling to 
                return the same points over multiple iterations.
                Defaults to None.

        Returns:
            [type]: [description]
        """
        if minimum != None:
            n_points = self.gene_n_points[gene]
            if n_points < minimum:
                frac = 1
            elif frac * n_points < minimum:
                frac = minimum / n_points
        
        gene_i= self.gene_index[gene]
        
        if include_z:
            columns = ['x', 'y', 'z']
        else:
            columns = ['x', 'y']
            
        return self.df.get_partition(gene_i).loc[:, columns].sample(frac=frac, random_state=random_state).compute()
    
    
    
    

    
    
    
    
class _old_stuff:
    
    def _group_by(self, by='g'):
        return self.df.groupby(by).apply(lambda g: np.array([g.x, g.y]).T, meta=('float64')).compute()
    
    
    def make_gene_coordinates(self, save_z = False) -> None:
        """Make a dictionary with point coordinates for each gene.

        Output will be in self.gene_coordinates. Output will be cached so that
        this function can be called many times but the calculation is only
        performed the first time.

        """
        if self._offset_flag == True:
                   
            if save_z:
                self.gene_coordinates = {g: np.column_stack((xy, np.array([self.z_offset]*xy.shape[0]))).astype('float64') for g, xy in self._group_by().to_dict().items()}
                
            else:
                self.gene_coordinates = self._group_by().to_dict()
                
            self._offset_flag = False
            
        else:
            print('Gene coodinates already calculated. skipping')
            pass
    

    def xy_groupby_gene_generator(self, gene_order: np.ndarray = None) -> Generator[Tuple[str, np.ndarray, np.ndarray], None, None]:
        """Generator function that groups XY coordinates by gene.

        Uses the Pandas groupby() function for speed on unsorted numpy arrays.

        Yields:
            Iterator[Union[str, np.ndarray, np.ndarray]]: Gene name, X 
                coordinates, Y coordinates.

        """
        df = pd.DataFrame(data = np.column_stack((self.x, self.y, self.gene)), columns = [self.x_label, self.y_label, self.gene_label])
        grouped = df.groupby(self.gene_label)
        
        if not isinstance(gene_order, np.ndarray):
            gene_order = self.unique_genes
        
        for g in gene_order:
            data = grouped.get_group(g)
            yield g, data.loc[:, self.x_label].to_numpy(), data.loc[:, self.y_label].to_numpy()

    def make_pandas(self):
        pandas_df = pd.DataFrame(data = np.column_stack([self.x, self.y, self.gene]), columns = [self.x_label, self.y_label, self.gene_label])
        return pandas_df
    
    @lru_cache(maxsize=None)
    def _make_xy_coordinates(self):
        return {g: np.column_stack((x, y)).astype('float32') for g, x, y in self.xy_groupby_gene_generator()}
    
    def _make_xyz_coordinates(self):
        return  {g: np.column_stack((xy, np.array([self.z_offset]*xy.shape[0]))).astype('float32') for g, xy in self._make_xy_coordinates().items()}

    def make_gene_coordinates(self, save_z = False) -> None:
        """Make a dictionary with point coordinates for each gene.

        Output will be in self.gene_coordinates. Output will be cached so that
        this function can be called many times but the calculation is only
        performed the first time.

        """
        if save_z:
            self.gene_coordinates = self._make_xyz_coordinates()
        else:
            self.gene_coordinates = self._make_xy_coordinates()
            



class MultiIteration(Iteration):

    def make_multi_gene_coordinates(self, n_jobs=None) -> None:

        #if n_jobs == None:
        #    n_jobs = self.cpu_count

        for d in self.datasets:
            d.make_gene_coordinates()

        #with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        #    parallel(delayed(d.make_gene_coordinates) for d in self.datasets)

