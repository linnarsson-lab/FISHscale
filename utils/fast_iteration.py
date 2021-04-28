import pandas as pd
import numpy as np
from typing import Generator, Tuple
from functools import lru_cache
from joblib import Parallel, delayed

class Iteration:

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


    @lru_cache(maxsize=None) #Change to functools.cache when supporting python >= 3.9
    def make_gene_coordinates(self) -> None:
        """Make a dictionary with point coordinates for each gene.

        Output will be in self.gene_coordinates. Output will be cached so that
        this function can be called many times but the calculation is only
        performed the first time.

        """
        self.gene_coordinates = {g: np.column_stack((x, y)) for g, x, y in self.xy_groupby_gene_generator()}


class MultiIteration(Iteration):

    def make_multi_gene_coordinates(self, n_jobs=None) -> None:

        #if n_jobs == None:
        #    n_jobs = self.cpu_count

        for d in self.datasets:
            d.make_gene_coordinates()

        #with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        #    parallel(delayed(d.make_gene_coordinates) for d in self.datasets)

