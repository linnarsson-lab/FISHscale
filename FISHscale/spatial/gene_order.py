import numpy as np
import pandas as pd
from typing import Any

class Gene_order:
    
    def gini(self, x: np.ndarray, w:np.ndarray= None) -> float:
        """Calcualte gini index.
        
        From: https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python/48999797#48999797

        Args:
            x (np.ndarray): Numpy array.
            w (np.ndarray, optional): Weights. Defaults to None.

        Returns:
            float: gini index.
        """
        x = np.asarray(x)
        if w is not None:
            w = np.asarray(w)
            sorted_indices = np.argsort(x)
            sorted_x = x[sorted_indices]
            sorted_w = w[sorted_indices]
            # Force float dtype to avoid overflows
            cumw = np.cumsum(sorted_w, dtype=float)
            cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
            return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                    (cumxw[-1] * cumw[-1]))
        else:
            sorted_x = np.sort(x)
            n = len(x)
            cumx = np.cumsum(sorted_x, dtype=float)
            # The above formula, with all weights equal to 1 simplifies to:
            return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
        
    def sort_genes_by_gini(self, df_hex: Any = None, spacing: float = 100, min_count: int = 10) -> np.ndarray:
        """Sort genes by gini index. 
        
        Bins the data in hexagonal bins and calculates the gini index per gene.
        Then returns the ordered genes with the most self-clustered genes at 
        the end.
        
        If the most clustered genes turn out to be genes with low expression,
        increase the spacing.
        
        Args:
            df_hex (pandas DataFrame): Dataframe with the hexagonal binning 
                of the data. If not given it will be calculated with the
                spacing and min_count parameters.
            spacing (float): Center to center distance between hexagonal tiles.
            min_count (int): Minimum number of molecules per hexagonal tile.

        Returns:
            np.ndarray: Array with gene names in order, with least self-
            clusterd at the start (low gini), and most self-clusterd at the end
            (high gini).
        """
        
        #Calculate hex bin counts
        if type(df_hex) != pd.core.frame.DataFrame:
            df_hex, hex_coord = self.hexbin_make(spacing, min_count)
        
        #Calculate gini index
        gini_i = np.array([self.gini(df_hex.loc[g]) for g in self.unique_genes])
        
        #Sort gini indexes
        sort = gini_i.argsort()
        
        #Order genes
        order = np.array([self.unique_genes[i] for i in sort])
        
        return order
        
        