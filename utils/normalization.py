import numpy as np
from typing import Any
import pandas as pd


class Normalization:
    
    def log_norm(self, df):
        """Log normalize pandas dataframe.
        
        Formula: log(X + 1)

        Args:
            df ([pd.DataFrame]): Pandas dataframe with features as rows and 
                samples as columns.

        Returns:
            [pd.DataFrame]: Log normalized Pandas dataframe.
        """
        return np.log(df + 1)
    
    def sqrt_norm(self, df):
        """Square root normalize pandas dataframe.
        
        Formula: sqrt(X)

        Args:
            df ([pd.DataFrame]): Pandas dataframe with features as rows and 
                samples as columns.

        Returns:
            [pd.DataFrame]: Square root normalized Pandas dataframe.
        """
        
        return np.sqrt(df)

    def z_norm(self, df):
        """Z normalize pandas dataframe.
        
        Formula: (X - mean) / std

        Args:
            df ([type]): Pandas dataframe with features as rows and 
                samples as columns.

        Returns:
            [pd.DataFrame]: Pandas dataframe.
        """
        
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        return (df.subtract(mean, axis=0)).divide(std, axis=0)
    
    def div0(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~np.isfinite(c)] = 0  # -inf inf NaN
        return c
    
    def APR(self, df, clip:float =None):
        """Analytic Pearson residuals. 
        
        Calculate bionimial deviance statistics. 
        
        Based on: https://doi.org/10.1101/2020.12.01.405886

        Args:
            df ([pd.DataFrame, np.ndarray]): Pandas dataframe or numpy array 
                with features as rows and samples as columns.
            clip (float, optional): If given data below -`clip` and above
                `clip` will be set to -`clip` and `clip` respectively.
                
        Returns:
            [pd.DataFrame]: Pandas dataframe with results.
        """
        pandas = False
        if type(df) == pd.core.frame.DataFrame:
            pandas = True
            index = df.index
            columns = df.columns
            df = df.to_numpy()
                    
        totals = df.sum(axis=0)
        gene_totals = df.sum(axis=1)
        overall_total = df.sum()
        
        expected = totals[:, None] @ self.div0(gene_totals[None, :], overall_total)
        expected = expected.T
        result = self.div0((df - expected), np.sqrt(expected + np.power(expected, 2) / 100))
        
        if pandas:
            result = pd.DataFrame(data=result, index=index, columns=columns)
        
        if clip != None:
            result = result.clip(lower=-clip, upper=clip)
        
        return result
    
    def normalize(self, data: Any, mode:str = 'log', **kwargs) -> Any:
        """Simple data normalization.

        Args:
            data (np.ndarray, pd.DataFrame): Array or data frame with data.
                Features in rows and samples in columns.
            mode (str, optional): Normalization method. Choose from: "log",
                "sqrt",  "z", "APR" or None. for log +1 transform, square root 
                transform, z scores or Analytic Pearson residuals respectively.
                When the mode is None, no normalization will be performed and
                the input is the output. Defaults to 'log'.

        Raises:
            Exception: If mode is not properly defined

        Returns:
            [np.ndarray, pd.Dataframe]: Normalzed data.
        """

        if mode == 'log':
            result = self.log_norm(data)
        elif mode == 'sqrt':
            result = self.sqrt_norm(data)
        elif mode == 'z':
            result = self.z_norm(data)
        elif mode.lower() == 'apr':
            result = self.APR(data, **kwargs)
        elif mode == None:
            result = data
        else:
            raise Exception(f'Invalid "mode": {mode}')

        return result
    