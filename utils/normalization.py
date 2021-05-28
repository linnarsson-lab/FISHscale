import numpy as np

class Normalization:
    
    def log_norm(self, df):
        """Log normalize pandas dataframe.
        
        Formula: log(X + 1)

        Args:
            df ([type]): Pandas dataframe.

        Returns:
            [type]: Log normalized Pandas dataframe.
        """
        return np.log(df + 1)
    
    def sqrt_norm(self, df):
        """Square root normalize pandas dataframe.
        
        Formula: sqrt(X)

        Args:
            df ([type]): Pandas dataframe.

        Returns:
            [type]: Square root normalized Pandas dataframe.
        """
        
        return np.sqrt(df)

    def z_norm(self, df):
        """Z normalize pandas dataframe.
        
        Formula: (X - mean) / std

        Args:
            df ([type]): Pandas dataframe.

        Returns:
            [type]: Pandas dataframe.
        """
        
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        return (df.subtract(mean, axis=0)).divide(std, axis=0)
    