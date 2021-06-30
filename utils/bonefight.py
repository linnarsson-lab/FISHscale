import bone_fight as bf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BoneFight:
    
    def bonefight(self, X_1, volume_1, X_2=None, volume_2=None, transform: bool=False, plot: bool=False,
                  spacing: float=100, min_count: int=10, **kwargs):
        """Perform BoneFight to align two datasets.
        
        This is a wrapper around BoneFight to make implementation with 
        FISHscale data easier. Made for 2D data (genes by objects).
        For more advanced features, use Bonefight directly.
        
        WARNING: This function requires input for X_1 and X_2 as (genes, 
        clusters) which is the opposite of BoneFight itself. Clusters could
        also be cells, groups or any other object.
        
        Args:
            X_1 (Pandas dataframe ornp.ndarray): Preferred a Pandas dataframe 
                with features as rows, with names in the index. And cells/
                clusters/groups in columns. Alternatively a numpy array.
            volume_1 (np.ndarray): Volume prior for X_1 for each column of X_1.
            X_2 ([type], optional): Preferred a Pandas dataframe 
                with features as rows, with names in the index. And cells/
                clusters/groups in columns. Alternatively a numpy array can be
                provided in the same format, in which case, make sure the order
                of the index is identical with X_1.
                If None is provided a hexagonal bin will be made of the data
                which will be used as X_2. Defaults to None.
            volume_2 ([type], optional): Volume prior for X_1 for each column 
                of X_1. If X_2 and volume_2 are set to None it will take the
                sum count of the hexbin tiles. Defaults to None.
            transform (bool, optional): If True transform the data and returns
                the transformed tensor. Defaults to False.
            plot (bool, optional): If True plots the loss function.
                Defaults to False.
            spacing (float, optional): If X_2 is None, makes hexagonal bins
                with this spacing. Defaults to 100.
            min_count (int, optional): If X_2 is None, makes hexagonal bins
                with min_count. Defaults to 10.

        Returns:
            [bone_fight.bone_fight.BoneFight]: Resulting BoneFight model.
            if transform is True it also return the transform tensor as numpy
            array.
        """
                
        if X_2 == None:
            self.vp(f'No input given for X_2, making hexagonal binning of data with spacing {spacing} {self.unit_scale.units} and minimum count {min_count}.')
            X_2, centroids, polygon = self.hexbin_make(spacing=spacing, min_count=min_count)
        
            if volume_2 == None:
                self.vp('No input given for X_2, taking sum hexagonal tile count.')
                volume_2 = X_2.sum().to_numpy()
        
        #handle pandas dataframes
        if isinstance(X_1, pd.core.frame.DataFrame) and isinstance(X_2, pd.core.frame.DataFrame):
            print('got here')
            genes_1, genes_2 = X_1.index.to_numpy(), X_2.index.to_numpy()
            
            print(X_1.shape, X_2.shape)
            #Dataset 2 has more rows
            if len(genes_1) < len(genes_2):
                gene_filt_2 = np.isin(genes_2, genes_1)
                genes_2 = genes_2[gene_filt_2]
                self.vp(f'{len(genes_2)} matching features between X_1 and X_2')
                #missing = [g for g in genes_1 if g not in genes_2]
                #print(f'Genes present in X_1 but not in X_2: {missing}')
                X_1 = X_1.loc[genes_2, :].to_numpy()
                X_2 = X_2.loc[genes_2, :].to_numpy()
            
            #Dataset 1 has more rows
            else:
                print('got to this')
                gene_filt_1 = np.isin(genes_1, genes_2)
                genes_1 = genes_1[gene_filt_1]
                self.vp(f'{len(genes_1)} matching features between X_1 and X_2')
                #missing = [g for g in genes_2 if g not in genes_1]
                #print(f'Genes present in X_2 but not in X_1: {missing}')
                X_1 = X_1.loc[genes_1, :].to_numpy()
                X_2 = X_2.loc[genes_1, :].to_numpy()
            print(X_1.shape, X_2.shape)
        
        else:
            if X_1.shape[1] != X_2.shape[1]:
                raise Exception('Both datasets should have the same number of rows')
            else:
                print('Continuing with un-indexed input, make sure feature order is identical')
                
        if X_1.shape[1] == volume_1.shape:
            raise Exception('X_1 columns should match the volume_1')
        
        if X_2.shape[1] == volume_2.shape:
            raise Exception('X_2 columns should match the volume_2')
              
        #Make views        
        view_1 = bf.View(X_1.T, volume_1)
        view_2 = bf.View(X_2.T, volume_2) 
                
        #make the model
        model = bf.BoneFight(view_1, view_2, *kwargs).fit()
        
        #Plot losses
        if plot:
            plt.figure()
            plt.plot(model.losses)
            plt.title('Convergence')
            plt.xlabel('Epoch')
            plt.ylabel('Losses')
        
        #Transform the data and return
        if transform:
            labels = np.eye(X_1.shape[1]) #it is transposed now
            y = model.transform(labels)
            return model, y
        else:
            return model
        