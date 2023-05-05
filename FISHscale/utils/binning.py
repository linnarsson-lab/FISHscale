import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from typing import Tuple

class Binning:
       
    def squarebin_single(self, genes: list, width: float=None, height:float=None, 
                       bin_size: float=100, percentile: float=97, 
                       plot: bool=True):
        """Bin the data of one or multiple genes in a square grid.

        Args:
            genes (list): List of genes to bin. If multiple genes are provided
                the function will combine and rescale the binned data.
            width (float, optional): Width to use. If None, will default to the 
                x_extent of the dataset. Defaults to None.
            height (float, optional): Height to use. If None, will default to
                the y_extent of the dataset. Defaults to None.
            bin_size (float, optional): Size of the bins in the unit of the 
                dataset. Defaults to 100.
            percentile (float, optional): Clip data to upper percentile.
                Defaults to 97.
            plot (bool, optional): Plots. Defaults to True.

        Returns:
            np.ndarray: (Combined) array of binned data.
            np.ndarray: X bin intervals.
            np.ndarray: Y bin intervals.
            
        """
        if width == None:
            width = self.x_extent
        if height == None:
            height = self.y_extent
        
        x_half = width/2
        y_half = height/2

        use_range = np.array([[self.xyz_center[0] - x_half, self.xyz_center[0] + x_half],
                            [self.xyz_center[1] - y_half, self.xyz_center[1] + y_half]])

        x_nbins = int(width / bin_size)
        y_nbins = int(height / bin_size)
        
        #Bin data
        images = []
        for gene in genes:
            data = self.get_gene(gene)

            img, xbin, ybin = np.histogram2d(data.x, data.y, range=use_range, bins=np.array([x_nbins, y_nbins]))
            pc = np.percentile(img, percentile)
            if pc == 0:
                pc = img.max()
            img = img/pc
            img = np.clip(img, a_min=0, a_max=1)
            images.append(img)
        
        #Integrate multiple genes
        img = np.mean(np.array(images), axis=0)    
        
        if plot:
            plt.figure(figsize=(7,7))
            plt.imshow(img)
            plt.title(f'Composite image: {genes}', fontsize=10)
            plt.xlabel('Your gene slection should give a high contrast image.')
        
        return img, xbin, ybin
    
    def _squarebin_single_worker(self, data, x_nbins, y_nbins, use_range, return_bins):
        
        img, xbin, ybin = np.histogram2d(data.x, data.y, range=use_range, bins=np.array([x_nbins, y_nbins]))
        
        if return_bins:
            return img, xbin, ybin
        else:
            return img

    
    def squarebin_single(self, gene: str, width: float=None, height:float=None, 
                       bin_size: float=100,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin the data of a gene into a square bin.

        Args:
            gene (str): Gene name to bin.
            width (float, optional): Width to use. If None, will default to the 
                x_extent of the dataset. Defaults to None.
            height (float, optional): Height to use. If None, will default to
                the y_extent of the dataset. Defaults to None.
            bin_size (float, optional): Size of the bins in the unit of the 
                dataset. Defaults to 100.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            Binned data as np.ndarray.
            Array with X bins.
            Array with Y bins.
            
        """      
        #Get XY localizations of the gene
        data = self.get_gene(gene)
        
        #Get width and height if not provided
        if width == None:
            width = self.x_extent
        if height == None:
            height = self.y_extent

        #Calculate range
        x_half = width/2
        y_half = height/2
        use_range = np.array([[self.xyz_center[0] - x_half, self.xyz_center[0] + x_half],
                            [self.xyz_center[1] - y_half, self.xyz_center[1] + y_half]])

        #Calculate number of bins
        x_nbins = int(width / bin_size)
        y_nbins = int(height / bin_size)
        
        #Bin the data
        img, xbin, ybin = self._squarebin_single_worker(data, x_nbins, y_nbins, use_range, return_bins=True)
        
        return img, xbin, ybin
        
    def squarebin_mask_make(self, squarebin: np.ndarray, min_count: float=1):
        """Make masks for square binned data for pixels that contain data.
        
        Args:
            squarebin (np.ndarray): List of np.ndarrays with binned data into a
                square grid. The arrays should have the shape (X, Y, genes).
                Can be the output of self.squarebin_multi_make()
            min_count (float): Minimum number of counts for a pixel to be 
                considered a valid pixel. Defaults to 1. 

        Returns:
            list: List with boolean masks for each dataset in the shape (X, Y)
        
        """
        mask = np.sum(squarebin, axis=2) >= min_count 
        return mask
    
    def squarebin_combine(self, genes: list, width: float=None, 
                          height:float=None, bin_size: float=100, 
                          percentile: float=100, plot: bool=True):
        """Weighted combination of multiple binned genes. 
        
        Combines the binned results of multiple genes into a single image.
        Before combination each gene is rescaled between 0 and the max so that
        each gene has the same weight. The final images can be rescaled with
        the percentile.  
        
        Args:
            genes (list): List of gene names. 
            width (float, optional): Width to use. If None, will default to the 
                x_extent of the dataset. Defaults to None.
            height (float, optional): Height to use. If None, will default to
                the y_extent of the dataset. Defaults to None.
            bin_size (float, optional): Size of the bins in the unit of the 
                dataset. Defaults to 100.
            percentile (float, optional): Clip data to upper percentile.
                Defaults to 100.
            plot (bool, optional): Plots. Defaults to True.

        Returns:
            np.ndarray: Combined image of binned data. 
        """
        
        images = []
        for gene in genes:
            img, xbin, ybin = self.squarebin_single(gene, width, height, bin_size)
            img = img/img.max()
            #img = np.clip(img, a_min=0, a_max=1)
            images.append(img)
            
        #Integrate multiple genes
        img = np.sum(np.array(images), axis=0)
        pc = np.percentile(img, percentile)
        img = img/pc
        img = np.clip(img, a_min=0, a_max=1)
        
        if plot:
            plt.figure(figsize=(7,7))
            plt.imshow(img)
            plt.title(f'Composite image: {genes}', fontsize=10)

        return img

    def squarebin_make(self, width: float=None, height:float=None, 
                       bin_size: float=100) -> np.ndarray:
        
        #Get width and height if not provided
        if width == None:
            width = self.x_extent
        if height == None:
            height = self.y_extent

        #Calculate range
        x_half = width/2
        y_half = height/2
        use_range = np.array([[self.xyz_center[0] - x_half, self.xyz_center[0] + x_half],
                            [self.xyz_center[1] - y_half, self.xyz_center[1] + y_half]])

        #Calculate number of bins
        x_nbins = int(width / bin_size)
        y_nbins = int(height / bin_size)
        
        binned = []
        for gene in self.unique_genes:
            data = delayed(self.get_gene(gene))
            r = delayed(self._squarebin_single_worker)(data, x_nbins, y_nbins, use_range, False)
            binned.append(r)
            
        with ProgressBar():
            binned_results = compute(*binned)   
            
        return np.stack(binned_results, axis=2)
    
    def pixels_to_pandas(self, data: np.ndarray, mask: np.ndarray=None, 
                         min_count: float=1):
        """Convert pixels to pandas.
        
        Converts an array of (X, Y, n_genes) to a pandas dataframe of 
        (pixels, genes).

        Args:
            data (np.ndarray): Array with gene data in the shape 
                (X, Y, n_genes). Assumes genes are in the same order as 
                self.unique_genes.
            mask (np.ndarray, optional): Mask to select pixels that contain
                data. If not providied will use `min_count` to calculate the 
                count. Defaults to None.
            min_count (float, optional): If mask is not provided, the mask will
                be calculated to select pixels that after summing over all
                genes contain at least the `min_count`. Defaults to 1.
        
        Returns:
            Pandas Dataframe: Dataframe in the shape (pixels, genes). The index
                will have the name 'X_Y' with the X and Y index of the original
                pixel.
            np.ndarray: Array with the mask. 
            
        The original data can be recreated with:
        x,y = mask.shape
        n_genes = self.unique_genes.shape[0]
        original = np.zeros((x, y, n_genes))
        for i, g in enumerate(df.columns):
            original[:,:,i][mask] = df.loc[:, g]
            
        """
        #Assumes interpolated is in same order as self.unique_genes
        
        if type(mask) == type(None):
            mask = self.squarebin_mask_make(data, min_count=min_count)

        index = [f'{i}_{j}' for i,j in zip(*np.where(mask))]
        df = pd.DataFrame(data[mask], index=index, columns=self.unique_genes)
        return df, mask   
    
class Binning_multi:
    
    def _check_z_sort(self):
        """Check if self.datasets is ordered by Z coordinate.
        
        The function returns indexes that can be used to sort the dataset by Z.
        Example:
        z_sorted = md._chec_z_sort()
        md.datasets = md.datasets[z_sorted]

        Raises:
            Exception: If not all datasets have a Z coordinate.

        Returns:
            np.ndarray: Array with indexes of dataset to sort them by Z. 
        """
        
        if not all([hasattr(dd, 'z') for dd in self.datasets]):
            raise Exception('Not all datasets have a z coordinate. Not possbile to run 3D alignment without.')
        
        z_sorted = np.argsort([d.z for d in self.datasets])
        
        return z_sorted
    
    def get_z_range_datasets(self):
        """Get Z range of datasets

        Returns:
            float: lowest Z value.
            float: highest Z value.
        """
        
        all_z = np.array([d.z for d in self.datasets])
        return all_z.min(), all_z.max()
    
    def find_max_width_height(self, margin: float=0.0):
        """Find largest witdh and hight of datasets.

        Args:
            margin (float, optional): Add a margin to the max widht and hight.
                Usefull when subsequently binning the data so that the dataset
                does not touch the edge. As fraction. Defaults to 0.0.

        Returns:
            float: Largest widht (with added margin).
            float: Largest height (with added margin).
        """
        
        width = 0
        height = 0

        #Find largest dataset
        for dd in self.datasets:
            if dd.x_extent > width:
                width = dd.x_extent
                
            if dd.y_extent > height:
                height = dd.y_extent
        
        #Add margin
        width = width + (margin * width)
        height = height + (margin * height)
        
        return width, height
     
    def squarebin_multi_make(self, bin_size: float=100, margin=0.1):
        """Bin the datasets with square bins.

        Args:
            bin_size (float, optional): Size of the bins in the unit of the 
                dataset. Defaults to 100.
            margin (float, optional): Add a margin to the max widht and hight.
                So that the dataset does not touch the edge. As fraction. 
                Defaults to 0.1.

        Returns:
            list: List with an np.ndarray with the binned results for each 
                dataset (should be sorted by Z). The binned results have a 
                hape of (max_widht, max_height, n_genes). Where max_width and 
                max_height are the max values of all datasets with the addition 
                of the margin. The order of genes corresponds to 
                self.unique_genes.
        """
        
        width, height = self.find_max_width_height(margin=margin)
        z_ordered = self._check_z_sort()
        
        #Bin datasets in sorted Z order
        results = []
        for dd in self.datasets:
            results.append(dd.squarebin_make(width, height, bin_size))
            
        return results     
    
    def squarebin_mask_make(self, squarebin: list, min_count: float=1):
        """Make masks for square binned data for pixels that contain data.
        
        Args:
            squarebin (list): List of np.ndarrays with binned data into a
                square grid. The arrays should have the shape (X, Y, genes).
                Can be the output of self.squarebin_multi_make()
            min_count (float): Minimum number of counts for a pixel to be 
                considered a valid pixel. Defaults to 1. 

        Returns:
            list: List with boolean masks for each dataset in the shape (X, Y)
        """
        
        masks = [np.sum(i, axis=2) >= min_count for i in squarebin]
        
        return masks
    
    def squarebin_combine_multi(self, genes: list, width: float=None, 
                          height:float=None, bin_size: float=100, 
                          percentile: float=100, margin: float=0.1,
                          plot: bool=False):
        """Weighted combination of multiple binned genes. 
        
        Combines the binned results of multiple genes into a single image.
        Before combination each gene is rescaled between 0 and the max so that
        each gene has the same weight. The final images can be rescaled with
        the percentile.  
        
        Args:
            genes (list): List of gene names. 
            width (float, optional): Width to use. If None, will default to the 
                x_extent of the dataset. Defaults to None.
            height (float, optional): Height to use. If None, will default to
                the y_extent of the dataset. Defaults to None.
            bin_size (float, optional): Size of the bins in the unit of the 
                dataset. Defaults to 100.
            percentile (float, optional): Clip data to upper percentile.
                Defaults to 100.
            margin (float, optional): Add a margin to the max widht and hight.
                So that the dataset does not touch the edge. As fraction. 
                Defaults to 0.1.
            plot (bool, optional): Plots resutls. Defaults to True.

        Returns:
            list: List of combined image of binned data. 
        """
        width, height = self.find_max_width_height(margin=margin)
        
        results = []
        for dd in self.datasets:
            img = dd.squarebin_combine(genes, width, height, bin_size, percentile, plot)
            results.append(img)
        return results
    
    def voxels_to_pandas(self, data: list, masks: np.ndarray=None, 
                         min_count: float=1):
        """Convert voxels to pandas.
        
        Converts an list of arrays with (X, Y, n_genes) to a pandas dataframe
        of (voxels, genes).

        Args:
            data (np.ndarray): Array with gene data in the shape 
                (X, Y, n_genes). Assumes genes are in the same order as 
                self.unique_genes.
            mask (np.ndarray, optional): Mask to select pixels that contain
                data. If not providied will use `min_count` to calculate the 
                count. Defaults to None.
            min_count (float, optional): If mask is not provided, the mask will
                be calculated to select pixels that after summing over all
                genes contain at least the `min_count`. Defaults to 1.
        
        Returns:
            Pandas Dataframe: Dataframe in the shape (pixels, genes). The index
                will have the name 'X_Y' with the X and Y index of the original
                pixel.
            np.ndarray: Array with the mask. 
            
        The original data can be recreated with:
        x,y = mask.shape
        n_genes = self.unique_genes.shape[0]
        original = np.zeros((x, y, n_genes))
        for i, g in enumerate(df.columns):
            original[:,:,i][mask] = df.loc[:, g]
            
        """
        #Assumes interpolated is in same order as self.unique_genes
        
        if type(masks) == type(None):
            masks = self.squarebin_mask_make(data, min_count=min_count)
            
        dfs = []
        indexes = []
        samples = []
        for i, (dd, arr, m) in enumerate(zip(self.datasets, data, masks)):
            #Make dataframe
            df, _ = dd.pixels_to_pandas(arr, mask=m)
            dfs.append(df)
            #Prepend index with Z coordinate
            index = df.index
            index = np.array([f'{i}_{j}' for j in index])
            indexes.append(index)
            #Make sample
            sample = np.full(df.shape[0], i)
            samples.append(sample)
            
        #Combine everything
        df = pd.concat(dfs)
        df.index = np.concatenate(indexes)
        samples = np.concatenate(samples)
        
        return df, samples, masks