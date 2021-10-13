import numpy as np
import ripleyk as rk
from typing import Union, Any, List
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import math
import dask


def _ripleyk_calc(r, s, xy, bc, csr):

    x = xy.x
    y = xy.y
    k = rk.calculate_ripley(r, s, x, y, boundary_correct=bc, CSR_Normalise=csr)
    return k


class SpatialMetrics:

    def largest_extend(self):

        if self.x_extend > self.y_extend:
            return self.x_extend
        else:
            return self.y_extend

    def _ripleyk_to_calc(self, r: List, genes: Union[np.ndarray, List]):
        
        missing_genes = set()
        present_genes = []
        missing_r = set()
        
            
        current_genes = list(self.ripleyk.keys())
        for g in genes:
            #Test if gene is already present, if not add gene and all r
            if g not in current_genes:
                missing_genes.add(g)
                missing_r.update(r)
            else:
                present_genes.append(g)
                
        for i in r:
            if not np.all([i in self.ripleyk[g] for g in present_genes]):
                missing_r.update(r)
                missing_genes.update(present_genes)
        
        if missing_genes != set():
            print(f'\nMissing_genes: {missing_genes}')
            print(f'Present_genes: {present_genes}')
            print(f'Missing_r: {missing_r}')
            
        return list(missing_r), list(missing_genes)

    def ripleyk_calc(self, r: Union[float, List], genes: Union[np.ndarray, List, str] = [],
                    sample_shape: str='circle', boundary_correct: bool=False, CSR_Normalise: bool=False,
                    re_calculate: bool=False):

        #Check input
        if isinstance(r, int) or isinstance(r, float):
            r = [r]
        if isinstance(r, np.ndarray):
            r = list(r)
        if isinstance(genes, str):
            genes = [genes]
        if sample_shape == 'circle':
            sample_size = self.largest_extend()
        elif sample_shape == 'rectangle':
            sample_size = [self.x_extend, self.y_extend]
        else:
            raise Exception('Sample_shape not valid: {sample_shape}, Choose either "circle" or "rectangle".')

        #Collect genes
        if genes == []:
            genes = self.unique_genes

        #Make dictionary
        if not hasattr(self, 'ripleyk'):
            self.ripleyk = {g : {} for g in genes}
        else:
            #Check if there is already existing data            
            if re_calculate == False:
                r, genes = self._ripleyk_to_calc(r, genes)
            for g in genes:
                if g not in self.ripleyk:
                    self.ripleyk[g] = {}

        lazy_result = []
        for g in genes:
            lr = dask.delayed(_ripleyk_calc) (r, sample_size, self.get_gene(g), boundary_correct, CSR_Normalise)
            lazy_result.append(lr)
        futures = dask.persist(*lazy_result, num_workers=1, num_threads=self.cpu_count)
        result = dask.compute(*futures)

        if len(r) > 1:
            for g, res in zip(genes, result):
                for i,j in zip(r, res):
                    self.ripleyk[g][i] = j
                    
        else:
            for g, res in zip(genes, result):
                self.ripleyk[g][r[0]] = res    
    
    def plot_ripleyk_r(self, r: float, frac: float = 0.05):
        """Helper function to pick r value.
        
        Plots a subsample of the data and draws the chosen length scale for r,
        so that the user can see 

        Args:
            r (float): Length scale in the same unit as the data.
            frac (float): Fraction of the data to plot. Defaults to 0.05.
        """
        plt.figure(figsize=(10,10))
        
        data = self.df.loc[:,['x', 'y']].sample(frac=0.05).compute()
        data = data.to_numpy()
        plt.scatter(data[:,0], data[:,1], s=0.02, c='gray')
        
        y_positions = np.linspace(self.y_min, self.y_max, len(r))
        center = self.xy_center[0]
        for i, y in zip(r, y_positions):
            plt.hlines(y, center - (0.5*i), center + (0.5*i), color='red')
            plt.text(center + (0.5*i), y, f' {i*self.unit_scale}', va = 'center')
            
        plt.title('Ripley K radius')
        plt.gca().set_aspect('equal')


#RipleK profile

    def plot_ripley_profile(self, r: list, gene: str, mode: str = 'k'):
        """Plot the profile of one of the Ripley's parameters.
        
        With the mode you can pick the data you want to plot.     

        Args:
            r (list): List of radii to plot.
            gene (str): Gene of interest to plot.
            mode (str): Plotting mode:
                "k" = Ripley's K.
                "l" = Ripley's L.
                "l-r" = L value minus radius at which the L was calculated.
                "k-a" = Ripley's K minus the area at that radius.

        Raises:
            Exception: Raises exception if requested Ripley's K is not computed
                yet, and gives instructions on how to do that. 
        """
        
        if not isinstance(r, list) and not isinstance(r, np.ndarray):
            print('For a Ripley K profile pick multiple radii.')
            r = [r]

        missing_r, missing_gene = self._ripleyk_to_calc(r, [gene])
        if missing_r != []:
            print('The Ripley K value for the following gene and radii are not calculated yet.')
            print('Please calculate them with the same setting for all using the "ripleyk_calc()" function.')
            print('If some values were previously calculated with different settings, set the "re_calculate" parameter to True to overwrite.')
            print(f'Missing gene: {missing_gene}')
            print(f'Missing radii: {missing_r}')
            raise Exception('Missing Ripley K values, please calculate')
        
        if not isinstance(r, np.ndarray):
            r = np.array(r)

        plt.figure(figsize=(10,10))
        k = np.array([self.ripleyk[gene][i] for i in r])
        if mode == 'l':
            data = np.sqrt(k / np.pi)
            label = "Ripley's l"
        elif mode == 'l-r':
            data = np.sqrt(k / np.pi)
            data = data - r
            label = "Ripley's l - radius  "
        elif mode == 'k-a':
            data = k - np.pi * r ** 2
            label = "Ripley's k - area"            
        else:
            data = k
            label = "Ripley's k"
        
        plt.plot(r, data)
        plt.ylabel(label)
        plt.xlabel('Radii')
        plt.title(f"{label} profile")


#KDE

#KDE percentile area?

#Density 1D

#Density 2D

#Convex hull

#Moran I