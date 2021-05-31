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
                
        print(f'\nMissing_genes: {missing_genes}')
        print(f'Present_genes: {present_genes}')
        print(f'Missing_r: {missing_r}')
            
        return list(missing_r), list(missing_genes)

    def ripleyk_calc(self, r: Union[float, List], genes: Union[np.ndarray, List, str] = [],
                    sample_shape: str='circle', boundary_correct: bool=False, CSR_Normalise: bool=False,
                    re_calculate: bool=False, n_jobs: int=-1):

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
                
        print(f'\nMain r: {r}')
        print(f'Main genes: {genes}\n')
                
        if n_jobs == -1:
            n_jobs = self.cpu_count
            
        result = []
        for g in genes:
            lazy_result = dask.delayed(_ripleyk_calc) (r, sample_size, self.get_gene(g), boundary_correct, CSR_Normalise)
            result.append(lazy_result)
        dask.compute(*result)

        #with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        #    result = parallel(delayed(_ripleyk_calc)(r, sample_size, self.get_gene(g), boundary_correct, CSR_Normalise) for g in genes)

        if len(r) > 1:
            for g, res in zip(genes, result):
                for i,j in zip(r, res):
                    self.ripleyk[g][i] = j
                    
        else:
            for g, res in zip(genes, result):
                self.ripleyk[g][r[0]] = res    
    
    def plot_ripleyk_r(self, r):
        
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

    def plot_ripleyk_profile(self, r, gene):
        
        if not isinstance(r, list):
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

        plt.figure(figsize=(10,10))
        plt.plot(r, [self.ripleyk[gene][i] for i in r])
        plt.ylabel("Ripley's k")
        plt.xlabel('Radii')
        plt.title("Ripley's k profile")


#Order genes

#KDE

#KDE percentile area?

#Density 1D

#Density 2D

#Convex hull