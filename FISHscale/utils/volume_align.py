import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import optical_flow_tvl1, phase_cross_correlation
from skimage.transform import warp
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import pandas as pd
import math
from scipy.interpolate import LinearNDInterpolator
import logging

def _register_worker(imgA: np.ndarray, imgB: np.ndarray, 
                         attachment: int=20, tightness: float=0.3, 
                         num_warp: int=10, num_iter: int=10, 
                         tol: float=0.0001, prefilter: bool=False):
        
        if not imgA.shape == imgB.shape:
            raise Exception('Shapes of both imput images needs to be identical.')
        
        v, u = optical_flow_tvl1(imgA, imgB,
                        attachment=attachment,
                        tightness=tightness,
                        num_warp=num_warp,
                        num_iter=num_iter,
                        tol=tol,
                        prefilter=prefilter)
        
        return v, u
    
def _warp( img: np.ndarray, v: np.ndarray, u: np.ndarray, factor: float=1, 
          categorical_data: bool=False):
        
        if factor < 0 or factor > 1:
            raise Exception('Factor should be between 0 and 1')
        v = v * factor
        u = u * factor
        
        nr, nc = img.shape[:2]
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        #Indexing of old data for categorical input
        if categorical_data:
            rc = row_coords + np.round(v).astype(int)
            cc = col_coords + np.round(u).astype(int)
            rc = np.clip(rc, 0, nr-1)
            cc = np.clip(cc, 0, nc-1)
            img_warp = img[rc, cc]
        #Image warping
        else:
            img_warp = warp(img, np.array([row_coords + v, col_coords + u]), mode='edge', clip=False)
        
        return img_warp
    
class Volume_Align():
    """UNDER CONSTRUCTION DO NOT USE"""

    
    def find_warp(self, images: list, attachment: int=20, 
                  tightness: float=0.3, num_warp: int=10, num_iter: int=10, 
                  tol: float=0.0001, prefilter: bool=False, 
                  mixing_factor: float=0.3, second_order=True,
                  categorical_data: bool=False):
        
        #Mixing_factor = 0 #Completely use synthetic made by img0 and img2
        #Mixing_factor = 0.3 # All three images weigh equal. 
        #Mixing_factor = 0.5 #img1 weighs 50% and img0 and img2 weigh 25% each
        #Mixing_factor = 1 #Img1 weighs 100%
        
        z_ordered = self._check_z_sort()
        z_loc = [self.datasets[i].z for i in z_ordered]
        n_datasets = len(self.datasets)
        
        def worker(i, images, attachment, tightness, num_warp, num_iter, tol, 
                   prefilter, mixing_factor, second_order, categorical_data):
            
            def reg_warp(img0, img1, factor, categorical_data):
                v, u = _register_worker(img0, img1,
                                attachment=attachment,
                                tightness=tightness,
                                num_warp=num_warp,
                                num_iter=num_iter,
                                tol=tol,
                                prefilter=prefilter)
                
                if categorical_data:
                    v = np.round(v).astype(int)
                    u = np.round(u).astype(int)
                
                #Make synthetic image that would be the image inbetween img0 and img1
                warped = _warp(img1, v, u, factor=factor, categorical_data=categorical_data)
                
                return warped, v, u
            
            #Prepare input
            #First  and second section
            if i <= 1:
                #img0
                img0 = images[0]
                #img1
                img1 = images[i]
                #img2
                if second_order == True:
                    img2, _, _ = reg_warp(images[i+1], images[i+2], factor=0.5, categorical_data=categorical_data)
                else:
                    img2 = images[i+1]   
                factor = 0.333 #Closer to first section

            #Last two sections
            elif i >= n_datasets-2:
                #img0
                if second_order == True:
                    img0, _, _ = reg_warp(images[i-1], images[i-2], factor=0.5, categorical_data=categorical_data)
                else:
                    img0 = images[i-1] 
                #img1
                img1 = images[i]
                #img2
                img2 = images[-1] 
                factor = 0.666 #Closer to last section

            #Sections inbetween   
            else:
                #img0
                if second_order == True:
                    img0, _, _ = reg_warp(images[i-1], images[i-2], factor=0.5, categorical_data=categorical_data)
                    z0 = z_loc[i-1] -  z_loc[i-2]
                    z0 = z_loc[i-2] + (0.5 * z0)
                else:
                    img0 = images[i] 
                    z0 = z_loc[i-1]
                #img1
                img1 = images[i]
                z1 = z_loc[i]
                #img2
                if second_order == True:
                    img2, _, _ = reg_warp(images[i+1], images[i+2], factor=0.5, categorical_data=categorical_data)
                    z2 = z_loc[i+2] - z_loc[i+1]
                    z2 = z_loc[i+1] + (0.5 * z2)
                else:
                    img2 = images[i+1]
                    z2 = z_loc[i+1]
                    
                #Calculate, as a fraction, the location of img1 between img0 and img2                                  
                factor = (z1 - z0) / (z2 - z0) 
                    
            #Warp
            #Make synthetic image
            synt_image, _, _ = reg_warp(img0, img2, factor=0.5, categorical_data=categorical_data)
            #Let img1 weigh into the synthetic image
            synt_image, _, _ = reg_warp(img1, synt_image, factor=mixing_factor, categorical_data=categorical_data)
            #Warp image
            warped_image, v, u = reg_warp(synt_image, img1, factor=factor, categorical_data=categorical_data)
            
            return v, u, synt_image, warped_image

        lazy_results = []
        images_delayed = delayed(images)

        for i in range(n_datasets):
            r = delayed(worker)(i, images_delayed, attachment, tightness, num_warp, num_iter, tol, prefilter, 
                                mixing_factor, second_order, categorical_data)
            lazy_results.append(r)
            
        with ProgressBar():
            result = dask.compute(*lazy_results, scheduler='processes', n_workers=self.cpu_count)

        vs = []
        us = []
        synthetic_images = []
        warped_images = []    
        for r in result:
            vs.append(r[0])
            us.append(r[1])
            synthetic_images.append(r[2])
            warped_images.append(r[3])
            
        return vs, us, synthetic_images, warped_images
    
    def warp_all(self, squarebin:list, v:list, u:list):
        
        result = []
        for sq, vv, uu in zip(squarebin, v, u):
            warped = np.zeros_like(sq)
            for i in range(warped.shape[2]):
                warped[:,:,i] = self._warp(sq[:,:,i], v=vv, u=uu)
            result.append(warped)
        
        return result
    
    
    def _warped_to_pandas_OLD(self, warped, min_count=1): #See binning.py
        
        z_ordered = self._check_z_sort()
        
        #Bin datasets in sorted Z order
        results = []
        filters = []        
        for i, zi in enumerate(z_ordered):
            d = self.datasets[zi]
            w = warped[i]
            w_sum = w.sum(axis=2)
            filt = w_sum >= min_count
            dataset_name = d.dataset_name
            columns = [f'{dataset_name}_{i}' for i in range(filt.sum())]
            df = pd.DataFrame(data = w[filt].T, index=d.unique_genes, columns=columns)
            results.append(df)
            filters.append(filt)
            
        return results, filters, w.shape[:2]
    
    def _voxels_to_pandas_OLD(self, interpolated, min_count=1): #See binning.py
        #Assumes interpolated is in same order as self.unique_genes
        
        interp_sum = np.sum(np.stack(interpolated), axis=0)
        interp_filt = interp_sum >= min_count
        interp_stack = np.stack(interpolated, axis=3) #Genes as 4th dimension
        shape = interp_stack.shape
        
        results = []
        for i in range(shape[2]): #Iterate through Z
            filt = interp_filt[:,:,i]
            data = interp_stack[:,:,i,:]#.reshape((shape[0] * shape[1], shape[3]))
            data = data[filt]
            columns = [f'{i}_{j}' for j in range(data.shape[0])]
            
            df = pd.DataFrame(data=data.T, index=self.unique_genes, columns=columns)
            results.append(df)
        return results, interp_filt, shape
        
        
            
    def warped_per_gene(self, warped, bin_size: int=100, return_dict=False, z_locations=None):
        
        x, y = warped[0].shape[:2]
        zmin, zmax = self.get_z_range_datasets()
        if type(z_locations) == type(None):
            z_locations = [int(d.z / bin_size) for d in self.datasets]
            z = math.ceil((zmax - zmin) / bin_size)
        else:
            z = z_locations.shape[0]
        n_genes = len(self.unique_genes)
        
        gene_data = [np.zeros((x,y,z), dtype='float32') for i in range(n_genes)]
        
        for w, zi in zip(warped, z_locations):
            for gi in range(n_genes):
                gene_data[gi][:,:,zi] = w[:,:,gi]
                
        if return_dict == True:
            gene_data = dict(zip(self.unique_genes, gene_data))
            
        return gene_data
    
    def interpolate_genes(self, warped_genes, bin_size: int=100):
        
        
        
        gene_data = self.warped_per_gene(warped_genes, bin_size=bin_size, return_dict=False)
        n_genes = len(gene_data)
        #warped_sum = np.rollaxis(np.sum(gene_data, axis=3), 0, 3)
        warped_sum = np.sum(np.stack(gene_data), axis=0)
        
        #Make regular 3D grid
        x, y, z = warped_sum.shape
        grid_3d_complete = np.mgrid[0:x, 0:y, 0:z]

        #Find missing Z slices and make data mask for them
        z_data_filt = []
        missing_masks = []
        for i in range(warped_sum.shape[2]):
            if warped_sum[:,:,i].sum() > 0:
                z_data_filt.append(True)
            
            else:
                z_data_filt.append(False)
                synthetic_mask = np.logical_or(warped_sum[:,:,i-1]>0, warped_sum[:,:,i+1]>0)
                missing_masks.append(synthetic_mask)
        z_data_filt = np.array(z_data_filt)
        missing_masks = np.stack(missing_masks, axis=2)
        self.vp(f'Interpolating for {missing_masks.sum()} points over {len(self.unique_genes)} genes. Total: {len(self.unique_genes) * missing_masks.sum()} calculations.')

        grid_3d = grid_3d_complete[:, :, :, z_data_filt]
        grid_3d = np.rollaxis(grid_3d, 0, 4)
        grid_3d = grid_3d.reshape((x*y*z_data_filt.sum(), 3))

        grid_3d_missing = grid_3d_complete[:,:, :,~z_data_filt]
        grid_3d_missing = np.rollaxis(grid_3d_missing, 0, 4)
        
        def worker(data, z_data_filt, grid_3d, missing_masks, grid_3d_missing):
            #logging.info('got to worker')
            #Get voxels with data
            values = data[:,:,z_data_filt]
            #logging.info(values.shape, values.ravel().shape)
            #Scale to one
            for i in range(values.shape[2]):
                values[:,:,i] = np.sqrt(values[:,:,i]) #Normalize the data
            value_filt = values > 0
            value_filtered = values[value_filt]
            grid_3d = grid_3d[value_filt.ravel()]
            
            try:
                #Learn
                #logging.info(f'Starting learining')
                values_x = LinearNDInterpolator(grid_3d, value_filtered.ravel())
                #logging.info(f'Done with learning')
                
                #Interpolate
                interpolated = []
                mm_shape = missing_masks.shape
                for i in range(mm_shape[2]):
                    mm = missing_masks[:,:,i]
                    points = grid_3d_missing[:,:,i,:].reshape((mm_shape[0] * mm_shape[1], 3))[mm.ravel()]
                    #logging.info(f'{i}/{mm_shape[2]}, n points: {points.shape[0]}')
                    #logging.info(f': {points.shape} points')
                    interp = np.zeros(mm_shape[:2])
                    interp[mm] = values_x(points) #Interpolate
                    interpolated.append(interp)
                    logging.info('Done')
            
            except Exception as e:
                logging.info(e)
                logging.info(f'Encountered error during interpolation in gene: "". Filling with zeros')
                interpolated = []
                mm_shape = missing_masks.shape
                for i in range(mm_shape[2]):
                    interpolated.append(np.zeros(mm_shape[:2]))
                
            #Merge with real data
            #interp = np.copy(data)
            #interp[:,:,~z_data_filt] = np.stack(interpolated, axis=2)
            
            interp_full = np.zeros_like(data)
            interp_full[:,:,z_data_filt] = values
            interp_full[:,:,~z_data_filt] = np.stack(interpolated, axis=2)
            interp_full[np.isnan(interp_full)] = 0
            
            return interp_full
        
        logging.info('Starting')
        
        gene_data = dask.delayed(gene_data)
        
        results = []
        #for data in gene_data:
        for i in range(n_genes):
            r = delayed(worker)(gene_data[i], z_data_filt, grid_3d, missing_masks, grid_3d_missing)
            results.append(r)
        
        with ProgressBar():
            final_results = dask.compute(*results)   
            
        return final_results