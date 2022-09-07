import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import pandas as pd
import math

class Volume_Align():
    
    def _check_z_sort(self):
        
        if not all([hasattr(dd, 'z') for dd in self.datasets]):
            raise Exception('Not all datasets have a z coordinate. Not possbile to run 3D alignment without.')
        
        z_sorted = np.argsort([d.z for d in self.datasets])
        
        return z_sorted
    
    def get_z_range_datasets(self):
        
        all_z = np.array([d.z for d in self.datasets])
        return all_z.min(), all_z.max()
        

    def find_max_width_height(self, margin: float=0.05):
        
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
    
    def squarebin_data_single(self, dd, genes: list, width: float, height: float, bin_size: float=100, percentile: float=97,
                         plot: bool=True):
        
        x_half = width/2
        y_half = height/2

        use_range = np.array([[dd.xy_center[0] - x_half, dd.xy_center[0] + x_half],
                            [dd.xy_center[1] - y_half, dd.xy_center[1] + y_half]])

        x_nbins = int(width / bin_size)
        y_nbins = int(height / bin_size)
        
        #Bin data
        images = []
        for gene in genes:
            data = dd.get_gene(gene)

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
    
    def squarebin_data_multi(self, genes: list, margin: float=0.05, bin_size: float=100, percentile: float=97):
        
        z_ordered = self._check_z_sort()
        width, height = self.find_max_width_height(margin=margin)
        images = []
        xbins = []
        ybins = []
        
        for i in z_ordered:
            d = self.datasets[i]
            self.vp(f'Z level: {d.z} (these should be consecutive)')
            img, xbin, ybin = self.squarebin_data_single(d, 
                                                    genes, 
                                                    width, 
                                                    height, 
                                                    bin_size=bin_size, 
                                                    percentile=percentile, 
                                                    plot=False)
            
            images.append(img)
            xbins.append(xbin)
            ybins.append(ybin)
            
        return images, xbins, ybins
    
    def _squarebin_worker(self, dataset, width: float, height: float, bin_size: float=100):
        
        x_half = width/2
        y_half = height/2

        use_range = np.array([[dataset.xy_center[0] - x_half, dataset.xy_center[0] + x_half],
                            [dataset.xy_center[1] - y_half, dataset.xy_center[1] + y_half]])

        x_nbins = int(width / bin_size)
        y_nbins = int(height / bin_size)
        
        def worker(x, y, use_range, bins):
            img, xbin, ybin = np.histogram2d(x, y, range=use_range, bins=bins)
            return img
        
        #Bin data      
        binned = []
        for g in dataset.unique_genes:
            xy = dataset.get_gene(g)
            r = delayed(worker)(xy.x, xy.y, use_range, np.array([x_nbins, y_nbins]))
            binned.append(r)
        
        with ProgressBar():
            binned_results = dask.compute(*binned)   
            
        return np.stack(binned_results, axis=2)
    
    def squarebin_make(self, bin_size: float=100, margin=0.1):
        
        width, height = self.find_max_width_height(margin=margin)
        z_ordered = self._check_z_sort()
        
        #Bin datasets in sorted Z order
        results = []        
        for i in z_ordered:
            d = self.datasets[i]
            results.append(self._squarebin_worker(d, width, height, bin_size=bin_size))
            
        return results        
    

    
    def _register_worker(self, imgA: np.ndarray, imgB: np.ndarray, 
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
    
    def find_warp(self, images: list, attachment: int=20, 
                  tightness: float=0.3, num_warp: int=10, num_iter: int=10, 
                  tol: float=0.0001, prefilter: bool=False, 
                  mixing_factor: float=0.3):
        
        #Mixing_factor = 0 #Completely use synthetic made by img0 and img2
        #Mixing_facotro = 0.3 # All three images weigh equal. 
        #Mixing_factor = 0.5 #img1 weighs 50% and img0 and img2 weigh 25% each
        #Mixing_factore = 1 #Img1 weighs 100%
        
        z_ordered = self._check_z_sort()
        n_datasets = len(self.datasets)
        dataset_names = [self.datasets[i].dataset_name for i in z_ordered]

        synthetic_images = []
        warped_images = []
        vs = []
        us = []

        self.vp(f'Warping datasets. Z levels should be consecutive')
        for i, dn in enumerate(dataset_names):
            self.vp(f'Finding warp for dataset {i}/{n_datasets} with Z: {self.datasets[i].z}')

            #First section
            if i == 0:
                section_type = 'first'
                before = 0
                after = 1
                factor=0.333 #Closer to first section

            #Last section    
            elif i == n_datasets-1:
                section_type = 'last'
                before = i-1
                after = i 
                factor=0.666 #Closer to last section

            #Sections inbetween   
            else:
                section_type = 'normal'
                before = i-1
                after = i+1
                #Calculate how close the section is to the next image as a fraction
                z_before = self.datasets[z_ordered[before]].z
                z_after = self.datasets[z_ordered[after]].z
                z_middle = self.datasets[z_ordered[i]].z
                factor = (z_middle - z_before) / (z_after - z_before)
  

            img0 = images[before]
            img1 = images[i]
            img2 = images[after]

            ##########################################    
            #Register img0 and img2 to calculate synthetic image inbetween img0 and img2
            v0, u0 = self._register_worker(img0, img2,
                            attachment=attachment,
                            tightness=tightness,
                            num_warp=num_warp,
                            num_iter=num_iter,
                            tol=tol,
                            prefilter=prefilter)
            #Make synthetic image that would be the image inbetween img0 and img2
            synt_image = self._warp(img2, v0, u0, factor=factor) #Factor based on relative distances of img1 between img0 and img2
            
            ##########################################    
            #Register img1 and syntetic image so that img1 weigs in into the synthetic image
            if section_type == 'normal':
                v1, u1 = self._register_worker(img1, synt_image,
                            attachment=attachment,
                            tightness=tightness,
                            num_warp=num_warp,
                            num_iter=num_iter,
                            tol=tol,
                            prefilter=prefilter)
                #Make synthetic image that would be the image inbetween img0 and img2 guided by img1
                synt_image = self._warp(synt_image, v1, u1, factor=mixing_factor)
            
            ##########################################
            #Register img1 and synt_image to calculate the warp of img0 to the image that should be between img0 and img2
            v2, u2 = self._register_worker(synt_image, img1,
                            attachment=attachment,
                            tightness=tightness,
                            num_warp=num_warp,
                            num_iter=num_iter,
                            tol=tol,
                            prefilter=prefilter)
            #Warp img1 to the synthetic image so that it fits inbetween img0 and img2
            warped_image = self._warp(img1, v2, u2, factor=1) #Completely warp

            synthetic_images.append(synt_image)
            warped_images.append(warped_image)
            vs.append(v2)
            us.append(u2)
            
        return vs, us, synthetic_images, warped_images
    
    def find_warp2(self, images: list, attachment: int=20, 
                  tightness: float=0.3, num_warp: int=10, num_iter: int=10, 
                  tol: float=0.0001, prefilter: bool=False, 
                  mixing_factor: float=0.3, second_order=True):
        
        #Mixing_factor = 0 #Completely use synthetic made by img0 and img2
        #Mixing_facotro = 0.3 # All three images weigh equal. 
        #Mixing_factor = 0.5 #img1 weighs 50% and img0 and img2 weigh 25% each
        #Mixing_factore = 1 #Img1 weighs 100%
        
        z_ordered = self._check_z_sort()
        n_datasets = len(self.datasets)
        dataset_names = [self.datasets[i].dataset_name for i in z_ordered]

        synthetic_images = []
        warped_images = []
        vs = []
        us = []
        
        def reg_warp(img0, img1, factor):
            v, u = self._register_worker(img0, img1,
                            attachment=attachment,
                            tightness=tightness,
                            num_warp=num_warp,
                            num_iter=num_iter,
                            tol=tol,
                            prefilter=prefilter)
            #Make synthetic image that would be the image inbetween img0 and img1
            warped = self._warp(img1, v, u, factor=factor)
            
            return warped, v, u
        

        self.vp(f'Warping datasets. Z levels should be consecutive')
        for i, dn in enumerate(dataset_names):
            self.vp(f'Finding warp for dataset {i}/{n_datasets} with Z: {self.datasets[i].z}')

            #Prepare imput
            #First  and second section
            if i <= 1:
                print('first two sections')
                #img0
                img0 = images[0]
                #img1
                img1 = images[i]
                #img2
                if second_order == True:
                    img2, _, _ = reg_warp(images[i+1], images[i+2], factor=0.5)
                else:
                    img2 = images[i]   
                factor = 0.3333 #Closer to first section

            #Last two sections
            elif i >= n_datasets-2:
                print('last 2 sections')
                #img0
                if second_order == True:
                    img0, _, _ = reg_warp(images[i-1], images[i-2], factor=0.5)
                else:
                    img0 = images[i] 
                #img1
                img1 = images[i]
                #img2
                img2 = images[-1] 
                factor = 0.666 #Closer to last section

            #Sections inbetween   
            else:
                #img0
                if second_order == True:
                    img0, _, _ = reg_warp(images[i-1], images[i-2], factor=0.5)
                else:
                    img0 = images[i] 
                #img1
                img1 = images[i]
                #img2
                if second_order == True:
                    img2, _, _ = reg_warp(images[i+1], images[i+2], factor=0.5)
                else:
                    img2 = images[i]
                factor = 0.5   
                    
            #Warp
            #Make synthetic image
            synt_image, _, _ = reg_warp(img0, img2, factor=0.5)
            #Let img1 weigh into the synthetic image
            synt_image, _, _ = reg_warp(img1, synt_image, factor=mixing_factor)
            #Warp image
            warped_image, v, u = reg_warp(synt_image, img1, factor=factor) #Closer to first section

            #Output handling
            synthetic_images.append(synt_image)
            warped_images.append(warped_image)
            vs.append(v)
            us.append(u)
            
        return vs, us, synthetic_images, warped_images
        
    def _warp(self, img: np.ndarray, v: np.ndarray, u: np.ndarray, factor: float=1):
        
        if factor < 0 or factor > 1:
            raise Exception('Factor should be between 0 and 1')
        v = v * factor
        u = u * factor
        
        nr, nc = img.shape[:2]
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        img_warp = warp(img, np.array([row_coords + v, col_coords + u]), mode='edge', clip=False)
        
        return img_warp
    
    def warp_all(self, squarebin:list, v:list, u:list):
        
        result = []
        for sq, vv, uu in zip(squarebin, v, u):
            warped = np.zeros_like(sq)
            for i in range(warped.shape[2]):
                warped[:,:,i] = self._warp(sq[:,:,i], v=vv, u=uu)
            result.append(warped)
        
        return result
    
    def warped_to_pandas(self, warped, min_count=1):
        
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
            
    def warped_per_gene(self, warped, bin_size: int=100, return_dict=False):
        
        x, y = warped[0].shape[:2]
        zmin, zmax = self.get_z_range_datasets()
        z_locations = [int(d.z / bin_size) for d in self.datasets]
        z = math.ceil((zmax - zmin) / bin_size)
        n_genes = len(self.unique_genes)
        
        gene_data = [np.zeros((x,y,z)) for i in range(n_genes)]
        
        for w, zi in zip(warped, z_locations):
            for gi in range(n_genes):
                gene_data[gi][:,:,zi] = w[:,:,gi]
        
        #for wi, (w, zi) in enumerate(zip(warped, z_locations)):
        #    for gi in range(n_genes):
        #        gene_data[gi][:,:,zi] = w[:,:,gi]
                
        if return_dict == True:
            gene_data = dict(zip(self.unique_genes, gene_data))
            
        return gene_data