import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from typing import Union
from os import path, makedirs
from skimage.segmentation import expand_labels
    
try:
    from cellpose import models, io
except ModuleNotFoundError as e:
    print(f'Could not import Cellpose. Ignore if cell segmentation is not needed. Error: {e}')
    
class Cellpose():
    """Wrapper around Cellpose:"""
    
    def cellpose_init_model(self, gpu: bool=False, model_type: str='nuclei', 
                 net_avg: bool=True, device: object=None, 
                 torch: bool=True) -> None:
        """Initiate the Cellpose model.
        
        Initiates the model and stores it as self.cellpose_model.
        This function needs to run before cellpose_segment() can be used.
        Refer to Cellpose documentation for more details on input.
        https://doi.org/10.1038/s41592-020-01018-x 

        Args:
            gpu (bool, optional): True if Cuda is installed and is to be used.
                Defaults to False.
            model_type (str, optional): "nuclei" or "cytoplasm" segmentation.
                Defaults to 'nuclei'.
            net_avg (bool, optional): Averages build-in networks. See Cellpose
                documentation. Defaults to True.
            device (object, optional): Use saved model. See Cellpose 
                documentation. Defaults to None.
            torch (bool, optional): If True uses Torch instead of Mxnet.
                Defaults to True.
        """
        
        self.cellpose_model = models.Cellpose(gpu = gpu,
                                              model_type = model_type,
                                              net_avg = net_avg,
                                              device = device,
                                              torch = torch)
        
    def cellpose_segment(self, images: Union[list, np.ndarray],
                         image_names: Union[list, np.ndarray] = None,
                         diameter: float = None,
                         channels: list = [0,0],
                         output_folder = 'cellpose_results',
                         save_mask: bool = True,
                         save_flows: bool = False,
                         save_styles : bool = False,
                         save_diams : bool = False,
                         dilate_distance: int = 0,
                         **kwargs):
        """Cellpose segmentation of list of images. 
        
        Cellpose model first needs to be initiated with 
        self.cellpose_init_model().
        
        Cellpose paper: https://doi.org/10.1038/s41592-020-01018-x 

        Args:
            images (Union[list, np.ndarray]): Iterable with images as numpy 
                arrays to segment.
            image_names (Union[list, np.ndarray], optional): List of image
                names. Images will be saved with this name. If None provided,
                will number the images. Defaults to None.
            diameter (float, optional): Cell diameter prior in pixels. If None
                is given, Cellpose will calculate the diameter.
                Defaults to None.
            channels (list, optional): Channel to segment. First element is 
                channel to segment, second is nuclear signal. See Cellpose
                documentation. Defaults to [0,0].
            output_folder (str, optional): Name or path where output should be 
                saved. Will create the folder if it does not exist. 
                Defaults to 'cellpose_results'.
            save_mask (bool, optional): If True, saves the masks.
                Defaults to True.
            save_flows (bool, optional): If True, saves the flows.
                Defaults to False.
            save_styles (bool, optional): If True, saves the styles.
                Defaults to False.
            save_diams (bool, optional): If True saves the diameters.
                Defaults to False.
            dilate_distance (int, optional): Dilate the mask with the given
                distance. The expanded mask will be saved if a non-zero value
                is given. Defaults to 0.
            **kwargs (optional): Kwargs will be passed to the cellpose.eval()
                function. See Cellpose documentation of details.

        Raises:
            Exception: If model has not been initiated.
            
        Requested output is saved as pickled files in the output folder.
        """
        
        #Check if model exists.
        if not hasattr(self, 'cellpose_model'):
            raise Exception('No Cellpose model found. Please intiate model by running: "self.cellpose_init_model()" first')
        
        #Check image names
        if type(image_names) == type(None):
            self.vp('Images have no names, output will be numbered.')
            image_names = np.arange(len(images))
            
        #Make output folder if it does not exist.
        makedirs(output_folder, exist_ok=True)
        
        #Segmentation
        for i, n in zip(images, image_names):
            mask, flow, style, diam = self.cellpose_model.eval(i, diameter=diameter, channels=channels, **kwargs)
            if save_mask:
                pkl.dump(mask, open(path.join(output_folder, f'{n}_mask.pkl'), 'wb'))
            if save_flows:
                pkl.dump(flow, open(path.join(output_folder, f'{n}_flows.pkl'), 'wb'))
            if save_styles:
                pkl.dump(style, open(path.join(output_folder, f'{n}_styles.pkl'), 'wb'))
            if save_diams:
                pkl.dump(diam, open(path.join(output_folder, f'{n}_diams.pkl'), 'wb'))
            if dilate_distance != 0:
                expanded = expand_labels(mask, distance=dilate_distance)
                pkl.dump(expanded, open(path.join(output_folder, f'{n}_expanded_mask.pkl'), 'wb'))
    
    def cellpose_segment_form_zarr(self,
                         zarr_filename: str,
                         diameter: float = None,
                         channels: list = [0,0],
                         output_folder = 'cellpose_results',
                         save_mask: bool = True,
                         save_flows: bool = False,
                         save_styles : bool = False,
                         save_diams : bool = False,
                         dilate_distance: int = 0,
                         **kwargs):
        """Cellpose segmentation of list of images. 
        
        Cellpose model first needs to be initiated with 
        self.cellpose_init_model().
        
        Cellpose paper: https://doi.org/10.1038/s41592-020-01018-x 

        Args:
            zarr_filename (str): File location
                names. Images will be saved with this name. If None provided,
                will number the images. Defaults to None.
            diameter (float, optional): Cell diameter prior in pixels. If None
                is given, Cellpose will calculate the diameter.
                Defaults to None.
            channels (list, optional): Channel to segment. First element is 
                channel to segment, second is nuclear signal. See Cellpose
                documentation. Defaults to [0,0].
            output_folder (str, optional): Name or path where output should be 
                saved. Will create the folder if it does not exist. 
                Defaults to 'cellpose_results'.
            save_mask (bool, optional): If True, saves the masks.
                Defaults to True.
            save_flows (bool, optional): If True, saves the flows.
                Defaults to False.
            save_styles (bool, optional): If True, saves the styles.
                Defaults to False.
            save_diams (bool, optional): If True saves the diameters.
                Defaults to False.
            dilate_distance (int, optional): Dilate the mask with the given
                distance. The expanded mask will be saved if a non-zero value
                is given. Defaults to 0.
            **kwargs (optional): Kwargs will be passed to the cellpose.eval()
                function. See Cellpose documentation of details.

        Raises:
            Exception: If model has not been initiated.
            
        Requested output is saved as pickled files in the output folder.
        """
        
        #Check if model exists.
        if not hasattr(self, 'cellpose_model'):
            raise Exception('No Cellpose model found. Please intiate model by running: "self.cellpose_init_model()" first')
        
            
        #Make output folder if it does not exist.
        makedirs(output_folder, exist_ok=True)
        
        #Segmentation
        import zarr
        zarr_file = zarr.open(zarr_filename, mode='r')
        for n in zarr_file:
            for sub in zarr_file[n]:
                break
            i = zarr_file[n][sub][:]
            mask, flow, style, diam = self.cellpose_model.eval(i, diameter=diameter, channels=channels, **kwargs)
            if save_mask:
                pkl.dump(mask, open(path.join(output_folder, f'{n}_mask.pkl'), 'wb'))
            if save_flows:
                pkl.dump(flow, open(path.join(output_folder, f'{n}_flows.pkl'), 'wb'))
            if save_styles:
                pkl.dump(style, open(path.join(output_folder, f'{n}_styles.pkl'), 'wb'))
            if save_diams:
                pkl.dump(diam, open(path.join(output_folder, f'{n}_diams.pkl'), 'wb'))
            if dilate_distance != 0:
                expanded = expand_labels(mask, distance=dilate_distance)
                pkl.dump(expanded, open(path.join(output_folder, f'{n}_expanded_mask.pkl'), 'wb'))
                
    def cellpose_inspect(self, image: np.ndarray, mask: np.ndarray, 
                         figsize: float=15, vmin: float=None, vmax: float=None,
                         alpha: float=0.5):
        """Plot segmentation mask over input image.

        Args:
            image (np.ndarray): Original image.
            mask (np.ndarray): Segmentation mask.
            figsize (float, optional): Figure size. Figures will be square.
                Defaults to 15.
            vmin (float, optional): Vmin of the image. Image is normalized
                between 0-1. Defaults to None.
            vmax (float, optional): Vmax of the image. Image is normalized 
                between 0-1. Defaults to None.
            alpha (float, optional): Alpha value of the mask. Defaults to 0.5.
        """
        
        #Shuffle the labels to help visualization
        unique = np.unique(mask)
        unique = np.delete(unique, np.array([0]))
        shuffeled = unique.copy()
        np.random.shuffle(shuffeled)
        mix = dict(zip(unique, shuffeled))
        mix[0] = 0
        shuffeled_mask = np.array([mix[i] for  i in mask.ravel()])
        shuffeled_mask = shuffeled_mask.reshape(mask.shape)
        shuffeled_mask = np.ma.masked_where(shuffeled_mask==0, shuffeled_mask)
        
        #Rescale image
        image = image / image.max()

        #Plot
        fig = plt.figure(figsize=(figsize,figsize))
        plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.imshow(shuffeled_mask, cmap='hsv', alpha=alpha)
        
        