import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

try:
    import zarr
except ModuleNotFoundError as e:
    print(f'Could not import "Zarr". Ignore Zarr files are not used. Error: {e}')
    
try:
    from cellpose import models, io
except ModuleNotFoundError as e:
    print(f'Could not import Cellpose. Ignore if cell segmentation is not needed. Error: {e}')
    
class Cellpose():
    
    def __init__(self, gpu: bool=False, model_type: str='cyto', 
                 net_avg: bool=True, device: object=None, 
                 torch: bool=True) -> None:
        
        self.cellpose_model = models.Cellpose(gpu = gpu,
                                              model_type = model_type,
                                              net_avg = net_avg,
                                              device = device,
                                              torch = torch)
        
    def cellpose_get_zarr(self, fname):
        pass
        
        
        