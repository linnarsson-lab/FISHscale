import colorsys
import numpy as np
import pickle
from os.path import isfile
import warnings
from typing import Optional, Union

class ManyColors:

    def make_colors(self, ncolors: int, saturation_levels: Optional[list]=None, value_levels: Optional[list]=None,
                shuffle: bool=False) -> list:
        """Make unique colors based on the HSV color wheel.

        HSV stands for: 
        Hue - the base color. 0 & 1 are red, the full color spectrum is inbetween.
        Saturation - the intenseness of the hue, 0 is white, 1 is fully saturated. 
        Value - the brightness of the color, 0 is black, 1 in fully bright. 

        Args:
            ncolors (int): Number of requested colors.
            saturation_levels (Optional[list], optional): List of saturation 
                levels. If None, saturation will be set to 1. Defaults to None.
            value_levels (Optional[list], optional): List of value levels. If 
                None, value will be set to 1. Defaults to None.
            shuffle (bool, optional): [description]. If True returns the color 
                array shuffled so that similar colors are not next to each 
                other.

        Returns:
            np.ndarray: A list of RGB tuples.
        
        """        
        colors = np.ones((ncolors, 3))
        
        #If saturation and value are not defined, only vary hue.
        if saturation_levels == None and value_levels == None:
            colors[:,0] = np.linspace(0, 1, ncolors+1)[:-1]
        
        else:
            #Calculate number of hues
            n_sat = len(saturation_levels)
            n_val = len(value_levels)
            n_levels = n_sat + n_val
            
            n_col_per_level = int(ncolors / n_levels)
            residue = ncolors - (n_col_per_level * n_levels)        
            
            #Generate colors with specified saturation levels
            index = 0
            shift = 0 
            for sat in saturation_levels:
                #Set the hue. Hue is shifted to prevent different saturations of the same hue.
                colors[index:index+n_col_per_level, 0] = np.linspace(0, 1, n_col_per_level+1)[:-1] + shift
                #Set the saturation
                colors[index:index+n_col_per_level, 1] = sat
                index += n_col_per_level
                shift += (1 / n_col_per_level) / len(saturation_levels) + 1
            
            #Generate colors with specified saturation levels
            shift = 0
            for val in value_levels:
                #Correct for the residue
                if index == (ncolors - residue - n_col_per_level):
                    n_col_per_level += residue
                #Set the hue. Hue is shifted to prevent different saturations of the same hue.
                colors[index:index+n_col_per_level, 0] = np.linspace(0, 1, n_col_per_level+1)[:-1] + shift
                #Set the saturation
                colors[index:index+n_col_per_level, 2] = val
                index += n_col_per_level
                shift += (1 / n_col_per_level) / len(saturation_levels)
        
        #Shuffle colors
        if shuffle:
            np.random.shuffle(colors)
        
        #Convert to RGB
        rgb = []
        for i in colors:
            rgb.append(colorsys.hsv_to_rgb(i[0], i[1], i[2]))
            
        return rgb

    def make_color_dict(self) -> None:
        """Make a color dictionary for every gene. 
        
        """
        n_genes = len(self.unique_genes)
        if n_genes > 10 and n_genes < 500:
            saturation_levels = [0.9, 0.5]
            value_levels = [0.9, 0.7]
        elif n_genes >= 500:
            saturation_levels = [0.9, 0.7, 0.5]
            value_levels = [0.9, 0.7, 0.5]
        else:
            saturation_levels = None
            value_levels = None
        
        self.color_dict = dict(zip(self.unique_genes, self.make_colors(n_genes, saturation_levels, value_levels, shuffle=True)))


    def save_color_dict(self) -> str:
        """Saves color dictionary.

        Saves in current working directory with name: 
        "<dataset_name>_color_dictionary.pkl"

        Raises:
            AttributeError: If "color_dict" is not generated.

        Returns:
            str: File name

        """
        if not hasattr(self, 'color_dict'):
            raise AttributeError('Can not save "color_dict" because it does not exist. Generate a color dictionary with the "make_color_dict()" function.')
        
        file_name = f'{self.dataset_name}_color_dictionary.pkl'
        pickle.dump(self.color_dict, open(file_name, 'wb'))
        return file_name

    def load_color_dict(self, file: str) -> bool:
        """Load a color dictionary from file.

        Args:
            file (str): Name of file.

        Returns:
            bool: True if loading was successful, False if not.

        """
        try:
            self.color_dict = pickle.load(open(file, 'rb'))
            return True
        except FileNotFoundError:
            return False


    def auto_load_color_dict(self, main: bool=False) -> bool:
        """Automatically load an previously generated color dictionary.

        Looks in the current working dicrectory.
        File should have the name: '<dataset_name>_color_dictionary.pkl'

        Args:
            main (bool): If True gives warning if it failed.

        Returns:
            bool: True if loading was successful, False if not.

        """
        file_name = f'{self.dataset_name}_color_dictionary.pkl'
        if isfile(file_name):
            self.load_color_dict(file_name)
            return True
        else:
            if main:
                warnings.warn('"auto_load_colors" failed. No saved color dictionary in working directory.')
            return False

    def auto_handle_color_dict(self, color_input: Optional[Union[str, dict]] = None) -> None:
        """Automatically handle color dictionary loading.

        Has the option to open a specified file, automatically open an 
        previously generated color dictionary, or make a new color dictionary.

        Args:
            color_input (Optional[str, dict], optional): If a filename is 
                specifiedthat endswith "_color_dictionary.pkl" the function 
                will try to load that dictionary. If "auto" is provided it will
                try to load an previously generated color dictionary for this 
                dataset. If "make", is provided it will make a new color 
                dictionary for this dataset and save it. If a dictionary is 
                proivided it will use that dictionary. If None is provided the 
                function will first try to load a previously generated color 
                dictionary and make a new one if this fails. Defaults to None.

        """
        #Try opening specified color dictionary file
        success = True
        if type(color_input) == dict:
            self.color_dict = color_input
        
        elif type(color_input) == str or color_input == None:
        
            if str(color_input).endswith('_color_dictionary.pkl'):
                success = self.load_color_dict(color_input)
                if success == False:
                    self.vp('Loading specified color dictionary failed, falling back to auto loading.')
            
            #Try opening existing color dictionary
            if color_input == 'auto' or color_input == None or success == False:
                success = self.auto_load_color_dict(main=False)
                if success == False:
                    self.vp('Auto loading color dictionary failed, generating new color dictionary.')

            #Make new color dictionary
            if color_input == 'make' or (color_input == None and success == False):
                self.make_color_dict()
                file_name = self.save_color_dict()
                self.vp(f'Generated new color dictionary and saved as: {file_name}')
        
        else:
            raise Exception(f'Input: {color_input} not understood.')