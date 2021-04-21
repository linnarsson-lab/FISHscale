import colorsys
import numpy as np

class 

def make_colors(ncolors, saturation_levels=None, value_levels=None,
               shuffle=False):
    """
    Make unique colors based on the HSV color wheel.
    HSV stands for: 
    Hue - the base color. 0 & 1 are red, the full color spectrum is inbetween.
    Saturation - the intenseness of the hue, 0 is white, 1 is fully saturated. 
    Value - the brightness of the color, 0 is black, 1 in fully bright. 
    
    Input:
    `ncolors`(int): Number of requested colors.
    `saturation_levels`(list): List of saturation levels. If None, saturation
        will be set to 1. 
    `value_levels`(list): List of value levels. If None, value will be set to 1.
    `shuffle`(bool): If True returns the color list shuffled so that similar
        colors are not next to each other.
    
    Returns:
    A list of RGB tuples.
    
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