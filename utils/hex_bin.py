import pandas as pd
import numpy as np
from matplotlib.pyplot import hexbin
from math import ceil

def make_hexbin(spacing, title, data, min_count=1):
    """
    Bin 2D expression data into hexagonal bins.
    
    Input:
    `spacing`(int): distance between tile centers, in same units as the data. 
        The actual spacing will have a verry small deviation (tipically lower 
        than 2e-9%) in the y axis, due to the matplotlib hexbin function.
        The function makes hexagons with the point up: ⬡
    `title`(list): List of names of the datasets.
    `data`(list): List pandas dataframes with the location data. Should have
        columns "r_stitched_coords" and "c_stitched_coords" for row (X) and 
        column (Y) coordinates respectively. And a columns named "gene" for the 
        gene labels of the detected spots.
    `min_count`(int): Minimal number of molecules in a tile to keep the tile in 
        the dataset. Default = 1. The algorithm will generate a lot of empty 
        tiles, which are later discarded using the min_count threshold.
    
    Output:
    Dictionary with the datasets titles as kyes.
    For each dataset the following items are stored:
    `gene` --> Dictionary with tile counts for each gene.
    `coordinates` --> XY coordinates for all tiles.
    `coordinates_filt` --> XY coordinates for all tiles that have
        enough counts according to "min_count"
    `df` --> Pandas dataframe with counts for all genes in all 
        valid tiles.
    `spacing` --> Chosen spacing. Keep in mind that the distance between tile
        centers in different rows might deviate slightly. 
    `hexagon_shape` --> matplotlib polygon. The first array contains the corner 
        coordinates relative to the centroid. Centroids for each hexagon can be
        found under "coordinates" or "coordinates_filt".
    
    """
    hex_binned = {}

    #Iterate over datasets
    for i, (molecules, name) in enumerate(zip(data, title)):
        print(f'Start processing {title[i]}          ', end='\r')

        #Determine canvas space
        max_x = molecules.loc[:,'r_stitched_coords'].max()
        min_x = molecules.loc[:,'r_stitched_coords'].min()
        max_y = molecules.loc[:,'c_stitched_coords'].max()
        min_y = molecules.loc[:,'c_stitched_coords'].min()

        #Determine largest axes and use this to make a hexbin grid with square extent.
        #If it is not square matplotlib will stretch the hexagonal tiles to an asymetric shape.
        xlength = max_x - min_x
        ylength = max_y - min_y

        if xlength > ylength:
            #Find number of points   
            n_points = ceil((max_x - min_x) / spacing)
            #Correct x range to match whole number of tiles
            extent = n_points * spacing
            difference_x = extent - xlength
            min_x = min_x - (0.5 * difference_x)
            max_x = max_x + (0.5 * difference_x)
            # *
            #Adjust the y scale to match the number of tiles in x
            #For the same lengt the number of tiles in x is not equal
            #to the number of rows in y.
            #For a hexagonal grid with the tiles pointing up, the distance
            #between rows is (x_spacing * sqrt(3)) / 2
            #(sqrt(3)/2 comes form sin(60))
            xlength = max_x - min_x
            y_spacing = (spacing * np.sqrt(3)) / 2
            n_points_y = int(xlength / y_spacing)
            extent_y = n_points_y * y_spacing
            difference_y = extent_y - ylength
            min_y = min_y - (0.5 * difference_y)
            max_y = max_y + (0.5 * difference_y)

        else:
            #Find number of points  
            n_points = ceil((max_y - min_y) / spacing)
            #Correct y range to match whole number of tiles
            extent = n_points * spacing
            difference_y = extent - ylength
            min_y = min_y - (0.5 * difference_y)
            max_y = max_y + (0.5 * difference_y)
            #Correct x range to match y range
            #because the x dimension is driving for the matplotlib hexbin function 
            ylength = max_y - min_y
            difference_x = ylength - xlength
            min_x = min_x - (0.5 * difference_x)
            max_x = max_x + (0.5 * difference_x)
            #Adjust the y scale to match the number of tiles in x
            #See explantion above at *
            y_spacing = (spacing * np.sqrt(3)) / 2
            n_points_y = int(ylength / y_spacing)
            extent_y = n_points_y * y_spacing
            difference_y = extent_y - ylength
            min_y = min_y - (0.5 * difference_y)
            max_y = max_y + (0.5 * difference_y)

        #Get genes
        unique_genes = np.unique(molecules.loc[:,'gene'])
        n_genes = len(unique_genes)

        #Make result dictionarys
        hex_binned[name] = {}
        hex_binned[name]['gene'] = {}
        hex_binned[name]['spacing'] = spacing

        #Perform hexagonal binning for each gene
        for gene, coords in  molecules.loc[:, ['gene', 'r_stitched_coords', 'c_stitched_coords']].groupby('gene'):
            coords_r = np.array(coords.loc[:,'r_stitched_coords'])
            coords_c = np.array(coords.loc[:,'c_stitched_coords'])
            #Make hex bins and get data
            hb = hexbin(coords_r, coords_c, gridsize=int(n_points), extent=[min_x, max_x, min_y, max_y], visible=False)
            hex_binned[name]['gene'][gene] = hb.get_array()

        #Get the coordinates of the tiles, parameters should be the same regardles of gene.
        hex_bin_coord = hb.get_offsets()
        hex_binned[name]['coordinates'] = hex_bin_coord
        hex_binned[name]['hexagon_shape'] = hb.get_paths()

        #Make dataframe with data
        tiles = hex_bin_coord.shape[0]
        df_hex = pd.DataFrame(data=np.zeros((len(hex_binned[name]['gene'].keys()), tiles)),
                             index=unique_genes, columns=[f'{title[i]}_{j}' for j in range(tiles)])
        for gene in df_hex.index:
            df_hex.loc[gene] = hex_binned[name]['gene'][gene]

        #Filter on number of molecules
        filt = df_hex.sum() >= min_count
        df_hex_filt = df_hex.loc[:,filt]

        #Save data
        hex_binned[name]['coordinates_filt'] = hex_binned[name]['coordinates'][filt]
        hex_binned[name]['df'] = df_hex_filt
        hex_binned[name]['filt'] = filt

    return hex_binned

