import pandas as pd
import numpy as np
from matplotlib.pyplot import hexbin

def make_hexbin(spacing, title, data, min_count=1):
    """
    Bin 2D expression data into hexagonal bins.
    Input:
    `spacing`(int): distance between tile centers, in same units as
        the data.
    `title`(list): List of names of the datasets.
    `data`(list): List of the datastes
    `min_count`(int): Minimal number of molecules in a tile to keep
        the tile in the dataset. Default = 1. The algorithm will 
        generate a lot of empty tiles, which are later discarded 
        using the min_count threshold.
    
    Output:
    Dictionary with the datasets titles as kyes.
    For each dataset the following items are stored:
    `gene` --> Dictionary with tile counts for each gene.
    `hexbin` --> matplotlib PolyCollection for hex bins.
    `coordinates` --> XY coordinates for all tiles.
    `coordinates_filt` --> XY coordinates for all tiles that have
        enough counts according to "min_count"
    `df` --> Pandas dataframe with counts for all genes in all 
        valid tiles.    
    
    """
    hex_binned = {}

    for i, molecules in enumerate(data):
        print(f'Start processing {title[i]}')

        #Determine canvas space
        max_x = molecules.loc[:,'r_stitched_coords'].max()
        min_x = molecules.loc[:,'r_stitched_coords'].min()
        max_y = molecules.loc[:,'c_stitched_coords'].max()
        min_y = molecules.loc[:,'c_stitched_coords'].min()

        #Determine largest axes and use this to make a square hexbin grid
        #If it is not square matplotlib will stretch the hexagonal tiles. 
        xlength = abs(min_x) + abs(max_x)
        ylength = abs(min_y) + abs(max_y)
        if xlength > ylength:
            delta = xlength - ylength
            min_y = min_y - (0.5*delta)
            max_x = max_y + (0.5*delta)
        else:
            delta = ylength - xlength
            min_x = min_x - (0.5*delta)
            max_x = max_x + (0.5*delta)

        n_points_x = round((abs(max_x)+abs(min_x))/spacing)
        n_points_y = round((abs(max_y)+abs(min_y))/spacing)

        print(f'X: {min_x} {max_x}, n points: {n_points_x}')
        print(f'Y: {min_y} {max_y}, n points: {n_points_y}')

        n_genes = len(np.unique(molecules.loc[:,'gene']))

        #Make result dictionarys
        name = title[i]
        hex_binned[name] = {}
        hex_binned[name]['gene'] = {}


        for gene, coords in  molecules.loc[:, ['gene', 'r_stitched_coords', 'c_stitched_coords']].groupby('gene'):

            coords_r = np.array(coords.loc[:,'r_stitched_coords'])
            coords_c = np.array(coords.loc[:,'c_stitched_coords'])

            #Make hex bins and get data
            hb = hexbin(coords_r, coords_c, gridsize=n_points_x, extent=[min_x, max_x, min_y, max_y], visible=False)
            hex_binned[name]['hexbin'] = hb
            hex_binned[name]['gene'][gene] = hb.get_array()

        #Get the coordinates of the tiles
        hex_bin_coord = hb.get_offsets()
        hex_binned[name]['coordinates'] = hex_bin_coord
        print(len(np.unique(hex_bin_coord[:,0])), len(np.unique(hex_bin_coord[:,1])))

        #Make dataframe with data
        tiles = hex_bin_coord.shape[0]
        df_hex = pd.DataFrame(data=np.zeros((len(hex_binned[name]['gene'].keys()), tiles)),
                             index=np.unique(molecules.loc[:,'gene']), columns=[f'{title[i]}_{j}' for j in range(tiles)])

        for gene in df_hex.index:
            df_hex.loc[gene] = hex_binned[name]['gene'][gene]

        #Filter on number of molecules
        filt = df_hex.sum() >= min_count

        #Filter the dataframe
        df_hex_filt = df_hex.loc[:,filt]

        hex_binned[name]['coordinates_filt'] = hex_binned[name]['coordinates'][filt]
        hex_binned[name]['df'] = df_hex_filt
        print(f'    Done processing {title[i]}\n')
        
    return hex_binned

