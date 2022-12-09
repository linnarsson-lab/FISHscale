# Implementation of Coordinate-based colocalization analysis by:
# Sebastian Malkusch, Ulrike Endesfelder, Justine Mondry, Márton Gelléri, Peter J. Verveer & Mike Heilemann
# Malkusch, S., Endesfelder, U., Mondry, J. et al. Coordinate-based colocalization analysis of single-molecule 
#localization microscopy data. Histochem Cell Biol 137, 1–10 (2012). https://doi.org/10.1007/s00418-011-0880-5
# https://link.springer.com/article/10.1007/s00418-011-0880-5

# Formula: (AA / max dots) * (max area A / max area r)


from scipy.spatial import KDTree
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import logging

def CBC(geneA, geneB, radius, gene_coord, gene_KDTree, plot=False, workers=-1):
    """
    Calculate coordinate based colocalization.
    Based on:
    Malkusch, S., Endesfelder, U., Mondry, J. et al. Coordinate-based colocalization 
    analysis of single-molecule localization microscopy data. Histochem Cell Biol 137, 
    1–10 (2012). https://doi.org/10.1007/s00418-011-0880-5
    
    For two sets of coordinates calculates the correlation in localization.
    Input:
    `geneA`(str): Name of gene A.
    `geneB`(str): Name of gene B.
    `radius`(float): Radius within to look for neighbouring spots. The algorithm counts
        the number of spots of gene A and gene B within the radius for all spots of 
        gene A and gene B. The plot has a black bar for the size of the radius.
    `gene_coord`(dict): Dictionary with gene names as keys and pandas dataframe with
        the X and Y coordinates as columns. These should be named 'c_stitched_coords'
        and 'r_stitched_coords'
    `gene_KDTree`(dict): Dictionary with gene names as keys and scipy KDTree on the 
        coordinates of the spots.
    `plot`(bool): If True, it plots the distribution of the spots of both genes.
    `worker`(int): Number of processes for the scipy KDTree .query_ball_point()
        function. -1 is all processors. 
    
    
    """


    if plot:
        plt.figure(figsize=(12,6))
        plt.scatter(gene_coord[geneA]['c_stitched_coords'], gene_coord[geneA]['r_stitched_coords'], s=1)
        plt.scatter(gene_coord[geneB]['c_stitched_coords'], gene_coord[geneB]['r_stitched_coords'], s=1)
        plt.hlines(10000, -45000, -45000+radius, color='k')
        plt.gca().set_aspect('equal')
        plt.tight_layout()


    #Area of the chosen radius
    area = np.pi * (radius**2)

    #Calculate maximum radius to fit all spots and the max area for that radius
    max_radius_A_x = abs(gene_coord[geneA]['c_stitched_coords'].min() - gene_coord[geneA]['c_stitched_coords'].max())
    max_radius_A_y = abs(gene_coord[geneA]['r_stitched_coords'].min() - gene_coord[geneA]['r_stitched_coords'].max())
    max_radius_A = max_radius_A_x if max_radius_A_x > max_radius_A_y else max_radius_A_y
    max_area_A = np.pi * (max_radius_A**2)

    max_radius_B_x = abs(gene_coord[geneB]['c_stitched_coords'].min() - gene_coord[geneB]['c_stitched_coords'].max())
    max_radius_B_y = abs(gene_coord[geneB]['r_stitched_coords'].min() - gene_coord[geneB]['r_stitched_coords'].max())
    max_radius_B = max_radius_B_x if max_radius_B_x > max_radius_B_y else max_radius_B_y
    max_area_B = np.pi * (max_radius_B**2)

    #Get max number of dots
    n_spots_A = gene_coord[geneA].shape[0]
    n_spots_B = gene_coord[geneB].shape[0]

    #Get the trees
    treeA = gene_KDTree[geneA]
    treeB = gene_KDTree[geneB]

    #Get number of neighbours of A for each spot in A
    AA = treeA.query_ball_point(gene_coord[geneA], radius,return_length=True, workers=workers)
    #get number of neighbours of B for each spot in A
    AB = treeB.query_ball_point(gene_coord[geneA], radius,return_length=True, workers=workers)
    #get number of neighbours of B for each spot in B
    BB = treeB.query_ball_point(gene_coord[geneB], radius,return_length=True, workers=workers)
    #get number of neighbours of A for each spot in B
    BA = treeA.query_ball_point(gene_coord[geneB], radius,return_length=True, workers=workers)

    #Calculate coordinate-based colocalization. corrected for area and total density. 
    DAA = (AA / area) * (max_area_A / n_spots_A)
    DAB = (AB / area) * (max_area_B / n_spots_B)

    DBB = (BB / area) * (max_area_B / n_spots_B)
    DBA = (BA / area) * (max_area_A / n_spots_A)

    #Calculate correlation between the number of self counts and other counts
    cor_AB = spearmanr(DAA, DAB)
    cor_BA = spearmanr(DBB, DBA)
    
    return cor_AB, cor_BA

def make_CBC_matrix(radius, gene_coord, gene_KDTree, workers=-1):
    """
    Make matrix of coordinate based correlation between all genes.
    Input:
    `radius`(float): Radius within to look for neighbouring spots. The algorithm counts
        the number of spots of gene A and gene B within the radius for all spots of 
        gene A and gene B. The plot has a black bar for the size of the radius.
    `gene_coord`(dict): Dictionary with gene names as keys and pandas dataframe with
        the X and Y coordinates as columns. These should be named 'c_stitched_coords'
        and 'r_stitched_coords'
    `gene_KDTree`(dict): Dictionary with gene names as keys and scipy KDTree on the 
        coordinates of the spots.
    `worker`(int): Number of processes for the scipy KDTree .query_ball_point()
        function. -1 is all processors.
    Returns:
    Correlation matrix. Note that the matrix is not symetrical as the correlation
    of A to B is not the same as B to A.
    
    """
    #make empty matrix to put the correlations in.
    genes = sorted(gene_coord.keys())
    n_genes = len(genes)
    cor = pd.DataFrame(np.zeros((n_genes, n_genes)), index = genes, columns = genes)

    combinations = np.array(np.meshgrid(cor.index, cor.index)).T.reshape(-1,2)

    #Iterate through all combinations of genes and fill the correlation matrix
    for i, (geneA, geneB) in enumerate(combinations):
        logging.info(f'Radius {radius}: {i}/{combinations.shape[0]}             ',)
        #Get the spearman r for gene A versos gene B and gene B versus gene A
        cor_AB, cor_BA = CBC(geneA, geneB, radius, gene_coord, gene_KDTree, plot=False, workers=workers)
        #Place in matrix
        if geneA == geneB:
            cor.loc[geneA, geneB] = cor_AB[0]
        else:
            cor.loc[geneA, geneB] = cor_AB[0]
            cor.loc[geneB, geneA] = cor_BA[0]
        
    return cor