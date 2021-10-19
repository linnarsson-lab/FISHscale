from scipy.spatial import KDTree
from scipy.stats import spearmanr
import numpy as np
from functools import lru_cache
import pandas as pd
from typing import Tuple, Any

class GeneCorr:

    @lru_cache(maxsize=None)
    def make_gene_KDTree(self) -> None:
        """Make gene coordinate and gene KDTree dictionaries.

        Results stored as:
        "self.gene_KDTree" With a scipy.spatial.KDTree for the point of each
        gene. 
        
        """
        gene_KDTree = {}
        for gene in self.unique_genes:
            gene_KDTree[gene] = KDTree(self.get_gene(gene))

        self.gene_KDTree = gene_KDTree

    def _CBC(self, geneA: str, geneB: str, radius: float, workers: int=-1) -> Tuple[Any, Any]:
        """Calculate coordinate based colocalization.

        Malkusch, S., Endesfelder, U., Mondry, J. et al. Coordinate-based 
        colocalization analysis of single-molecule localization microscopy 
        data. Histochem Cell Biol 137, 1–10 (2012). 
        https://doi.org/10.1007/s00418-011-0880-5

        # Formula: (AA / max dots) * (max area A / max area r)

        Args:
            geneA (str): Name of first gene.
            geneB (str): name of second gene.
            geneA (pandas.core.frame.DataFrame): Dataframe with x and y 
                coordinates for geneA.
            geneB (pandas.core.frame.DataFrame): Dataframe with x and y 
                coordinates for geneB.
            radius (float): Radius within to look for neighbouring spots. The 
                algorithm counts the number of spots of gene A and gene B 
                within the radius for all spots of gene A and gene B. The plot
                has a black bar for the size of the radius.
            workers (int, optional): Number of processes for the scipy KDTree 
                .query_ball_point() function. -1 is all processors. 
                Defaults to -1.

        Returns:
            [Tuple]: Tuple of Spearman coorrelation results. The first is the 
                correlation of geneA to geneB, and the second the reverse. 
                Each object contains the r value and a p value.
        """
        #Area of the chosen radius
        area = np.pi * (radius**2)
        
        #Get the points
        geneA_points = self.get_gene(geneA)
        geneB_points = self.get_gene(geneB)

        #Calculate maximum radius to fit all spots and the max area for that radius
        max_radius_A_x = abs(geneA_points.x.min() - geneA_points.x.max())
        max_radius_A_y = abs(geneA_points.y.min() - geneA_points.y.max())
        max_radius_A = max_radius_A_x if max_radius_A_x > max_radius_A_y else max_radius_A_y
        max_area_A = np.pi * (max_radius_A**2)

        max_radius_B_x = abs(geneB_points.x.min() - geneB_points.x.max())
        max_radius_B_y = abs(geneB_points.y.min() - geneB_points.y.max())
        max_radius_B = max_radius_B_x if max_radius_B_x > max_radius_B_y else max_radius_B_y
        max_area_B = np.pi * (max_radius_B**2)

        #Get max number of dots
        n_spots_A = self.gene_n_points[geneA]
        n_spots_B = self.gene_n_points[geneB]

        #Get the trees
        treeA = self.gene_KDTree[geneA]
        treeB = self.gene_KDTree[geneB]

        #Get number of neighbours of A for each spot in A
        AA = treeA.query_ball_point(geneA_points, radius,return_length=True, workers=workers)
        #get number of neighbours of B for each spot in A
        AB = treeB.query_ball_point(geneA_points, radius,return_length=True, workers=workers)
        #get number of neighbours of B for each spot in B
        BB = treeB.query_ball_point(geneB_points, radius,return_length=True, workers=workers)
        #get number of neighbours of A for each spot in B
        BA = treeA.query_ball_point(geneB_points, radius,return_length=True, workers=workers)

        #Calculate coordinate-based colocalization. corrected for area and total density. 
        DAA = (AA / area) * (max_area_A / n_spots_A)
        DAB = (AB / area) * (max_area_B / n_spots_B)

        DBB = (BB / area) * (max_area_B / n_spots_B)
        DBA = (BA / area) * (max_area_A / n_spots_A)

        #Calculate correlation between the number of self counts and other counts
        cor_AB = spearmanr(DAA, DAB)
        cor_BA = spearmanr(DBB, DBA)
        
        return cor_AB, cor_BA

    def _CBC_matrix(self, radius: float, workers: int=-1) -> Tuple[Any, Any]:
        """Calculate coordinate based colocalization for all genes.

        Malkusch, S., Endesfelder, U., Mondry, J. et al. Coordinate-based 
        colocalization analysis of single-molecule localization microscopy 
        data. Histochem Cell Biol 137, 1–10 (2012). 
        https://doi.org/10.1007/s00418-011-0880-5

        Args:
            radius (float): Radius within to look for neighbouring spots. The 
                algorithm counts the number of spots of gene A and gene B 
                within the radius for all spots of gene A and gene B.
            workers (int, optional): Number of processes for the scipy KDTree 
                .query_ball_point() function. -1 is all processors. 
                Defaults to -1.

        Returns:
            [Tuple[pd.DataFrame, pd.DataFrame]]: Tuple of two pandas dataframes
            The first contains the spearman r values and the second the p
            values.

        """
        #Make sure self.gene_KDTree is made.
        self.make_gene_KDTree()
        
        #make empty matrices to put the r and p values in.
        genes = self.unique_genes
        n_genes = len(genes)
        cor = pd.DataFrame(np.zeros((n_genes, n_genes)), index = genes, columns = genes)
        p = pd.DataFrame(np.zeros((n_genes, n_genes)), index = genes, columns = genes)

        combinations = np.array(np.meshgrid(cor.index, cor.index)).T.reshape(-1,2)

        #Iterate through all combinations of genes and fill the r and p matrices
        for i, (geneA, geneB) in enumerate(combinations):
            print(f'Radius {radius}: {i}/{combinations.shape[0]}             ', end='\r')
            #Get the spearman r and p value for gene A versus gene B and gene B versus gene A
            cor_AB, cor_BA = self._CBC(geneA, geneB, radius, workers=workers)
            #Place in matrix
            if geneA == geneB:
                cor.loc[geneA, geneB] = cor_AB[0]
                p.loc[geneA, geneB] = cor_AB[1]
            else:
                cor.loc[geneA, geneB] = cor_AB[0]
                cor.loc[geneB, geneA] = cor_BA[0]
                p.loc[geneA, geneB] = cor_AB[1]
                p.loc[geneB, geneA] = cor_BA[1]
            
        return cor, p

    def gene_corr_CBC(self, radius: float, workers: int = -1) -> Tuple[Any, Any]:
        """Spatially correlate genes based on Colocalization Based Correlation.

        Using:
        Malkusch, S., Endesfelder, U., Mondry, J. et al. Coordinate-based 
        colocalization analysis of single-molecule localization microscopy 
        data. Histochem Cell Biol 137, 1–10 (2012). 
        https://doi.org/10.1007/s00418-011-0880-5

        Note that the matrix is not symetrical as the correlation of gene A to 
        gene B is not the same as gene B to gene A. CBC for gene A to gene B 
        is calculated by counting the number of points of identity A and 
        identity B around all points of A, and correlating the counts. For the 
        correlation between B and A, it does this process for all points of B.
        Therefore, the correlation is dependent on the direction.

        Args:
            radius (float): [description]
            workers (int, optional): [description]. Defaults to -1.

        Returns:
            Tuple[Any, Any]: [description]
        """
        #Make sure self.gene_KDTree is made.
        self.make_gene_KDTree()
        
        return self._CBC_matrix(radius, workers)


    def gene_corr_hex(self, df_hex: Any=None, method: str='spearman', spacing: float=None, min_count: int=1) -> Any:
        """Correlate gene expression over hexagonally binned data.

        Args:
            df_hex (Any, optional): Pandas dataframe with gene counts for each 
                hexagonal tile. Genes in rows, tiles as columns. If not given,
                it will be calculated. In which case "spacing" and "min_count"
                need to be defined. Defaults to None. 
            method (str, optional): Method input for Pandas .corr() method.
                Defaults to 'spearman'.
           spacing (float, optional): distance between tile centers, in same 
                units as the data. The actual spacing will have a verry small 
                deviation (tipically lower than 2e-9%) in the y axis, due to 
                the matplotlib function. The function makes hexagons 
                with the point up: ⬡
            min_count (int, optional): Minimal number of molecules in a tile to
                keep the tile in the dataset. The algorithm will generate a lot
                of empty tiles, which are later discarded using the min_count 
                threshold. Suggested to be at least 1.


        Returns:
            Pandas dataframe with correlation matrix between genes.
        """
        if type(df_hex) == type(None):
            if spacing == None and min_count == None:
                raise Exception('If "df_hex" is not defined, both "spacing" and "min_count" need to be defined.')
            df_hex, hex_coord, hexbin_hexagon_shape = self.make_hexbin(spacing, min_count)

        return df_hex.T.corr(method)    

    def gene_corr_region(self, gdf) -> Any:
        """Correlate gene expression over regions.
        
        Args:
            gdf (Geopandas dataframe): Geopandas Dataframe with expression 
            per region.

        Returns:
            Pandas dataframe with correlation matrix between genes.
        """

        return gdf.iloc[:-1].corr()