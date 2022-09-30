import sys
import numpy as np
import pandas as pd
import scipy
import numexpr as ne
from sklearn.neighbors import NearestNeighbors
from pciSeq.src.cell_call.log_config import logger


class Cells(object):
    # Get rid of the properties where not necessary!!
    def __init__(self, _cells_df, config):
        self.config = config
        self.cell_props, self.mcr = self.read_image_objects(_cells_df, config)
        self.nC = len(self.cell_props['cell_label'])
        self.classProb = None
        self.class_names = None
        self._cov = self.ini_cov()
        self._gene_counts = None
        self._background_counts = None

    # -------- PROPERTIES -------- #
    @property
    def yx_coords(self):
        coords = [d for d in zip(self.cell_props['y'], self.cell_props['x']) if not np.isnan(d).any()]
        return np.array(coords)

    @property
    def geneCount(self):
        return self._gene_counts

    @geneCount.setter
    def geneCount(self, val):
        self._gene_counts = val

    @property
    def background_counts(self):
        return self._background_counts

    @background_counts.setter
    def background_counts(self, val):
        # assert val[1:, :].sum() == 0, 'Input array must be zero everywhere apart from the top row'
        # self._background_counts = val[0, :]
        self._background_counts = val

    @property
    def total_counts(self):
        # tc = self.geneCount.sum(axis=1)
        return self.geneCount.sum(axis=1)

    # -------- METHODS -------- #
    def ini_cov(self):
        mcr = self.dapi_mean_cell_radius()
        cov = mcr * mcr * np.eye(2, 2)
        return np.tile(cov, (self.nC, 1, 1))

    def dapi_mean_cell_radius(self):
        return np.nanmean(np.sqrt(self.cell_props['area'] / np.pi)) * 0.5

    def nn(self):
        n = self.config['nNeighbors'] + 1
        # for each spot find the closest cell (in fact the top nN-closest cells...)
        nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(self.yx_coords)
        return nbrs

    # def geneCountsPerKlass(self, single_cell_data, egamma, ini):
    #     # ********************************************
    #     # DEPRECATED. Replaced by a simple einsum call
    #     # ********************************************
    #     temp = np.einsum('ck, c, cgk -> gk', self.classProb, self.cell_props['area_factor'], egamma)
    #
    #     # total counts predicted by all cells of each class (nG, nK)
    #     ClassTotPredicted = temp * (single_cell_data.mean_expression + ini['SpotReg'])
    #
    #     # total of each gene
    #     isZero = ClassTotPredicted.columns == 'Zero'
    #     labels = ClassTotPredicted.columns.values[~isZero]
    #     TotPredicted = ClassTotPredicted[labels].sum(axis=1)
    #     return TotPredicted

    def read_image_objects(self, img_obj, cfg):
        meanCellRadius = np.mean(np.sqrt(img_obj.area / np.pi)) * 0.5
        relCellRadius = np.sqrt(img_obj.area / np.pi) / meanCellRadius

        # append 1 for the misreads
        relCellRadius = np.append(1, relCellRadius)

        nom = np.exp(-relCellRadius ** 2 / 2) * (1 - np.exp(cfg['InsideCellBonus'])) + np.exp(cfg['InsideCellBonus'])
        denom = np.exp(-0.5) * (1 - np.exp(cfg['InsideCellBonus'])) + np.exp(cfg['InsideCellBonus'])
        CellAreaFactor = nom / denom

        out = {}
        out['area_factor'] = CellAreaFactor
        # out['area_factor'] = np.ones(CellAreaFactor.shape)
        # logger.info('Overriden CellAreaFactor = 1')
        out['rel_radius'] = relCellRadius
        out['area'] = np.append(np.nan, img_obj.area)
        out['x'] = np.append(-sys.maxsize, img_obj.x.values)
        out['y'] = np.append(-sys.maxsize, img_obj.y.values)
        out['cell_label'] = np.append(0, img_obj.label.values)
        # First cell is a dummy cell, a super neighbour (ie always a neighbour to any given cell)
        # and will be used to get all the misreads. It was given the label=0 and some very small
        # negative coords

        return out, meanCellRadius

# ----------------------------------------Class: Genes--------------------------------------------------- #
class Genes(object):
    def __init__(self, spots):
        # self.gamma = np.ones(len(spots.unique_gene_names))
        # self.gamma = None
        self.gene_panel = np.unique(spots.data.gene_name.values)
        self._eta = None
        self.nG = len(self.gene_panel)

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, val):
        self._eta = val

# ----------------------------------------Class: Spots--------------------------------------------------- #
class Spots(object):
    def __init__(self, spots_df, config):
        self._parent_cell_prob = None
        self._parent_cell_id = None
        self.config = config
        self.data = self.read(spots_df)
        self.nS = self.data.shape[0]
        self.call = None
        self.unique_gene_names = None
        self._gamma_bar = None
        self._log_gamma_bar = None
        [_, self.gene_id, self.counts_per_gene] = np.unique(self.data.gene_name.values, return_inverse=True, return_counts=True)

    # -------- PROPERTIES -------- #
    @property
    def gamma_bar(self):
        return self._gamma_bar.astype(self.config['dtype'])

    @gamma_bar.setter
    def gamma_bar(self, val):
        self._gamma_bar = val.astype(self.config['dtype'])

    @property
    def log_gamma_bar(self):
        return self._log_gamma_bar

    @log_gamma_bar.setter
    def log_gamma_bar(self, val):
        self._log_gamma_bar = val

    @property
    def xy_coords(self):
        lst = list(zip(*[self.data.x, self.data.y]))
        return np.array(lst)

    @property
    def parent_cell_prob(self):
        return self._parent_cell_prob

    @parent_cell_prob.setter
    def parent_cell_prob(self, val):
        self._parent_cell_prob = val

    @property
    def parent_cell_id(self):
        return self._parent_cell_id

    @parent_cell_id.setter
    def parent_cell_id(self, val):
        self._parent_cell_id = val


    # -------- METHODS -------- #
    def read(self, spots_df):
        # No need for x_global, y_global to be in the spots_df at first place.
        # Instead of renaming here, you could just use 'x' and 'y' when you
        # created the spots_df
        spots_df = spots_df.rename(columns={'x_global': 'x', 'y_global': 'y'})

        # remove a gene if it is on the exclude list
        exclude_genes = self.config['exclude_genes']
        gene_mask = [True if d not in exclude_genes else False for d in spots_df.target]
        spots_df = spots_df.loc[gene_mask]
        return spots_df.rename_axis('spot_id').rename(columns={'target': 'gene_name'})

    def cells_nearby(self, cells: Cells) -> np.array:
        spotYX = self.data[['y', 'x']].values

        # for each spot find the closest cell (in fact the top nN-closest cells...)
        nbrs = cells.nn()
        self.Dist, neighbors = nbrs.kneighbors(spotYX)

        # last column is for misreads.
        neighbors[:, -1] = 0
        return neighbors

    def ini_cellProb(self, neighbors, cfg):
        nS = self.data.shape[0]
        nN = cfg['nNeighbors'] + 1
        SpotInCell = self.data.label
        # assert (np.all(SpotInCell.index == neighbors.index))

        # sanity check (this actually needs to be rewritten)
        mask = np.greater(SpotInCell, 0, where=~np.isnan(SpotInCell))
        sanity_check = neighbors[mask, 0] + 1 == SpotInCell[mask]
        assert ~any(sanity_check), "a spot is in a cell not closest neighbor!"

        pSpotNeighb = np.zeros([nS, nN])
        pSpotNeighb[neighbors == SpotInCell.values[:, None]] = 1
        pSpotNeighb[SpotInCell == 0, -1] = 1

        ## Add a couple of checks here
        return pSpotNeighb

    def loglik(self, cells, cfg):
        # area = cells.cell_props['area'][1:]
        # mcr = np.mean(np.sqrt(area / np.pi)) * 0.5  # This is the meanCellRadius
        mcr = cells.mcr
        dim = 2  # dimensions of the normal distribution: Bivariate
        # Assume a bivariate normal and calc the likelihood
        D = -self.Dist ** 2 / (2 * mcr ** 2) - dim/2 * np.log(2 * np.pi * mcr ** 2)

        # last column (nN-closest) keeps the misreads,
        D[:, -1] = np.log(cfg['MisreadDensity'])

        mask = np.greater(self.data.label, 0, where=~np.isnan(self.data.label))
        D[mask, 0] = D[mask, 0] + cfg['InsideCellBonus']
        return D

    def zero_class_counts(self, geneNo, pCellZero):
        """
        Gene counts for the zero expressing class
        """
        # for each spot get the ids of the 3 nearest cells
        spotNeighbours = self.parent_cell_id[:, :-1]

        # get the corresponding probabilities
        neighbourProb = self.parent_cell_prob[:, :-1]

        # prob that a spot belongs to a zero expressing cell
        pSpotZero = np.sum(neighbourProb * pCellZero[spotNeighbours], axis=1)

        # aggregate per gene id
        TotPredictedZ = np.bincount(geneNo, pSpotZero)
        return TotPredictedZ

    def gammaExpectation(self, rho, beta):
        '''
        :param r:
        :param b:
        :return: Expectetation of a rv X following a Gamma(r,b) distribution with pdf
        f(x;\alpha ,\beta )= \frac{\beta^r}{\Gamma(r)} x^{r-1}e^{-\beta x}
        '''

        # sanity check
        # assert (np.all(rho.coords['cell_id'].data == beta.coords['cell_id'])), 'rho and beta are not aligned'
        # assert (np.all(rho.coords['gene_name'].data == beta.coords['gene_name'])), 'rho and beta are not aligned'

        dtype = self.config['dtype']
        r = rho[:, :, None]
        if dtype == np.float64:
            gamma = np.empty(beta.shape)
            ne.evaluate('r/beta', out=gamma)
            return gamma
        else:
            return (r/beta).astype(dtype)

    def logGammaExpectation(self, rho, beta):
        dtype = self.config['dtype']
        r = rho[:, :, None].astype(dtype)
        if dtype == np.float64:
            logb = np.empty(beta.shape)
            ne.evaluate("log(beta)", out=logb)
            return scipy.special.psi(r) - logb
        else:
            return scipy.special.psi(r) - np.log(beta).astype(dtype)


# ----------------------------------------Class: SingleCell--------------------------------------------------- #
class SingleCell(object):
    def __init__(self, scdata: pd.DataFrame, genes: np.array, config):
        self.config = config
        self._mean_expression, self._log_mean_expression = self._setup(scdata, genes, self.config)

    def _setup(self, scdata, genes, config):
        """
        calcs the mean (and the log-mean) gene counts per cell type. Note that
        some hyperparameter values have been applied before those means are derived.
        These hyperparameters and some bacic cleaning takes part in the functions
        called herein
        """
        expr = self._raw_data(scdata, genes)
        self.raw_data = expr
        me, lme = self._helper(expr.copy())
        dtype = self.config['dtype']

        assert me.columns[-1] == 'Zero', "Last column should be the Zero class"
        assert lme.columns[-1] == 'Zero', "Last column should be the Zero class"
        return me.astype(dtype), lme.astype(dtype)

    # -------- PROPERTIES -------- #
    @property
    def mean_expression(self):
        assert self._mean_expression.columns[-1] == 'Zero', "Last column should be the Zero class"
        return self._mean_expression

    @property
    def log_mean_expression(self):
        assert self._log_mean_expression.columns[-1] == 'Zero', "Last column should be the Zero class"
        return self._log_mean_expression

    @property
    def genes(self):
        return self.mean_expression.index.values

    @property
    def classes(self):
        return self.mean_expression.columns.values

    ## Helper functions ##
    def _set_axes(self, df):
        df = df.rename_axis("class_name", axis="columns").rename_axis('gene_name')
        return df

    def _remove_zero_cols(self, df):
        """
        Removes zero columns (ie if a column is populated by zeros only, then it is removed)
        :param da:
        :return:
        """
        out = df.loc[:, (df != 0).any(axis=0)]
        return out

    def _helper(self, arr):
        # order by column name
        arr = arr.sort_index(axis=0).sort_index(axis=1, key=lambda x: x.str.lower())

        # append at the end the Zero class
        arr['Zero'] = np.zeros([arr.shape[0], 1])

        expr = self.config['Inefficiency'] * arr
        me = expr.rename_axis('gene_name').rename_axis("class_name", axis="columns")  # mean expression
        lme = np.log(me + self.config['SpotReg'])  # log mean expression
        return me, lme

    def _raw_data(self, scdata, genes):
        """
        Reads the raw single data, filters out any genes outside the gene panel and then it
        groups by the cell type
        """
        assert np.all(scdata >= 0), "Single cell dataframe has negative values"
        logger.info(' Single cell data passed-in have %d genes and %d cells' % (scdata.shape[0], scdata.shape[1]))

        logger.info(' Single cell data: Keeping counts for the gene panel of %d only' % len(genes))
        df = scdata.loc[genes]

        # set the axes labels
        df = self._set_axes(df)

        df = self._remove_zero_cols(df.copy())
        dfT = df.T

        logger.info(' Single cell data: Grouping gene counts by cell type. Aggregating function is the mean.')
        out = dfT.groupby(dfT.index.values).agg('mean').T
        logger.info(' Grouped single cell data have %d genes and %d cell types' % (out.shape[0], out.shape[1]))
        return out


# ---------------------------------------- Class: CellType --------------------------------------------------- #
class CellType(object):
    def __init__(self, single_cell):
        assert single_cell.classes[-1] == 'Zero', "Last label should be the Zero class"
        self._names = single_cell.classes
        self._prior = None

    @property
    def names(self):
        assert self._names[-1] == 'Zero', "Last label should be the Zero class"
        return self._names

    @property
    def nK(self):
        return len(self.names)

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, val):
        self._prior = val

    @property
    def log_prior(self):
        return np.log(self.prior)

    def ini_prior(self, ini_family):
        if ini_family == 'uniform':
            self.prior = np.append([.5 * np.ones(self.nK - 1) / self.nK], 0.5)
        else:
            raise Exception('Method not implemented yet. Please pass "uniform" when you call ini_prior() ')





