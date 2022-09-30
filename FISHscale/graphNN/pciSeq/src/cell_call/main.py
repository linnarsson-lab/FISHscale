import numpy as np
import numpy_groupies as npg
from pciSeq.src.cell_call.datatypes import Cells, Spots, Genes, SingleCell, CellType
from pciSeq.src.cell_call.summary import collect_data
import pciSeq.src.cell_call.utils as utils
from pciSeq.src.cell_call.log_config import logger
from tqdm import trange

class VarBayes:
    def __init__(self, _cells_df, _spots_df, scRNAseq, config):
        self.config = config
        self.cells = Cells(_cells_df, config)
        self.spots = Spots(_spots_df, config)
        self.genes = Genes(self.spots)
        self.single_cell = SingleCell(scRNAseq, self.genes.gene_panel, self.config)
        self.cellTypes = CellType(self.single_cell)
        self.nC = self.cells.nC                         # number of cells
        self.nG = self.genes.nG                         # number of genes
        self.nK = self.cellTypes.nK                     # number of classes
        self.nS = self.spots.nS                         # number of spots
        self.nN = self.config['nNeighbors'] + 1         # number of closest nearby cells, candidates for being parent
                                                        # cell of any given spot. The last cell will be used for the
                                                        # misread spots. (ie cell at position nN is the background)

    def initialise(self):
        self.cellTypes.ini_prior('uniform')
        self.cells.classProb = np.tile(self.cellTypes.prior, (self.nC, 1))
        self.genes.eta = np.ones(self.nG)
        self.spots.parent_cell_id = self.spots.cells_nearby(self.cells)
        self.spots.parent_cell_prob = self.spots.ini_cellProb(self.spots.parent_cell_id, self.config)
        self.spots.gamma_bar = np.ones([self.nC, self.nG, self.nK]).astype(self.config['dtype'])

    # -------------------------------------------------------------------- #
    def run(self):
        p0 = None
        iss_df = None
        gene_df = None
        max_iter = self.config['max_iter']

        self.initialise()
        for i in trange(max_iter):

            # 1. For each cell, calc the expected gene counts
            self.geneCount_upd()

            # 2. calc expected gamma
            self.gamma_upd()

            # 3. assign cells to cell types
            self.cell_to_cellType()

            # 4. assign spots to cells
            self.spots_to_cell()

            # 5. update gene efficiency
            self.eta_upd()

            converged, delta = utils.hasConverged(self.spots, p0, self.config['CellCallTolerance'])
            logger.info(' Iteration %d, mean prob change %f' % (i, delta))

            # replace p0 with the latest probabilities
            p0 = self.spots.parent_cell_prob

            if converged:
                iss_df, gene_df = collect_data(self.cells, self.spots, self.genes, self.single_cell)
                break

            if i == max_iter-1:
                logger.info(' Loop exhausted. Exiting with convergence status: %s' % converged)
        return iss_df, gene_df

    # -------------------------------------------------------------------- #
    def geneCount_upd(self):
        """
        Produces a matrix numCells-by-numGenes where element at position (c,g) keeps the expected
        counts of gene g  in cell c.
        """
        # make an array nS-by-nN and fill it with the spots id
        gene_ids = np.tile(self.spots.gene_id, (self.nN, 1)).T

        # flatten it
        gene_ids = gene_ids.ravel()

        # make corresponding arrays for cell_id and probs
        cell_ids = self.spots.parent_cell_id.ravel()
        probs = self.spots.parent_cell_prob.ravel()

        # make the array to be used as index in the group-by operation
        group_idx = np.vstack((cell_ids, gene_ids))

        # For each cell aggregate the number of spots from the same gene.
        # It will produce an array of size nC-by-nG where the entry at (c,g)
        # is the gene counts of gene g within cell c
        N_cg = npg.aggregate(group_idx, probs, size=(self.nC, self.nG))

        # assert N_cg.sum() == self.spots.data.shape[0], \
        #     "The sum of the background spots and the cell gene counts should be equal to the total number of spots"

        # make output. This part needs to be rewritten
        out = np.zeros([self.nC, self.nG])
        out[1:, :] = N_cg[1:, :]

        # cell at position zero is the background
        self.cells.background_counts = N_cg[0, :]
        # Actual cells are on non-zero positions
        self.cells.geneCount = out

    # -------------------------------------------------------------------- #
    def gamma_upd(self):
        """
         Implements equation (3) of the Qian paper
        """
        cells = self.cells
        cfg = self.config
        dtype = self.config['dtype']
        beta = np.einsum('c, gk -> cgk', cells.cell_props['area_factor'], self.single_cell.mean_expression).astype(dtype) + cfg['rSpot']
        rho = cfg['rSpot'] + cells.geneCount

        self.spots.gamma_bar = self.spots.gammaExpectation(rho, beta)
        self.spots.log_gamma_bar = self.spots.logGammaExpectation(rho, beta)

    # -------------------------------------------------------------------- #
    def cell_to_cellType(self):
        """
        return a an array of size numCells-by-numCellTypes where element in position [i,j]
        keeps the probability that cell i has cell type j
        Implements equation (2) of the Qian paper
        :param spots:
        :param config:
        :return:
        """

        # gene_gamma = self.genes.eta
        dtype = self.config['dtype']
        # ScaledExp = np.einsum('c, g, gk -> cgk', self.cells.alpha, self.genes.eta, sc.mean_expression.data) + self.config['SpotReg']
        ScaledExp = np.einsum('c, g, gk -> cgk', self.cells.cell_props['area_factor'], self.genes.eta, self.single_cell.mean_expression).astype(dtype) + self.config['SpotReg']
        pNegBin = ScaledExp / (self.config['rSpot'] + ScaledExp)
        cgc = self.cells.geneCount
        contr = utils.negBinLoglik(cgc, self.config['rSpot'], pNegBin)
        wCellClass = np.sum(contr, axis=1) + self.cellTypes.log_prior
        pCellClass = utils.softmax(wCellClass, axis=1)

        ## self.cells.classProb = pCellClass
        # logger.info('Cell 0 is classified as %s with prob %4.8f' % (
        #     self.cells.class_names[np.argmax(wCellClass[0, :])], pCellClass[0, np.argmax(wCellClass[0, :])]))
        # logger.info('cell ---> cell class probabilities updated')
        self.cells.classProb = pCellClass
        # return pCellClass

    # -------------------------------------------------------------------- #
    def spots_to_cell(self):
        """
        spot to cell assignment.
        Implements equation (4) of the Qian paper
        """
        # nN = self.config['nNeighbors'] + 1
        nN = self.nN
        nS = self.spots.data.gene_name.shape[0]

        # initialise array with the misread density
        aSpotCell = np.zeros([nS, nN])
        gn = self.spots.data.gene_name.values
        expected_counts = self.single_cell.log_mean_expression.loc[gn].values

        # loop over the first nN-1 closest cells. The nN-th column is reserved for the misreads
        for n in range(nN - 1):
            # get the spots' nth-closest cell
            sn = self.spots.parent_cell_id[:, n]

            # get the respective cell type probabilities
            # cp = self.cells.classProb.sel({'cell_id': sn}).data
            cp = self.cells.classProb[sn]

            # multiply and sum over cells
            # term_1 = (expected_spot_counts * cp).sum(axis=1)
            term_1 = np.einsum('ij, ij -> i', expected_counts, cp)

            # logger.info('genes.spotNo should be something like spots.geneNo instead!!')
            expectedLog = self.spots.log_gamma_bar[self.spots.parent_cell_id[:, n], self.spots.gene_id]
            # expectedLog = utils.bi2(self.elgamma.data, [nS, nK], sn[:, None], self.spots.data.gene_id.values[:, None])

            term_2 = np.einsum('ij, ij -> i', cp, expectedLog)  # same as np.sum(cp * expectedLog, axis=1) but bit faster
            aSpotCell[:, n] = term_1 + term_2
        wSpotCell = aSpotCell + self.spots.loglik(self.cells, self.config)

        # update the prob a spot belongs to a neighboring cell
        pSpotNeighb = utils.softmax(wSpotCell, axis=1)
        self.spots.parent_cell_prob = pSpotNeighb
        # Since the spot-to-cell assignments changed you need to update the gene counts now
        self.geneCount_upd()
        # self.spots.update_cell_prob(pSpotNeighb, self.cells)
        # logger.info('spot ---> cell probabilities updated')

    # -------------------------------------------------------------------- #
    def eta_upd(self):
        """
        Calcs the expected eta
        Implements equation (5) of the Qian paper
        """
        grand_total = self.cells.background_counts.sum() + self.cells.total_counts.sum()
        assert round(grand_total) == self.spots.data.shape[0], \
            'The sum of the background spots and the total gene counts should be equal to the number of spots'

        classProb = self.cells.classProb
        mu = self.single_cell.mean_expression + self.config['SpotReg']
        area_factor = self.cells.cell_props['area_factor']
        gamma_bar = self.spots.gamma_bar

        zero_prob = classProb[:, -1]  # probability a cell being a zero expressing cell
        zero_class_counts = self.spots.zero_class_counts(self.spots.gene_id, zero_prob)

        # Calcs the sum in the Gamma distribution (equation 5). The zero class
        # is excluded from the sum, hence the arrays in the einsum below stop at :-1
        # Note. I also think we should exclude the "cell" that is meant to keep the
        # misreads, ie exclude the background
        class_total_counts = np.einsum('ck, gk, c, cgk -> g',
                                       classProb[:, :-1],
                                       mu.values[:, :-1],
                                       area_factor,
                                       gamma_bar[:, :, :-1])
        background_counts = self.cells.background_counts
        numer = self.config['rGene'] + self.spots.counts_per_gene - background_counts - zero_class_counts
        denom = self.config['rGene'] + class_total_counts
        res = numer / denom

        # Finally, update gene_gamma
        self.genes.eta = res







