"""
hyperparameters for the pciSeq method
"""
import numpy as np

DEFAULT = {
    # list of genes to be excluded during cell-typing, e.g ['Aldoc', 'Id2'] to exclude all spots from Aldoc and Id2
    'exclude_genes': [],

    # Maximum number of loops allowed for the Variational Bayes to run
    'max_iter': 1000,

    # Convergence achieved if assignment probabilities between two successive loops is less than the tolerance
    'CellCallTolerance': 0.02,

    # A gamma distribution expresses the efficiency of the in-situ sequencing for each gene. It tries to capture
    # the ratio of the observed over the theoretical counts for a given gene. rGene controls the variance and
    # Inefficiency is the average of this assumed Gamma distribution
    'rGene': 20,
    'Inefficiency': 0.2,

    # If a spot is inside the cell boundaries this bonus will give the likelihood an extra boost
    # in order to make the spot more probable to get assigned to the cell than another spot positioned
    # outside the cell boundaries
    'InsideCellBonus': 2,

    # To account for spots far from the some a uniform distribution is introduced to describe those misreads.
    # By default this uniform distribution has a density of 1e-5 misreads per pixel.
    'MisreadDensity': 0.00001,

    # Gene detection might come with irregularities due to technical errors. A small value is introduced
    # here to account for these errors. It is an additive factor, applied to the single cell expression
    # counts when the mean counts per class and per gene are calculated.
    'SpotReg': 0.1,

    # By default only the 3 nearest cells will be considered as possible parent cells for any given spot.
    # There is also one extra 'super-neighbor', which is always a neighbor to the spots so we can assign
    # the misreads to. Could be seen as the background. Hence, by default the algorithm tries examines
    # whether any of the 3 nearest cells is a possible parent cell to a given cell or whether the spot is
    # a misread
    
    #'nNeighbors': 3,
    'nNeighbors': 5,


    # A gamma distributed variate from Gamma(rSpot, 1) is applied to the mean expression, hence the counts
    # are distributed according to a Negative Binomial distribution.
    # The value for rSpot will control the variance/dispersion of the counts
    'rSpot': 2,

    # Boolean, if True the output will be saved as tsv files in a folder named 'pciSeq' in your system's temp dir.
    'save_data': True,

    # Use either np.float16 or np.float32 to reduce memory usage. In most cases RAM consumption shouldnt
    # need more than 32Gb RAM. If you have a dataset from a full coronal mouse slice with a high number of
    # segmented cells (around 150,000) a gene panel of more than 250 genes and 100 or more different
    # cell types (aka clusters, aka classes) in the single cell data then you might need at least 64GB on
    # your machine. Changing the datatype to a float16 or float32 will help keeping RAM usage to a lower
    # level
    'dtype': np.float32,
}

