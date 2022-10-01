
import loompy
import dask.dataframe as dd
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
try:
    import shoji
except ImportError:
    logging.info('Shoji not installed. Please install from')
from FISHscale.graphNN import pciSeq

class GraphPCI:

    def __init__(self, scRNAseq_path:str) -> None:
        
        with loompy.connect(scRNAseq_path, 'r') as ds:
            scRNAseq = ds[:,:]
            self.ref_clusters = ds.ca.Cluster[:]
            self.gene_order = ds.ra.Gene[:]
            self.scRNAseq = pd.DataFrame(columns=self.ref_clusters.astype('str'),data=scRNAseq,index=self.gene_order)

    def load_segmentation(self, segmentation_path:str, output_name:str) -> None:
        df = dd.read_parquet(segmentation_path).compute()
        df= df.rename(columns={"g":'Gene', 'Segmentation':'label'})
        labels = df.label.values
        label, counts= np.unique(df.label.values,return_counts=True)
        to_background = label[np.where(counts <= 3)]
        idxs = np.where(np.isin(df.label.values,to_background))
        labels[idxs] = -1
        df['label'] = labels

        filter_controls = [True if g.count('Control') ==0 else False for g in df.Gene]
        self.graph_df = df[filter_controls]
        df.to_parquet(output_name)

    def run(self, folder, analysis_name):
        pci = pciSeq.fit(self.graph_df, self.scRNAseq, opts={'max_iter': 2,})
        self.cellData, geneData = pci
        os.mkdir(os.path.join(folder,'GATpciseq'))
        self.cellData.to_parquet(os.path.join(folder,'GATpciseq/{}pci_celldata.parquet'.format(analysis_name)))
        geneData.to_parquet(os.path.join(folder,'GATpciseq/{}pci_genedata.parquet').format(analysis_name))

