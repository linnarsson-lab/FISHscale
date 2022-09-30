from FISHscale.pciSeq import pciSeq
import loompy
import dask.dataframe as dd
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import shoji

genes = pd.read_parquet('/wsfish/glioblastoma/EEL/codebookHG1_20201124.parquet')['Gene'].dropna().values.tolist()
genes += pd.read_parquet('/wsfish/glioblastoma/EEL/codebookHG2_20210508.parquet')['Gene'].dropna().values.tolist()
genes += pd.read_parquet('/wsfish/glioblastoma/EEL/codebookHG3_20211214.parquet')['Gene'].dropna().values.tolist()
genes = np.array([u for u in np.unique(genes) if u.count('Control') == 0 ])

unspliced, spliced = [], []
for g in genes:
    if g[-1] == 'i':
        unspliced.append(g)
    else:
        spliced.append(g)
unspliced, spliced = np.array(unspliced), np.array(spliced)

class GraphPCI:

    def __init__(self, scRNAseq_path:str) -> None:
        
        with loompy.connect(scRNAseq, 'r') as ds:
            scRNAseq = ds[:,:]
            clusters = ds.ca.Cluster[:]
            gene = ds.ra.Gene[:]
            self.scRNAseq = pd.DataFrame(columns=clusters.astype('str'),data=scRNAseq,index=gene)

    def load_segmentation(self, segmentation_path:str, output_name:str) -> None:
        #os.path.join(self.folder,'/Segmentation/*.parquet')
        #os.path.join(self.folder,"AMEXP20211210_EEL_SL001A_S2_RNA_transformed_assigned_GATsegment.parquet")

        df = dd.read_parquet(segmentation_path).compute()
        df.to_parquet(output_name)
        df= df.rename(columns={"g":'Gene', 'Segmentation':'label'})
        labels = df.label.values
        label, counts= np.unique(df.label.values,return_counts=True)
        to_background = label[np.where(counts <= 3)]
        idxs = np.where(np.isin(df.label.values,to_background))
        labels[idxs] = -1
        df['label'] = labels

        filter_controls = [True if g.count('Control') ==0 else False for g in df.Gene]
        self.graph_df = df[filter_controls]

    def run(self, folder, analysis_name):
        pci = pciSeq.fit(self.graph_df, self.scRNAseq)
        self.cellData, geneData = pci
        os.mkdir(os.path.join(folder,'/GATpciseq'))
        self.cellData.to_parquet(os.path.join(folder,'/GATpciseq/{}pci_celldata.parquet'.format(analysis_name)))
        geneData.to_parquet(os.path.join(folder,'/GATpciseq/{}pci_genedata.parquet').format(analysis_name))

