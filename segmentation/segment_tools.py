from sklearn.cluster import DBSCAN, MiniBatchKMeans, Birch, SpectralClustering
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np
import loompy
from FISHscale.utils.dataset import PandasDataset
#DBscan

class DBsegment(PandasDataset):
    def __init__(self,

        eps=30,
        min_samples=5):

        self.x,self.y = x,y
        self.data = data
        self.eps=eps
        self.min_samples=min_samples
        self.gene_column = gene_column

        print('Running DBscan')
        self.run()

    def assign_dots(self):
        X = self.data.loc[:,[self.x,self.y]].values
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        self.labels = clustering.labels_
        self.data['cell'] = self.labels

        print('DBscan assigned {}'.format(self.labels.max()))

    def make_loom(self,
        filename:str):

        cell = {x[0]:x[1] for x in self.data.groupby('cell')}
        genes = np.unique(self.data[self.gene_column])

        c0 = Counter({x:0 for x in genes})
        gene_cell = []

        with tqdm(total=self.labels.max()) as pbar:
            for x in cell:
                generow = cell[x][self.gene_column]
                #print(generow)
                c1 = Counter(generow)
                v = []
                for x in c0:  
                    if x in c1:
                        v.append(c1[x])
                    else:
                        v.append(0)
                gene_cell.append(v)
                pbar.update(1)
                # Updates in increments of 10 stops at 100

        print('Assembling Gene by Cell Matrix')
                
        df = np.stack(gene_cell).T
        y = df.sum(axis=0)
        data = pd.DataFrame(data=df,index=genes)

        print('Creating Loom File')

        grpcell  = self.data.groupby('cell')
        centroids = [[(cell[1][self.y].min()+ cell[1][self.y].max())/2,(cell[1][self.x].min()+ cell[1][self.x].max())/2] for cell in grpcell]
        max_l = [np.array([abs(cell[1][self.y].min() - cell[1][self.y].max()),abs(cell[1][self.x].min()- cell[1][self.x].max())]) for cell in grpcell]
        colattrs = {'cell':np.array([cell[0] for cell in grpcell]),'cell_xy':np.array(centroids),'cell_length':np.array(max_l)}
        rowattrs = {'gene':data.index.values}

        loompy.create(filename, data.values, rowattrs, colattrs)
        print('Loom File Created')
    