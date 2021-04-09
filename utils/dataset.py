import pandas as pd
from FISHscale.visualization import Window
from PyQt5 import QtWidgets
import sys

from sklearn.cluster import DBSCAN, MiniBatchKMeans, Birch, SpectralClustering
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np
import loompy

class PandasDataset:
    """
    Base Class for FISHscale, still under development

    Add methods to the class to run segmentation, make loom files, visualize (spatial localization, UMAP, TSNE).
    """

    def __init__(self,
        data: pd.DataFrame,
        x: str= 'r_px_microscope_stitched',
        y: str='c_px_microscope_stitched',
        gene_column: str='below3Hdistance_genes',
        other_columns: list = None):

        self.x,self.y = x,y
        self.data = data
        self.gene_column = gene_column


    def visualize(self,columns=[],width=2000,height=2000,color_dic=None):
        """
        Run open3d visualization on self.data

        Pass to columns a list of strings, each string will be the class of the dots to be colored by. example: color by gene
        """


        QtWidgets.QApplication.setStyle('Fusion')
        App = QtWidgets.QApplication.instance()
        if App is None:
            App = QtWidgets.QApplication(sys.argv)
        else:
            print('QApplication instance already exists: %s' % str(App))

        window = Window(self.data,[self.gene_column]+columns,width,height,color_dic) 
        App.exec_()
        App.quit()

    def DBsegment(self,eps=25,min_samples=5,column_name='cell'):
        """
        Run DBscan segmentation on self.data, this will reassign a column on self.data with column_name
        """

        print('Running DBscan segmentation')
        X = self.data.loc[:,[self.x,self.y]].values
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        self.labels = clustering.labels_
        self.data[column_name] = self.labels
        print('DBscan assigned {}'.format(self.labels.max()))

    def make_loom(self,
        filename:str,
        save=True):
        """
        Will generate the gene expression matrix and save it as loom file
        """


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
        
        if save:
            loompy.create(filename, data.values, rowattrs, colattrs)
            print('Loom File Created')
    

            

