import pandas as pd
from FISHscale.visualization import Window
from PyQt5 import QtWidgets
import sys

class PandasDataset:
    def __init__(self,
        data: pd.DataFrame,
        x: str,
        y: str,
        gene_column: str,
        other_columns: list = None):

        self.x,self.y = x,y
        self.data = data
        self.gene_column = gene_column

    def visualize(self,columns=[],width=2000,height=2000,color_dic=None):
        QtWidgets.QApplication.setStyle('Fusion')
        App = QtWidgets.QApplication.instance()
        if App is None:
            App = QtWidgets.QApplication(sys.argv)
        else:
            print('QApplication instance already exists: %s' % str(App))

        window = Window(self.data,[self.gene_column]+columns,width,height,color_dic) 
        App.exec_()
        App.quit()

            

