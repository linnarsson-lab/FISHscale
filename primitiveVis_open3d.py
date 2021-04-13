from PyQt5.QtWidgets import * 
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import sys 
import open3d as o3d
import pandas as pd
import numpy as np
#import laspy as lp
import random


class Window(QMainWindow): 

    def __init__(self,dataframe,width=2000,height=2000,color_dic=None): 
        
        super().__init__() 
        
        """
        GUI for Open3D Make Plots Fast Again
        
        dataframe: Pass the pandas dataframe, column names must be 'c_px_microscope_stitched','r_px_microscope_stitched' and gene
        color_dic: pass dictionary of desired color in RGB for each unique gene in the parquet_file
        
        """
        # setting title 
        self.setWindowTitle("Primitive Visualizer Open3D ") 
        
        self.dataframe = dataframe
        self.color_dic = color_dic
        
        self.dic_pointclouds = self.pass_data()

        # setting geometry 
        self.setGeometry(100, 100, 200, 800) 
          
        # calling method 
        self.list_widget = self.UiComponents()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.add_items()
        # showing all the widgets 
        self.opened = False
        
        self.show()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(height=height,width=width,top=0,left=500)
        
        allgenes = np.concatenate([self.dic_pointclouds[i][0] for i in self.dic_pointclouds.keys()])
        allcolors = np.concatenate([self.dic_pointclouds[i][1] for i in self.dic_pointclouds.keys()])
        
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(allgenes)
        self.pcd.colors = o3d.utility.Vector3dVector(allcolors)   
        self.vis.add_geometry(self.pcd)
        
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        
        
        self.list_widget.itemClicked.connect(self.visualizeItems)
        self.vis.run()
        self.vis.destroy_window()

        #self.show()
  
    # method for components 
    def UiComponents(self): 
  
        # creating a QListWidget 
        list_widget = QListWidget(self) 
  
        # setting geometry to it 
        list_widget.setGeometry(0, 0, 175, 775) 
        
        # scroll bar 
        scroll_bar = QScrollBar(self) 
  
        # setting style sheet to the scroll bar 
        scroll_bar.setStyleSheet("background : black;") 
  
        # adding extra scroll bar to it 
        list_widget.addScrollBarWidget(scroll_bar, Qt.AlignLeft)
        return list_widget
  
        # list widget items 
    
    def pass_data(self):
        r = lambda: random.randint(0,255)

        gene_grp = self.dataframe.groupby('gene')
        gene_grp = {g[0]:g[1] for g in gene_grp}

        dic_coords = {}
        for gene in gene_grp:
            coords = gene_grp[gene].loc[:,['c_px_microscope_stitched','r_px_microscope_stitched']].to_numpy()
            coords = np.vstack([coords[:,0],coords[:,1],np.ones(coords.shape[0])]).T
            if self.color_dic == None:
                col = np.array([(r(),r(),r())]*coords.shape[0])/255
                
            else:
                col = np.array([color_dic[gene]]*coords.shape[0])/255
            
            dic_coords[gene] = (coords,col)

        return dic_coords
    
    #def visualize_data(self):
    
    def add_items(self):
        for gene in self.dic_pointclouds:
            i = QListWidgetItem(gene) 
            c = self.dic_pointclouds[gene][1][0,:]*255
            i.setBackground(QColor(c[0],c[1],c[2],120))
            self.list_widget.addItem(i)
        # adding items to the list widget 
        
    def visualizeItems(self):
        items = self.list_widget.selectedItems()
        genes = []
        points, colors = [], [ ]
        
        for i in range(len(items)):
            genes.append(str(self.list_widget.selectedItems()[i].text()))
            ps,cs = self.dic_pointclouds[str(self.list_widget.selectedItems()[i].text())]
            points.append(ps)
            colors.append(cs)
            
        self.pcd.points = o3d.utility.Vector3dVector(np.concatenate(points))
        self.pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors))

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()


        

        
        

