import sys
from PyQt5.QtWidgets import (QPushButton, QDialog, QTreeWidget,
                            QTreeWidgetItem, QVBoxLayout,
                            QHBoxLayout, QFrame, QLabel,
                            QApplication,QListWidget,QScrollBar)

from PyQt5.QtWidgets import * 
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 

from PyQt5.QtWidgets import (QPushButton, QDialog, QTreeWidget,
                            QTreeWidgetItem, QVBoxLayout,
                            QHBoxLayout, QFrame, QLabel,
                            QApplication,QListWidget,QScrollBar)

from PyQt5.QtWidgets import * 
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 

import sys
import os

from torch_geometric import data 
try:
    import open3d as o3d
except:
    print('Import Error: open3d')
import pandas as pd
import numpy as np
import random
import time, threading
import FISHscale
import functools
from datetime import datetime, timedelta

class Window: 

    def __init__(self,dataset,columns:list=[],width=2000,height=2000,show_axis=False): 
        
        """
        GUI for Open3D Make Plots Fast Again
        
        dataframe: Pass the pandas dataframe, column names must be 'c_px_microscope_stitched','r_px_microscope_stitched' and gene
        color_dic: pass dictionary of desired color in RGB for each unique gene in the parquet_file
        
        """
        
        r = lambda: random.randint(0,255)
        self.columns= columns
        self.dataset = dataset

        self.color_dic = self.dataset.color_dict
        for g in self.dataset.unique_genes:
            if g in self.color_dic:
                pass
            else:
                col = (r()/255,r()/255,r()/255)
                self.color_dic[g] = col

        # setting title 
        #if str(self.dataset.__class__) == str(FISHscale.utils.dataset.Dataset):
        if isinstance(self.dataset,FISHscale.utils.dataset.Dataset):
            print('Single Dataset')
            self.unique_genes = self.dataset.unique_genes
            self.dataset = [dataset]
            self.dic_pointclouds ={'g':self.unique_genes}
            self.dic_pointclouds['File'] = []
            for x in self.dataset:
                self.dic_pointclouds['File'].append(str(x.filename))
            self.pass_multi_data()
            print('Data Loaded')

        #elif str(self.dataset.__class__) == str(FISHscale.utils.dataset.MultiDataset):
        elif isinstance(self.dataset, FISHscale.utils.dataset.MultiDataset):
            print('MultiDataset')
            self.dataset = dataset
            self.dic_pointclouds ={'g':self.dataset.unique_genes}
            self.dic_pointclouds['File'] = []
            for x in self.dataset:
                self.dic_pointclouds['File'].append(str(x.filename))
            self.pass_multi_data()
            print('Data Loaded')

        self.show_axis= show_axis
        self.vis = Visualizer(self.dataset,self.dic_pointclouds, color_dic=self.color_dic,width=2000, height=2000, show_axis=self.show_axis)
        self.collapse = CollapsibleDialog(self.dic_pointclouds,vis=self.vis)
        self.widget_lists = self.collapse.widget_lists
        self.collapse.show()
        self.vis.collapse = self.collapse
    
        for l in self.widget_lists:
            if l.section == 'File':
                l.list_widget.itemSelectionChanged.connect(l.selectionChanged)
                l.list_widget.itemSelectionChanged.connect(self.collapse.possible)
            else:
                l.list_widget.itemSelectionChanged.connect(l.selectionChanged)

        self.collapse.qbutton.clicked.connect(self.quit)
        self.vis.execute()
            
    def quit(self):
        self.collapse.break_loop = True
        self.vis.break_loop = True
        self.vis.visM.destroy_window()

    def pass_multi_data(self):
        r = lambda: random.randint(0,255)
        ds = []
        for dataframe in self.dataset:
            print(dataframe.filename)

            for c in self.columns:
                unique_ca = dataframe.dask_attrs[c].unique().values.compute()
                self.dic_pointclouds[c]= unique_ca
                for ca in unique_ca:
                    if ca in self.color_dic:
                        pass
                    else:
                        col = (r()/255,r()/255,r()/255)
                        self.color_dic[str(ca)] = col

            ds.append(dataframe)
        self.dataset = ds

class Visualizer:
    def __init__(self,data,dic_pointclouds,color_dic,width=2000,height=2000,show_axis=False):
        
        self.data = data
        self.color_dic = color_dic
        self.visM = o3d.visualization.Visualizer()
        self.visM.create_window(height=height,width=width,top=0,left=500)
        self.dic_pointclouds= dic_pointclouds

        points,maxx,minx,maxy,miny= 0,0,0,0,0
        for d in self.data:
            points += d.shape[0]
            Mx,mx = d.x_max, d.x_min
            My,my = d.y_max, d.y_min
            if Mx > maxx:
                maxx = Mx
            if mx < minx:
                minx= mx
            if My > maxy:
                maxy = My
            if my < miny:
                miny= my
                
        if points > 10000000:
            points = 10000000
       
        x= np.random.randint(int(minx),int(maxx),points, dtype='int32')
        y = np.random.randint(int(miny),int(maxy),points, dtype='int32')
        
        z = np.zeros(points, dtype='int32')

        self.allgenes = np.stack([x,y,z]).T
        self.allcolors = np.ones([points,3])*0#np.concatenate(colors)[:,0,:]

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.allgenes)
        self.pcd.colors = o3d.utility.Vector3dVector(self.allcolors)   

        self.visM.add_geometry(self.pcd)
        self.vp('Data loaded')
        opt = self.visM.get_render_option()

        if show_axis:
            opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0, 0, 0])
        self.break_loop = False

    def execute(self):
        self.visM.poll_events()
        self.visM.update_renderer()
        if sys.platform == 'linux':
            QCoreApplication.processEvents()

    def loop_execute(self):
        while True:
            if self.break_loop:
                break
            self.execute()
            time.sleep(0.05)

class SectionExpandButton(QPushButton):
    """a QPushbutton that can expand or collapse its section
    """
    def __init__(self, item, text = "", parent = None):
        super().__init__(text, parent)
        self.section = item
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        """toggle expand/collapse of section by clicking
        """
        if self.section.isExpanded():
            self.section.setExpanded(False)
        else:
            self.section.setExpanded(True)    
            
class ListWidget(QWidget):
    def __init__(self,subdic,section,vis):
        super().__init__()
        
        # creating a QListWidget 
        self.list_widget = QListWidget()
        # scroll bar 
        self.subdic = subdic
        self.section = section
        self.selected = False
        self.vis = vis
        scroll_bar = QScrollBar() 
        # setting style sheet to the scroll bar 
        scroll_bar.setStyleSheet("background : black;") 
        # adding extra scroll bar to it 
        self.list_widget.addScrollBarWidget(scroll_bar, Qt.AlignLeft)
        self.add_items()

        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tissue_selected = [x for x in self.vis.dic_pointclouds['File']]
        
    def add_items(self):
        for e in self.subdic:
            i = QListWidgetItem(str(e)) 
            try:
                c = self.vis.color_dic[str(e)]
                i.setBackground(QColor(c[0]*255,c[1]*255,c[2]*255,120))
            except:
                pass
            self.list_widget.addItem(i)
        # adding items to the list widget '''
    
    def selectionChanged(self):
        self.selected = [i.text() for i in self.list_widget.selectedItems()]
        if self.selected[0] in self.vis.dic_pointclouds['File'] and self.section == 'File':
            self.tissue_selected = [x for x in self.selected if x in self.vis.dic_pointclouds['File']]
        
        if self.section != 'File':
            points,colors = [],[]  
            for d in self.vis.data:
                if d.filename in self.tissue_selected:
                    if self.section == 'g':
                        for g in self.selected:
                            #d = grpg[g]
                            g= str(g)
                            ps = d.get_gene_sample(g, include_z=True, frac=0.1, minimum=2000000)
                            points.append(ps)
                            cs= np.array([[self.vis.color_dic[g]] *(ps.shape[0])])[0,:,:]
                            colors.append(cs)   
                    else:
                        #selected_indices = d.dask_attrs[d.dask_attrs['labels'].isin(self.selected)].index.compute() #.index.compute()
                        selected_features = d.dask_attrs[d.dask_attrs[self.section].isin(self.selected)].compute()
                        ps = d.df.loc[lambda x: x.index.isin(selected_features.index)].loc[:,['x','y','z']].values.compute()
                        cs = np.array([x for x in selected_features[self.section].apply(lambda x: d.color_dict[str(x)])])

                        points.append(ps)
                        colors.append(cs)
                        '''
                        grpg = d.df.groupby(self.section)
                        for g, d in grpg:
                            if str(g) in self.selected:
                                g= str(g)
                                ps = d.loc[:,['x','y','z']].values
                                       
                        '''

            ps,cs = np.concatenate(points), np.concatenate(colors)
            self.vis.pcd.points = o3d.utility.Vector3dVector(ps)
            self.vis.pcd.colors = o3d.utility.Vector3dVector(cs)
            self.vis.visM.update_geometry(self.vis.pcd)
            self.vis.loop_execute()

class CollapsibleDialog(QDialog,QObject):
    """a dialog to which collapsible sections can be added;
    subclass and reimplement define_section() to define sections and
        add them as (title, widget) tuples to self.sections
    """
    def __init__(self,dic,vis):
        super().__init__()
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.vis = vis
        layout = QVBoxLayout()
        layout.addWidget(self.tree)
        self.setLayout(layout)
        self.setGeometry(100, 100, 200, 800) 
        self.tree.setIndentation(0)
        self.dic = dic
        self.widget_lists = []
        self.sections = {}
        self.break_loop = False

        for x in self.dic:
            self.define_section(x)  
        self.add_sections()

        self.qbutton = QPushButton('Quit Visualizer')
        layout.addWidget(self.qbutton)
        
        app_icon = QtGui.QIcon()
        app_icon.addFile('Images/test16x16.png', QtCore.QSize(16,16))
        self.setWindowIcon(app_icon)
        
    def possible(self):
        for x in self.widget_lists:
            if x.section == 'File':
                ts = x.tissue_selected
        for x in self.widget_lists:
            if x.section != 'File':
                x.tissue_selected = ts

    def add_sections(self):
        """adds a collapsible sections for every 
        (title, widget) tuple in self.sections
        """
        for title in self.sections:
            widget = self.sections[title]
            button1 = self.add_button(title)
            section1 = self.add_widget(button1, widget)
            button1.addChild(section1)       

    def define_section(self,title):
        """reimplement this to define all your sections
        and add them as (title, widget) tuples to self.sections
        """
        widget = QFrame(self.tree)
        layout = QHBoxLayout(widget)

        #layout.addWidget(QLabel("Bla"))
        lw = ListWidget(self.dic[title],title,self.vis)
        list_widget = lw.list_widget
        layout.addWidget(list_widget)
        self.sections[title]= widget
        self.widget_lists.append(lw)
        
    def add_button(self, title):
        """creates a QTreeWidgetItem containing a button 
        to expand or collapse its section
        """
        item = QTreeWidgetItem()
        self.tree.addTopLevelItem(item)
        self.tree.setItemWidget(item, 0, SectionExpandButton(item, text = title))
        return item

    def add_widget(self, button, widget):
        """creates a QWidgetItem containing the widget,
        as child of the button-QWidgetItem
        """
        section = QTreeWidgetItem(button)
        section.setDisabled(True)
        self.tree.setItemWidget(section, 0, widget)
        return section
        

        
        

