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

from torch_geometric import data 
try:
    import open3d as o3d
except:
    print('Import Error: open3d')
import pandas as pd
import numpy as np
import random
import time
import FISHscale
import functools

class Window: 

    def __init__(self,dataset,columns:list=[],width=2000,height=2000,show_axis=False,color_dic={}): 
        
        super().__init__() 
        
        """
        GUI for Open3D Make Plots Fast Again
        
        dataframe: Pass the pandas dataframe, column names must be 'c_px_microscope_stitched','r_px_microscope_stitched' and gene
        color_dic: pass dictionary of desired color in RGB for each unique gene in the parquet_file
        
        """
        
        r = lambda: random.randint(0,255)
        self.color_dic = color_dic
        self.columns= columns
        self.dataset = dataset
        for g in self.dataset.unique_genes:
            if g in self.color_dic:
                pass
            else:
                col = [(r()/255,r()/255,r()/255)]
                self.color_dic[g] = col

        # setting title 
        if str(self.dataset.__class__) == str(FISHscale.utils.dataset.Dataset):
            print('Single Dataset')
            self.gene_label,self.x_label,self.y_label,self.unique_genes = self.dataset.gene_label,self.dataset.x_label, self.dataset.y_label,self.dataset.unique_genes
            self.dataset = [dataset]

            self.dic_pointclouds ={self.gene_label:self.unique_genes}
            self.dic_pointclouds['File'] = []
            for x in self.dataset:
                self.dic_pointclouds['File'].append(str(x.filename))

            self.pass_multi_data()
            print('Data Loaded')

        elif str(self.dataset.__class__) == str(FISHscale.utils.dataset.MultiDataset):
            print('MultiDataset')
            self.dataset = dataset
            self.gene_label,self.x_label,self.y_label = self.dataset.gene_label,self.dataset.x_label, self.dataset.y_label
            self.dic_pointclouds ={self.dataset.gene_label:self.dataset.unique_genes}

            self.dic_pointclouds['File'] = []
            for x in self.dataset:
                self.dic_pointclouds['File'].append(str(x.filename))

            self.pass_multi_data()
            print('Data Loaded')
        

        self.show_axis= show_axis
        self.vis = Visualizer(self.dataset,self.dic_pointclouds, x_label=self.x_label,y_label=self.y_label,gene_label=self.gene_label,
            color_dic=self.color_dic,width=2000, height=2000, show_axis=self.show_axis)
        self.collapse = CollapsibleDialog(self.dic_pointclouds,vis=self.vis)
        self.widget_lists = self.collapse.widget_lists
        self.collapse.show()
        
        for l in self.widget_lists:
            if l.section == 'File':
                l.list_widget.itemSelectionChanged.connect(l.selectionChanged)
                l.list_widget.itemSelectionChanged.connect(self.collapse.possible)
            else:
                l.list_widget.itemSelectionChanged.connect(l.selectionChanged)

        self.collapse.qbutton.clicked.connect(self.collapse.quit)

        while self.collapse.break_loop == False:
            self.vis.execute()

    #@functools.lru_cache
    def pass_multi_data(self):
        r = lambda: random.randint(0,255)
        ds = []
        for dataframe in self.dataset:
            print(dataframe.filename)
            if dataframe.gene_label != self.gene_label:
                dataframe.gene_label = self.gene_label
            pd = dataframe.make_pandas()
            pd['z_label'] = np.array([dataframe.z_offset]*pd.shape[0])

            for c in self.columns:
                colattr = getattr(dataframe,c)
                pd[c] = colattr
                unique_ca = np.unique(colattr)
                self.dic_pointclouds[c]= unique_ca
                for ca in unique_ca:
                    if ca in self.color_dic:
                        pass
                    else:
                        col = [(r()/255,r()/255,r()/255)]
                        self.color_dic[str(ca)] = col

            preloaded = {g:d for g,d in pd.groupby(dataframe.gene_label)}
            ds.append((pd,dataframe.filename,preloaded))

        self.dataset = ds

class Visualizer:
    def __init__(self,data,dic_pointclouds,x_label,y_label,gene_label,color_dic,width=2000,height=2000,show_axis=False):
        
        self.data = data
        self.x_label,self.y_label,self.gene_label = x_label,y_label,gene_label
        self.color_dic = color_dic
        self.visM = o3d.visualization.Visualizer()
        self.visM.create_window(height=height,width=width,top=0,left=500)
        
        self.dic_pointclouds= dic_pointclouds

        #points, colors = [], []
        points,maxx,minx,maxy,miny= 0,0,0,0,0
        for d,f,g in self.data: 
            points+= d.shape[0]
            Mx,mx = d.loc[:,[self.x_label]].values.max(),d.loc[:,[self.x_label]].values.max()
            My,my = d.loc[:,[self.y_label]].values.max(),d.loc[:,[self.x_label]].values.max()
            if Mx > maxx:
                maxx = Mx
            if mx < minx:
                minx= mx

            if My > maxy:
                maxy = My
            if my < miny:
                miny= my

            '''            
            ps = d.loc[:,[self.x_label,self.y_label,'z_label']].values
            gs = d.loc[:,[self.gene_label]].values
            cs= np.array([self.color_dic[g[0]] for g in gs])'''

            #points.append(ps)
            #colors.append(cs)

        x= np.random.random_integers(int(minx),int(maxx),points)
        y = np.random.random_integers(int(miny),int(maxy),points)
        z = np.zeros(points)

        self.allgenes = np.stack([x,y,z]).T
        self.allcolors = np.ones([points,3])*0#np.concatenate(colors)[:,0,:]

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.allgenes)
        self.pcd.colors = o3d.utility.Vector3dVector(self.allcolors)   

        self.visM.add_geometry(self.pcd)
        opt = self.visM.get_render_option()

        if show_axis:
            opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0, 0, 0])
    
    def execute(self):
        #self.visM.run()
        #self.visM.destroy_window()
        self.visM.poll_events()
        self.visM.update_renderer()

    def close(self):
        self.visM.destroy_window()

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
                c = self.vis.color_dic[e]
                i.setBackground(QColor(c[0][0]*255,c[0][1]*255,c[0][2]*255,120))

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

            for d,f,grpg in self.vis.data:
                if f in self.tissue_selected:
                    if self.selected == self.vis.gene_label:
                        grpg = grpg
                    else:
                        grpg = d.groupby(self.section)
                    for g, d in grpg:
                        if str(g) in self.selected:

                            g= str(g)
                            ps = d.loc[:,[self.vis.x_label,self.vis.y_label,'z_label']].values
                            cs= np.array([self.vis.color_dic[g] *(ps.shape[0])])[0,:,:]
                            
                            points.append(ps)
                            colors.append(cs)

            ps,cs = np.concatenate(points), np.concatenate(colors)
            self.vis.pcd.points = o3d.utility.Vector3dVector(ps)
            self.vis.pcd.colors = o3d.utility.Vector3dVector(cs)
            self.vis.visM.update_geometry(self.vis.pcd)
            self.vis.visM.poll_events()
            self.vis.visM.update_renderer()


class CollapsibleDialog(QDialog):
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


    def quit(self):
        self.vis.close()
        self.break_loop = True
        

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
        

        
        

