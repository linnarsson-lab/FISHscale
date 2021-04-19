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
import sys 
try:
    import open3d as o3d
except:
    pass
import pandas as pd
import numpy as np
import random
import time


class Window: 

    def __init__(self,pandas_dataset,columns,width=2000,height=2000,show_axis=False,color_dic=None): 
        
        super().__init__() 
        
        """
        GUI for Open3D Make Plots Fast Again
        
        dataframe: Pass the pandas dataframe, column names must be 'c_px_microscope_stitched','r_px_microscope_stitched' and gene
        color_dic: pass dictionary of desired color in RGB for each unique gene in the parquet_file
        
        """
        # setting title 

        self.dataframe = pandas_dataset.data
        self.x_label,self.y_label = pandas_dataset.x, pandas_dataset.y
        self.offset = np.array([pandas_dataset.z_offset, pandas_dataset.x_offset, pandas_dataset.y_offset])
        self.color_dic = color_dic
        self.dic_pointclouds ={c:self.pass_data(c) for c in columns}
        self.show_axis= show_axis
        print('Data Loaded')

        self.vis = Visualizer(self.dic_pointclouds, columns, width=2000, height=2000, show_axis=self.show_axis, color_dic=None)
        self.collapse = CollapsibleDialog(self.dic_pointclouds,vis=self.vis)
        self.widget_lists = self.collapse.widget_lists
        self.collapse.show()
        
        for l in self.widget_lists:
            l.list_widget.itemSelectionChanged.connect(l.selectionChanged)

        self.vis.execute()

    def pass_data(self,column):
        r = lambda: random.randint(0,255)
        gene_grp = self.dataframe.groupby(column)
        gene_grp = {g[0]:g[1] for g in gene_grp}
        dic_coords = {}
        for gene in gene_grp:
            coords = gene_grp[gene].loc[:,[self.x_label,self.y_label]].to_numpy()
            coords = np.vstack([coords[:,0],coords[:,1],np.zeros(coords.shape[0])]).T
            coords = coords + self.offset
            if self.color_dic == None:
                col = np.array([(r(),r(),r())]*coords.shape[0])/255 
            else:
                col = np.array([color_dic[gene]]*coords.shape[0])/255
            dic_coords[str(gene)] = (coords,col)
        return dic_coords
    
class Visualizer:
    def __init__(self,dic_pointclouds,columns,width=2000,height=2000,show_axis=False,color_dic=None):
        
        self.visM = o3d.visualization.Visualizer()
        self.visM.create_window(height=height,width=width,top=0,left=500)
        
        self.dic_pointclouds= dic_pointclouds
        
        self.allgenes = np.concatenate([self.dic_pointclouds[columns[0]][i][0] for i in self.dic_pointclouds[columns[0]].keys()])
        self.allcolors = np.concatenate([self.dic_pointclouds[columns[0]][i][1] for i in self.dic_pointclouds[columns[0]].keys()])
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.allgenes)
        self.pcd.colors = o3d.utility.Vector3dVector(self.allcolors)   
        self.visM.add_geometry(self.pcd)
        opt = self.visM.get_render_option()
        if show_axis:
            opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0, 0, 0])
    
    def execute(self):
        self.visM.run()
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
        
        #self.list_widget.itemClicked(self.on_clicked)
        
    def add_items(self):
        for e in self.subdic:
            i = QListWidgetItem(str(e)) 
            c = self.subdic[e][1][0,:]*255
            i.setBackground(QColor(c[0],c[1],c[2],120))
            self.list_widget.addItem(i)
        # adding items to the list widget '''
        
    def selectionChanged(self):
        self.selected = [i.text() for i in self.list_widget.selectedItems()]

        genes = []
        points, colors = [], [ ]
        for i in self.selected:
            ps,cs = self.vis.dic_pointclouds[self.section][i]
            points.append(ps)
            colors.append(cs)
            
        self.vis.pcd.points = o3d.utility.Vector3dVector(np.concatenate(points))
        self.vis.pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors))
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
        for x in self.dic:
            self.define_section(x)
            
        self.add_sections()
          
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
        

        
        

