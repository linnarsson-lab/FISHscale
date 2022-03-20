from re import search
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
try:
    import open3d as o3d
    from open3d.visualization import gui
    import open3d.visualization.rendering as rendering
    import threading
except:
    print('Import Error: open3d')
import pandas as pd
import numpy as np
import random
import time, threading
import FISHscale
import functools
from datetime import datetime, timedelta
from math import ceil
import threading

class Window: 

    def __init__(self,
                dataset,
                columns:list=[],
                width=2000,
                height=2000,
                show_axis=False,
                x_alt=None,
                y_alt=None,
                c_alt={},
                plot_type={}): 
        
        """
        GUI for Open3D Make Plots Fast Again
        
        dataframe: Pass the pandas dataframe, column names must be 'c_px_microscope_stitched','r_px_microscope_stitched' and gene
        color_dic: pass dictionary of desired color in RGB for each unique gene in the parquet_file
        
        
        """
        
        r = lambda: random.randint(0,255)
        self.columns= columns
        self.dataset = dataset
        self.x_alt, self.y_alt, self.c_alt, self.plot_type = x_alt, y_alt, c_alt, plot_type

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

        #elif str(self.dataset.__class__) == str(FISHscale.utils.dataset.MultiDataset):
        elif isinstance(self.dataset, FISHscale.utils.dataset.MultiDataset):
            print('MultiDataset')
            self.dataset = dataset
            self.dic_pointclouds ={'g':self.dataset.unique_genes}
            self.dic_pointclouds['File'] = []
            for x in self.dataset:
                self.dic_pointclouds['File'].append(str(x.filename))
            self.pass_multi_data()

        self.vis = Visualizer(self.dataset,
                                self.dic_pointclouds, 
                                color_dic=self.color_dic,
                                width=1024, 
                                height=768, 
                                x_alt=self.x_alt,
                                y_alt=self.y_alt,
                                alt=self.c_alt,
                                )

    def pass_multi_data(self):
        r = lambda: random.randint(0,255)
        ds = []
        for dataframe in self.dataset:
            print(dataframe.filename)

            for c in self.columns:
                unique_ca = dataframe.dask_attrs[c][c].unique().values.compute()
                self.dic_pointclouds[c]= unique_ca
                for ca in unique_ca:
                    if ca in self.color_dic:
                        pass
                    else:
                        col = (r()/255,r()/255,r()/255)
                        self.color_dic[str(ca)] = col
            ds.append(dataframe)
        
        if self.c_alt != {}:
            for c in self.c_alt:
                #unique_ca = np.unique(self.c_alt[c])
                colors = {}
                for ca in np.unique(self.c_alt[c]):
                    if int(ca) < 0:
                        colors[ca] = (r()/255,r()/255,r()/255)
                    else:
                        colors[ca]=(0,0,0) 
                colors =np.array([colors[point] for point in self.c_alt[c]])
                self.dic_pointclouds[c] = colors
                self.c_alt[c] = colors

        self.dataset = ds

class Visualizer:
    def __init__(self,
                data,
                dic_pointclouds,
                color_dic,
                width=2000,
                height=2000,
                show_axis=False,
                x_alt=None,
                y_alt=None,
                alt={},
                ):

        self.data = data
        self.color_dic = color_dic
        self.dic_pointclouds = dic_pointclouds
        self.height = height
        self.width = width
        self.x_alt, self.y_alt, self.alt = x_alt, y_alt, alt
        self.search_genes = []

        self.material = rendering.MaterialRecord()
        self.material.base_color = [0.9, 0.9, 0.9, 1.0]
        self.vis_init()
    
    def vis_init(self):
        self.visM = gui.Application.instance.create_window(
            "Open3D", self.width, self.height)
        self.point_size = 2
        self.selected = []
        self.tissue_selected = [x for x in self.dic_pointclouds['File']]

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.visM.renderer)
        self._scene.scene.set_background([0, 0, 0, 1])

        em = self.visM.theme.font_size
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        ''' view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))'''

        self._point_size = gui.Slider(gui.Slider.INT)#gui.NumberEdit(gui.NumberEdit.Type(50))
        self._point_size.set_limits(1, 50)
        self._point_size.set_on_value_changed(self._on_point_size)

        self._plot_all_button = gui.Button('Plot all')#gui.NumberEdit(gui.NumberEdit.Type(50))
        self._plot_all_button.set_on_clicked(self._plot_all)

        self._clear_all_button = gui.Button('Clear all')#gui.NumberEdit(gui.NumberEdit.Type(50))
        self._clear_all_button.set_on_clicked(self._clear_all)

        grid = gui.VGrid(1, 0.5 * em)
        ps = gui.Label("Point size")
        grid.add_child(ps)
        grid.add_child(self._point_size)
        grid.add_child(self._plot_all_button)
        grid.add_child(self._clear_all_button)

        self._text_edit_cell = gui.TextEdit()
        self.search_collapse = gui.CollapsableVert("Search", 0,
                                                gui.Margins(em, 0, 0, 0))
        #fileedit_layout = gui.Horiz()
        self.search_collapse.add_child(self._text_edit_cell)
        grid.add_child(self.search_collapse)
        #grid.add_child(self._text_edit_cell)
        self._text_edit_cell.set_on_value_changed(self._text_changed)

        self._show_axis = gui.Checkbox('Show Axes')
        grid.add_child(gui.Label("Show Axes"))
        grid.add_child(self._show_axis)

        self.g_collapse = gui.CollapsableVert("Genes", 0,
                                                gui.Margins(em, 0, 0, 0))

        self.gene_w = []
        for e in self.dic_pointclouds['g']:
            widget = gui.Button(str(e))
            widget.vertical_padding_em = 0.15
            widget.toggleable =  True
            widget.is_on = False
            c = self.color_dic[str(e)]
            widget.background_color= gui.Color(c[0],c[1],c[2],0.1)
            self.g_collapse.add_child(widget)
            self.gene_w.append(widget)
            widget.set_on_clicked(self._on_gene_pressed)

        self.f_collapse = gui.CollapsableVert("Files", 0,
                                        gui.Margins(em, 0, 0, 0))
        
        grid.add_child(self.g_collapse)
        grid.add_child(self.f_collapse)
 
        self.file_w = {}
        for e in self.dic_pointclouds['File']:
            widget = gui.Checkbox(str(e))
            widget.checked = True
            self.f_collapse.add_child(widget)
            self.file_w[e] = widget
            widget.set_on_checked(self._on_file_checked)

        self._show_axis.set_on_checked(self._on_show_axes)
        self._settings_panel.add_child(grid)

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

        bbox = o3d.geometry.AxisAlignedBoundingBox([minx, miny, -100],
                                                   [maxx, maxy, 100])
        self._scene.setup_camera(60, bbox, [0, 0, 0]) 

        self.visM.set_on_layout(self._on_layout)

        self.visM.add_child(self._scene)  
        self.visM.add_child(self._settings_panel) 
    
    def _on_point_size(self, size):
        self.point_size = size
        self._resize()

    def _on_file_checked(self, is_checked):
        self.tissue_selected = []
        for f in self.file_w:
            if self.file_w[f].checked:
                self.tissue_selected.append(f)
    
    def _on_gene_pressed(self):
        self.selected = []
        self.section = 'g'
        for g in self.gene_w:
            if g.is_on:
                self.selected.append(g.text)
                c = self.color_dic[g.text]
                g.background_color = gui.Color(c[0],c[1],c[2],0.9)
            elif not g.is_on:
                c = self.color_dic[g.text]
                g.background_color = gui.Color(c[0],c[1],c[2],0.1)

        self._text_edit_cell.text_value = ' '.join(self.selected)
        self._text_edit_cell.placeholder_text =  ' '.join(self.selected)#' '.join(self.data[0].unique_genes.tolist())
        self._selection_changed()

    def _plot_all(self):
        self.selected = []
        self.section = 'g'
        for g in self.gene_w:
            g.is_on = True
            self.selected.append(g.text)
            c = self.color_dic[g.text]
            g.background_color = gui.Color(c[0],c[1],c[2],0.9)
            self.selected.append(g.text)
        self._selection_changed()

    def _clear_all(self):
        self.selected = []
        self.section = 'g'
        for g in self.gene_w:
            g.is_on = False
            c = self.color_dic[g.text]
            g.background_color = gui.Color(c[0],c[1],c[2],0.1)
        self._selection_changed()

    def _text_changed(self, path):
        t = path
        t_list = sorted(t.split(' '))
        t_list += self.selected
        self.selected = np.unique(np.array(t_list)).tolist()
        self._text_edit_cell.placeholder_text = ' '.join(self.selected)
        self._text_edit_cell.text_value = ' '.join(self.selected)
        self.section = 'g'
        for g in self.gene_w:
            if t.count(g.text) :
                c = self.color_dic[g.text]
                g.background_color = gui.Color(c[0],c[1],c[2],0.9)

        self._selection_changed()
    
    def _on_show_axes(self, show):
        self._scene.scene.show_axes(show)
        
    def _resize(self):
        self._scene.scene.clear_geometry()
        
        pcd = o3d.geometry.PointCloud()
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        
        for g in self.previous_selection:
            ps, cs= self.previous_selection[g][0], self.previous_selection[g][1]
            pcd.points = o3d.utility.Vector3dVector(ps)
            mat.point_size = int(self.point_size)
            mat.base_color = [cs[0],cs[1],cs[2], 1.0]
            self._scene.scene.add_geometry(g, pcd, mat)

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.visM.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

        '''self._text_edit_cell.frame = gui.Rect(r.get_right() - width, r.y, width,
                                                height/3)'''
    
    def _selection_changed(self,extra=None):
        points,colors = [],[]  
        for d in self.data:
            if d.filename in self.tissue_selected:
                if self.section == 'g':
                    for g in self.selected:
                        g= str(g)
                        ps = d.get_gene_sample(g, include_z=True, frac=0.1, minimum=2000000)
                        points.append(ps.values)
                        colors.append(self.color_dic[g])

                elif self.section in self.alt:
                    ps = np.array([self.x_alt, self.y_alt, np.zeros_like(self.x_alt)]).T
                    #cs = np.array([d.color_dict[str(x)] for x in selected_features])
                    cs = self.alt[self.section]
                    points.append(ps)
                    colors.append(cs)

                else:
                    da = d.dask_attrs[self.section]
                    selected = da[da[self.section].isin(self.selected)].compute() #.index.compute(
                    ps =  selected.loc[:,['x','y','z']].values
                    cs = np.array([x for x in selected[self.section].apply(lambda x: d.color_dict[str(x)])])
                    points.append(ps)
                    colors.append(cs)

        self._scene.scene.clear_geometry()
        self.previous_selection = {}
        pcd = o3d.geometry.PointCloud()
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        for g, ps, cs in zip(self.selected, points, colors):
            pcd.points = o3d.utility.Vector3dVector(ps)
            mat.base_color = [cs[0],cs[1],cs[2], 1.0]
            mat.point_size = int(self.point_size)
            self.previous_selection[g] = [ps, cs]
            self._scene.scene.add_geometry(g, pcd, mat)
