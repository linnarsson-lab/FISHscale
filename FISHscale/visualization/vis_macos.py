from re import search
import sys
import logging

import sys
import os
try:
    import open3d as o3d
    from open3d.visualization import gui
    import open3d.visualization.rendering as rendering
except:
    logging.info('Import Error: open3d')
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
            logging.info('Single Dataset')
            self.unique_genes = self.dataset.unique_genes
            self.dataset = [dataset]
            self.dic_pointclouds ={'g':self.unique_genes}
            self.dic_pointclouds['File'] = []
            
            for x in self.dataset:
                self.dic_pointclouds['File'].append(str(x.filename))
            self.pass_multi_data()

        #elif str(self.dataset.__class__) == str(FISHscale.utils.dataset.MultiDataset):
        elif isinstance(self.dataset, FISHscale.utils.dataset.MultiDataset):
            logging.info('MultiDataset')
            
            self.dataset = dataset
            self.dic_pointclouds ={'g':self.dataset.unique_genes}
            self.dic_pointclouds['File'] = []
            #self.dataset[0].datasets_names = [self.dataset[0].dataset_name]
            for x in self.dataset:
                self.dic_pointclouds['File'].append(str(x.filename))
            self.pass_multi_data()

        images = [x.image for x in self.dataset]
        self.vis = Visualizer(self.dataset,
                                self.dic_pointclouds, 
                                color_dic=self.color_dic,
                                width=1624, 
                                height=1024, 
                                x_alt=self.x_alt,
                                y_alt=self.y_alt,
                                alt=self.c_alt,
                                image=images,
                                )

    def pass_multi_data(self):
        r = lambda: random.randint(0,255)
        ds = []
        for dataframe in self.dataset:
            logging.info(dataframe.filename)

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
                height=1000,
                show_axis=False,
                x_alt=None,
                y_alt=None,
                alt={},
                image=None,
                ):

        self.data = data
        self.color_dic = color_dic
        self.dic_pointclouds = dic_pointclouds
        self.height = height
        self.width = width
        self.x_alt, self.y_alt, self.alt = x_alt, y_alt, alt
        self.search_genes = []
        self.genes_points_materials = {}
        self.material = rendering.MaterialRecord()
        self.material.base_color = [0.9, 0.9, 0.9, 1.0]
        self.image = image
        self.vis_init()

    
    def vis_init(self):
        self.visM = gui.Application.instance.create_window(
            "EEL Library", self.width, self.height)
        self.point_size = 2
        self.selected = []
        self.tissue_selected = [x for x in self.dic_pointclouds['File']]

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.visM.renderer)
        self._scene.scene.set_background([0, 0, 0, 1])
        self._scene.scene.downsample_threshold = 10000000

        em = self.visM.theme.font_size
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self.depth = self._scene.scene.camera.get_view_matrix()[2,:]

        self._voxel_down = gui.Slider(gui.Slider.INT)#gui.NumberEdit(gui.NumberEdit.Type(50))
        self._voxel_down.set_limits(1, 100)
        self._voxel_down.int_value = 100
        self.voxel_down = 1
        self._voxel_down.set_on_value_changed(self._on_voxel_down)

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

        vd = gui.Label("Voxel Downsample (%)")
        grid.add_child(vd)
        grid.add_child(self._voxel_down)
        grid.add_child(self._plot_all_button)
        grid.add_child(self._clear_all_button)

        self._text_edit_cell = gui.TextEdit()
        self.search_collapse = gui.CollapsableVert("Search", 0,
                                                gui.Margins(em, 0, 0, 0))
        #fileedit_layout = gui.Horiz()
        self.search_collapse.add_child(self._text_edit_cell)
        grid.add_child(self.search_collapse)
        
        self._text_edit_cell.set_on_value_changed(self._text_changed)

        self._show_axis = gui.Checkbox('Show Axes')
        grid.add_child(gui.Label("Show Axes"))
        grid.add_child(self._show_axis)

        self._scene.set_on_key(self.kevent)

        self.g_collapse = gui.CollapsableVert("Genes", 0,
                                                gui.Margins(em, 0, 0, 0))

        self.gene_w = []
        self.gene_list = []
        for e in self.dic_pointclouds['g']:
            self.gene_list.append(str(e))
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

        datasets_names = [d.dataset_name for d in self.data]


        self.images = [o3d.geometry.Image(i) for i in self.image if i is not None]
        
        for i in self.images:
            self._scene.scene.add_geometry(i)

        for d, label in zip(self.data, datasets_names):
            x = d.x_min + 100
            y = d.y_min - 50
            z = 0

            try:
                label = label.split('_EEL_')[-1]
                label = label.split('_S')[0]
            except:
                pass
            l = self._scene.add_3d_label([x,y,z], label)
            l.color = gui.Color(1,1,1,1)
            #l.scale = 100000

    def _on_point_size(self, size):
        self.point_size = size
        self._resize()

    def _on_voxel_down(self, down):
        self.voxel_down = down/100
        #logging.info(self.voxel_down)

    def _on_file_checked(self, is_checked):
        self.tissue_selected = []
        for f in self.file_w:
            if self.file_w[f].checked:
                self.tissue_selected.append(f)

    def _get_zoom(self):
        while True:
            time.sleep(5)
            new_depth = self._scene.scene.camera.get_view_matrix()[2,:]
            dist = np.linalg.norm(new_depth-self.depth)/5
            new_point = self.point_size + dist
            print(new_point)

    def kevent(self,e):
        idx = self.gene_list.index(self.button_selection)
        if e.key == gui.KeyName.UP and e.type == gui.KeyEvent.UP and idx < len(self.gene_w)-1 and idx >= 0:
            self.gene_w[idx -1].is_on = True
            self.gene_w[idx].is_on = False
            self._on_gene_pressed()
        elif e.key == gui.KeyName.DOWN and e.type == gui.KeyEvent.DOWN and idx < len(self.gene_w)-1 and idx >= 0:
            #idx = self.gene_list.index(self.button_selection)
            self.gene_w[idx + 1].is_on = True
            self.gene_w[idx].is_on = False
            self._on_gene_pressed()

        return gui.Widget.EventCallbackResult.IGNORED

    
    def _on_gene_pressed(self):
        self.previous_selection = self.selected
        self.selected = []
        self.section = 'g'
        for g in self.gene_w:
            if g.is_on:
                if g.text not in self.previous_selection:
                    self.button_selection = g.text
                self.selected.append(g.text)
                c = self.color_dic[g.text]
                g.background_color = gui.Color(c[0],c[1],c[2],0.9)
            elif not g.is_on:
                c = self.color_dic[g.text]
                g.background_color = gui.Color(c[0],c[1],c[2],0.1)
                self._remove_geometry(g.text)

        self._text_edit_cell.text_value = ' '.join(self.selected)
        self._text_edit_cell.placeholder_text =  ' '.join(self.selected)#' '.join(self.data[0].unique_genes.tolist())
        self._selection_changed()

    def _remove_geometry(self, remove_gene):
        remove = []
        for g in self.genes_points_materials:
            t, gene_name = g.split('_')
            if gene_name == remove_gene:
                remove.append(g)
        for r in remove:
            self._scene.scene.remove_geometry(r)
            del self.genes_points_materials[r]

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
        
        self._text_edit_cell.placeholder_text = ' '.join(self.selected)
        self._text_edit_cell.text_value = ' '.join(self.selected)
        self._scene.scene.clear_geometry()
        self.genes_points_materials = {}
        self._selection_changed()
        

    def _text_changed(self, path):
        t = path
        t_list = sorted(t.split(' '))
        #print('tchange')

        self.selected = np.unique(np.array(t_list)).tolist()
        #print(self.selected)
        
        self.section = 'g'
        for g in self.gene_w:
            if t_list.count(g.text):
                g.is_on = True
                c = self.color_dic[g.text]
                g.background_color = gui.Color(c[0],c[1],c[2],0.9)
                self.button_selection = g.text

            else:
                g.is_on = False
                c = self.color_dic[g.text]
                g.background_color = gui.Color(c[0],c[1],c[2],0.1)
                self._remove_geometry(g.text)
                
        self.selected = [e for e in self.selected if e!= '' and e in self.gene_list]
        self._text_edit_cell.placeholder_text = ' '.join(self.selected)
        self._text_edit_cell.text_value = ' '.join(self.selected)
        self._selection_changed()
    
    def _on_show_axes(self, show):
        self._scene.scene.show_axes(show)
        
    def _resize(self):
        a= self._scene.scene.camera.get_field_of_view()
        self._scene.scene.clear_geometry()
        
        self._scene.scene.clear_geometry()
        #print([y[0].points for y in self.genes_points_materials.values()])
        for g in self.genes_points_materials:
            pcd,m = self.genes_points_materials[g]
            m.point_size = int(self.point_size)
            self._scene.scene.add_geometry(g, pcd, m)
            self.genes_points_materials[g] = (pcd,m)

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
         
    def _selection_changed(self,extra=None):
        points,colors = [],[]
        add_new = []
        for d in self.data:
            if d.filename in self.tissue_selected:
                if self.section == 'g':

                    for g in self.selected:
                        if g not in self.genes_points_materials:
                            g= str(g)
                            ps = d.get_gene_sample(g, include_z=True, frac=self.voxel_down)
                            points.append(ps.values)
                            idx_tissue = self.dic_pointclouds['File'].index(d.filename)
                            tissue_gene = str(idx_tissue) +'_'+g
                            add_new.append(tissue_gene)
                            colors.append(self.color_dic[g])

                '''elif self.section in self.alt:
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
                    colors.append(cs)'''

        self.previous_selection = {}
        added = [g for g in self.genes_points_materials]

        for g, ps, cs in zip(add_new, points, colors):
            if g in added:
                continue
            else:
                pcd = o3d.geometry.PointCloud()
                mat = rendering.MaterialRecord()
                mat.shader = "defaultLit"
                pcd.points = o3d.utility.Vector3dVector(ps)
                pcd.voxel_down_sample(voxel_size=self.voxel_down)
                mat.base_color = [cs[0],cs[1],cs[2], 1.0]
                mat.point_size = int(self.point_size)

                self._scene.scene.add_geometry(g, pcd, mat)
                self.genes_points_materials[g] = (pcd, mat)

            
        #added = []
        #for g, ps, cs in zip(sel, points, colors):
        #    added.append(g)
        #    f = added.count(g)
        #    g = g+'_'+str(f)
        #    pcd.points = o3d.utility.Vector3dVector(ps)
        #    pcd.voxel_down_sample(voxel_size=self.voxel_down)
        #    mat.base_color = [cs[0],cs[1],cs[2], 1.0]
        #    mat.point_size = int(self.point_size)
        #    #self.previous_selection[g] = [ps, cs]
        #    self._scene.scene.add_geometry(g, pcd, mat)
        
        #print(self._scene.scene.camera.get_view_matrix())
