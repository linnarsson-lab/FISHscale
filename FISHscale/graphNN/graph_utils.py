import torch as th
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from scipy import sparse
import pytorch_lightning as pl
import os
from typing import Any, Dict
from annoy import AnnoyIndex
import networkx as nx
from tqdm import trange
import dgl
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import loompy
from os import path
import pandas as pd
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, bundle_graph, spread
from sklearn.preprocessing import LabelEncoder
from FISHscale.graphNN.cluster_utils import ClusterCleaner
from FISHscale.graphNN.graph_pci import GraphPCI
import logging
hv.extension('bokeh')


color_alphabet = np.array([
	[240, 163, 255], [0, 117, 220], [153, 63, 0], [76, 0, 92], [0, 92, 49], [43, 206, 72], [255, 204, 153], [128, 128, 128], [148, 255, 181], [143, 124, 0], [157, 204, 0], [194, 0, 136], [0, 51, 128], [255, 164, 5], [255, 168, 187], [66, 102, 0], [255, 0, 16], [94, 241, 242], [0, 153, 143], [224, 255, 102], [116, 10, 255], [153, 0, 0], [255, 255, 128], [255, 255, 0], [255, 80, 5]
]) / 256

colors75 = np.concatenate([color_alphabet, 1 - (1 - color_alphabet) / 2, color_alphabet / 2])
def colorize(x: np.ndarray, *, bgval: Any = None, cmap: np.ndarray = None) -> np.ndarray:
	le = LabelEncoder().fit(x)
	xt = le.transform(x)
	if cmap is None:
		cmap = colors75
	colors = cmap[np.mod(xt, 75), :]
	if bgval is not None:
		colors[x == bgval, :] = np.array([0.8, 0.8, 0.8])
	return colors



class GraphUtils(object):

    def molecules_df(self, filter_molecules=None):
        """
        molecules_df 

        Transforms molecules FISHscale Dataset into a matrix of size 
        (molecules,genes), where contains only a positive value for the gene the
        molecule corresponds to.

        Returns:
            [type]: [description]
        """        
        rows,cols = [],[]
        if type(filter_molecules) == type(None):
            filt = self.data.df.g.values.compute()
        else:
            filt = self.data.df.g.values.compute()[filter_molecules]

        for r in range(self.data.unique_genes.shape[0]):
            g = self.data.unique_genes[r]
            expressed = np.where(filt == g)[0].tolist()
            cols += expressed
            rows += len(expressed)*[r]
        rows = np.array(rows)
        cols = np.array(cols)
        data= np.ones_like(cols)
        sm = sparse.csr_matrix((data.astype(np.uint8),(rows,cols))).T
        return sm
    
    def subsample_xy(self):
        """
        subsample_xy

        Deprecated. Data can be subsampled, but preferably just use instead of
        FISHscale Dataset polygon option to crop the region to run GraphSage on.
        """        
        if type(self.molecules) == type(None):
            self.molecules = np.arange(self.data.shape[0])
        if type(self.subsample) == float and self.subsample < 1:
            self.molecules = np.random.randint(0,self.data.shape[0], int(self.subsample*self.data.shape[0]))
        elif type(self.subsample) == dict:
            filt_x =  ((self.data.df.x > self.subsample['x'][0]) & (self.data.df.x < self.subsample['x'][1])).values.compute()
            filt_y =  ((self.data.df.y > self.subsample['y'][0]) & (self.data.df.y < self.subsample['y'][1])).values.compute()
            self.molecules = self.molecules[filt_x & filt_y]
            #self.molecules = np.random.choice(self.data.df.index.compute(),size=int(subsample*self.data.shape[0]),replace=False)

    def compute_distance_th(self,omega,tau):
        """
        compute_distance_th: deprecated, now inside BuildGraph

        Computes the distance at which 95 percentile molecules are connected to
        at least 2 other molecules. Like PArtel & Wahlby

        Args:
            omega ([type]): [description]
            tau ([type]): [description]
        """        
        if type(tau) == type(None):
            from scipy.spatial import cKDTree as KDTree
            x,y = self.data.df.x.values.compute(),self.data.df.y.values.compute()
            kdT = KDTree(np.array([x,y]).T)
            d,i = kdT.query(np.array([x,y]).T,k=3)
            d_th = np.percentile(d[:,-1],95)*omega
            self.distance_threshold = d_th
            logging.info('Chosen dist to connect molecules into a graph: {}'.format(d_th))
        else:
            self.distance_threshold = tau
            logging.info('Chosen dist to connect molecules into a graph: {}'.format(tau))

    def compute_library_size(self):
        '''data= self.g.ndata['ngh'].T
        sum_counts = data.sum(axis=1)
        masked_log_sum = np.ma.log(sum_counts)
        log_counts = masked_log_sum.filled(0)
        local_mean = (np.mean(log_counts).reshape(-1, 1)).astype(np.float32)
        local_var = (np.var(log_counts).reshape(-1, 1)).astype(np.float32)'''
        return 0,1#local_mean, local_var

    def buildGraph(self, coords=None):
        """
        buildGraph: makes networkx graph.

        Dataset coordinates are turned into a graph based on nearest neighbors.
        The graph will only generate a maximum of ngh_size (defaults to 100) 
        neighbors to avoid having a huge graph in memory and those neighbors at
        a distance below self.distance_threshold*self.distance_factor. 

        Args:
            coords ([type], optional): [description]. Defaults to None.

        Returns:
            dgl.Graph: molecule spatial graph.
        """        
        logging.info('Building graph...')
        if type(coords)  == type(None):
            supervised = False
            edge_file = os.path.join(self.save_to,'graph/DGL-Edges-{}Nodes-dst{}'.format(self.molecules.shape[0],self.distance_factor))
            tree_file = os.path.join(self.save_to,'graph/DGL-Tree-{}Nodes-dst{}.ann'.format(self.molecules.shape[0],self.distance_factor))
            coords = np.array([self.data.df.x.values.compute()[self.molecules], self.data.df.y.values.compute()[self.molecules]]).T
            neighborhood_size = self.ngh_size
        else:
            supervised=True
            edge_file = os.path.join(self.save_to,'graph/DGL-Supervised-Edges-{}Nodes-dst{}'.format(coords.shape[0],self.distance_factor))
            tree_file = os.path.join(self.save_to,'graph/DGL-Supervised-Tree-{}Nodes-dst{}.ann'.format(coords.shape[0],self.distance_factor))
            neighborhood_size = self.ngh_size

        t = AnnoyIndex(2, 'euclidean')  # Length of item vector that will be indexed
        for i in trange(coords.shape[0]):
            v = coords[i,:]
            t.add_item(i, v)

        t.build(5) # 10 trees
        t.save(tree_file)

        #subs_coords = np.random.choice(np.arange(coords.shape[0]),500000,replace=False)
        dists = np.array([t.get_nns_by_item(i, 2,include_distances=True)[1][1] for i in range(coords.shape[0])])
        d_th = np.percentile(dists[np.isnan(dists) == False],97)*self.distance_factor
        self.distance_threshold = d_th
        logging.info('Chosen dist: {}'.format(self.distance_threshold))
        
        def find_nn_distance(coords,tree,distance):
            logging.info('Find neighbors below distance: {}'.format(d_th))
            res,nodes,ngh_, ncoords = [],[],[], []
            for i in trange(coords.shape[0]):
                # 100 sets the number of neighbors to find for each node
                #  it is set to 100 since we usually will compute neighbors
                #  [20,10]
                search = tree.get_nns_by_item(i, neighborhood_size, include_distances=True)
                pair = []
                n_ = []
                for n,d in zip(search[0],search[1]):
                    if d < distance:
                        pair.append((i,n))
                        n_.append(n)
                ngh_.append(len(n_))
                #search = tree.get_nns_by_item(i, neighborhood_size)
                #pair = [(i,n) for n in search]

                add_node = 0
                if len(pair) > self.minimum_nodes_connected:
                    res += pair
                    add_node += 1
                if add_node:
                    nodes.append(i)
                else: 
                    res += [(i,i)] # Add node to itself to prevent errors
                    nodes.append(i)

            res= th.tensor(np.array(res)).T
            nodes = th.tensor(np.array(nodes))
            
            return res,nodes,ngh_

        d = self.molecules_df()
        edges, molecules,ngh_ = find_nn_distance(coords,t,self.distance_threshold)
        d= d[molecules,:]
        #d = self.molecules_df(molecules)
        g= dgl.graph((edges[0,:],edges[1,:]),)
        #g = dgl.to_bidirected(g)]
        g.ndata['gene'] = th.tensor(d.toarray(), dtype=th.uint8)#[molecules_id.numpy(),:]
        #nghs = []
        #for n in tqdm(ngh_):
        #    nghs.append(th.tensor(g.ndata['gene'][n,:].sum(axis=0),dtype=th.uint8).numpy())
        #nghs = np.array(ngh_,dtype=th.uint8)
        #g.ndata['ngh'] = nghs

        if self.smooth:
            #self.g.ndata['zero'] = th.zeros_like(self.g.ndata['gene'])
            #self.g.update_all(fn.u_add_v('gene','zero','e'),fn.sum('e','zero'))
            #self.g.ndata['gene'] = self.g.ndata['zero'] + self.g.ndata['gene']
            #del self.g.ndata['zero']
            g.ndata['gene'] = nghs

        '''sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            th.arange(g.num_nodes()),#.to(g.device),
            sampler,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0)

        ngh_ = []
        for _, _, blocks in tqdm(dataloader):
            ngh_.append(blocks[0].srcdata['gene'].sum(axis=0))
        ngh_ = th.stack(ngh_)
        g.ndata['ngh'] = ngh_'''
        # Finding fastest algorithm
        '''
        g.ndata['zero'] = th.zeros_like(g.ndata['gene'])
        g.update_all(fn.u_add_v('gene','zero','e'),fn.sum('e','zero'))
        g.ndata['gene'] = g.ndata['gene']
        g.ndata['ngh'] = g.ndata['zero'] + g.ndata['gene']
        del g.ndata['zero']
        '''
        sum_nodes_connected = th.tensor(np.array(ngh_,dtype=np.uint8))
        print('sum nodes' , sum_nodes_connected.shape , sum_nodes_connected.max())
        molecules_connected = molecules[sum_nodes_connected >= self.minimum_nodes_connected]
        remove = molecules[sum_nodes_connected < self.minimum_nodes_connected]
        g.remove_nodes(th.tensor(remove))
        g.ndata['indices'] = th.tensor(molecules_connected)
        g.ndata['coords'] = th.tensor(coords[molecules_connected])
        return g

    def prepare_reference(self):
        """
        prepare_reference: reference matrix for semi-supervised learning.

        Wraps the celltype by gene expression matrix and prepares it for the
        model. Sorts the gene list as the one in Dataset.unique_genes. Must have
        the number of cell per cluster (NCells) and the clusternames.
        """        
        if type(self.ref_celltypes) != type(None):
            self.supervised = True
            import loompy
            with loompy.connect(self.ref_celltypes,'r') as ds:
                logging.info(ds.ca.keys())
                try:
                    k = list(self.exclude_clusters.keys())[0]
                    v = self.exclude_clusters[k]
                    region_filt = np.isin(ds.ca[k], v, invert=True)
                    self.ClusterNames = ds.ca[k][region_filt]
                    logging.info('Selected clusters: {}'.format(self.ClusterNames))
                except:
                    self.ClusterNames = ds.ca[k]

                genes = ds.ra.Gene
                order = []
                for x in self.data.unique_genes:
                    try:
                        order.append(np.where(genes==x)[0].tolist()[0])
                    except:
                        pass
                self.ncells = ds.ca['NCells'][region_filt]
                ref = ds[:,:]
                ref = ref[order]
                ref = ref[:,region_filt]
                self.ref_celltypes = ref
                logging.info('Reference dataset shape: {}'.format(self.ref_celltypes.shape))
            
            if self.celltype_distribution == 'uniform':
                dist = th.ones(self.ncells.shape[0])
                self.dist = dist/dist.sum()
            elif self.celltype_distribution == 'ascending':
                n = self.ncells.reshape(-1,1)
                gm = GaussianMixture(n_components=int(n.shape[0]/2.5), random_state=42).fit(n)
                dist = gm.predict(n)
                self.dist = th.tensor(dist/dist.sum(),dtype=th.float32)
            elif self.celltype_distribution == 'molecules':
                dist = self.ncells*self.ref_celltypes.sum(axis=0)
                self.dist = th.tensor(dist/dist.sum(),dtype=th.float32)
            elif self.celltype_distribution == 'cells':
                self.dist = th.tensor(self.ncells/self.ncells.sum(),dtype=th.float32)
            else:
                self.dist = None

        else:
            self.supervised = False
            self.ref_celltypes = np.array([[0],[0]])
            self.ncells = 0
            self.dist=None


class GraphPlotting:
    def analyze(
                self,
                random_n=250000, 
                n_clusters=100, 
                eps=20, 
                min_samples=10,
                multigraph_classifier=None, #file to classifier
                pci_file=None,
                ):
        import umap
        import matplotlib.pyplot as plt

        reducer = umap.UMAP(
                n_neighbors=15,
                n_components=3,
                n_epochs=250,
                init='spectral',
                metric='euclidean',
                min_dist=0.1,
                spread=1,
                random_state=1,
                verbose=True,
                n_jobs= 4
            )

        if self.model.supervised:

            some = np.random.choice(np.arange(self.latent_unlabelled.shape[0]),random_n,replace=False)
            embedding = reducer.fit_transform(self.latent_unlabelled[some])
            #umap_embedding = reducer.fit(self.latent_unlabelled[some])
            #embedding = umap_embedding.transform(self.latent_unlabelled)
            Y_umap = embedding
            Y_umap -= np.min(Y_umap, axis=0)
            Y_umap /= np.max(Y_umap, axis=0)

            fig=plt.figure(figsize=(7,4),dpi=500)
            cycled = [0,2,1,0]
            for i in range(3):
                plt.subplot(1,3,i+1)
                plt.scatter(Y_umap[:,cycled[i]], Y_umap[:,cycled[i+1]], c=Y_umap,  s=0.5, marker='.', linewidths=0, edgecolors=None)
                plt.xlabel("Y"+str(cycled[i]))
                plt.ylabel("Y"+str(cycled[i+1]))
            plt.tight_layout()
            plt.savefig("{}/umap.png".format(self.folder), bbox_inches='tight', dpi=500)

            fig=plt.figure(figsize=(2,2),dpi=1000,)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_facecolor("black")
            width_cutoff = 1640 # um
            #plt.scatter(DS.df.x.values.compute()[self.cells], DS.df.y.values.compute()[self.cells], c=th.argmax(pred.softmax(dim=-1),dim=-1).numpy(), s=0.2,marker='.',linewidths=0, edgecolors=None,cmap='rainbow')
            plt.scatter(self.data.df.x.values.compute()[molecules_id.numpy()][some], self.data.df.y.values.compute()[self.g.ndata['indices'].numpy()][some], c=Y_umap, s=0.05,marker='.',linewidths=0, edgecolors=None)
            plt.xticks(fontsize=2)
            plt.yticks(fontsize=2)
            plt.axis('scaled')
            plt.savefig("{}/spatial_umap_embedding.png".format(self.folder), bbox_inches='tight', dpi=5000)

            clusters= self.prediction_unlabelled.argsort(axis=-1)[:,-1]
            import random
            r = lambda: random.randint(0,255)
            colors = colorize(np.arange(self.ClusterNames.shape[0]))
            color_dic = {}
            for x in range(self.ClusterNames.shape[0]):
                c = colors[x,:].tolist()
                color_dic[x] =  (c[0],c[1],c[2])
            clusters_colors = np.array([color_dic[x] for x in clusters])

            fig=plt.figure(figsize=(7,4),dpi=500)
            cycled = [0,1,2,0]
            for i in range(3):
                plt.subplot(1,3,i+1)
                plt.scatter(Y_umap[:,cycled[i]], Y_umap[:,cycled[i+1]], c=clusters_colors[some],  s=2, marker='.', linewidths=0, edgecolors=None,cmap='rainbow')
                plt.xlabel("Y"+str(cycled[i]))
                plt.ylabel("Y"+str(cycled[i+1]))
                plt.xticks(fontsize=2)
                plt.yticks(fontsize=2)
            plt.tight_layout()
            plt.savefig("{}/umap_clusters.png".format(self.folder), bbox_inches='tight', dpi=500)

            fig=plt.figure(figsize=(6,6),dpi=1000,)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_facecolor("black")
            width_cutoff = 1640 # um
            plt.scatter(self.data.df.x.values.compute()[self.g.ndata['indices'].numpy()], self.data.df.y.values.compute()[self.g.ndata['indices'].numpy()], c=clusters_colors, alpha=0.9,s=0.05,marker='.',linewidths=0, edgecolors=None)
            plt.xticks(fontsize=2)
            plt.yticks(fontsize=2)
            plt.axis('scaled')
            plt.savefig("{}/spatial_umap_embedding_clusters.png".format(self.folder), bbox_inches='tight', dpi=5000)

            import holoviews as hv
            hv.extension('matplotlib')
            molecules_y = self.data.df.y.values.compute()[self.g.ndata['indices'].numpy()]
            molecules_x = self.data.df.x.values.compute()[self.g.ndata['indices'].numpy()]
            nd_dic = {}

            allm = 0
            logging.info('Generating plots for cluster assigned to molecules...')
            
            if not os.path.isdir(os.path.join(self.folder,'Clusters')):
                os.mkdir('{}/Clusters'.format(self.folder))
            
            for cl in range(self.ClusterNames.shape[0]):
                    try:
                        x, y = molecules_x[clusters == cl], molecules_y[clusters == cl]
                        allm += x.shape[0]
                        color = ['red']*x.shape[0]
                        scatter =  hv.Scatter(np.array([x,y]).T).opts(
                            bgcolor='black',
                            aspect='equal',
                            fig_inches=50,
                            s=1,
                            title=str(self.ClusterNames[cl]),
                            color=color_dic[cl])
                        nd_dic[self.ClusterNames[cl]] = scatter
                        hv.save(scatter,"{}/Clusters/{}.png".format(self.folder,str(self.ClusterNames[cl])))
                        
                    except:
                        pass

            layout = hv.Layout([nd_dic[x] for x in nd_dic]).cols(5)
            hv.save(layout,"{}/molecule_prediction.png".format(self.folder))

            pred_labels = th.tensor(self.prediction_unlabelled)
            merge = np.concatenate([molecules_x[:,np.newaxis],molecules_y[:,np.newaxis]],axis=1)
            L = []
            logging.info('Generating plots for molecule cluster probabilities...')
            os.mkdir('{}/ClusterProbabilities'.format(self.folder))
            for n in range(self.ClusterNames.shape[0]):
                ps = pred_labels.detach().numpy()[:,n][:,np.newaxis]
                pdata= np.concatenate([merge,ps],axis=1)#[ps[:,0]>0.1,:]               
                scatter= hv.Scatter(pdata,
                                    kdims=['x','y'],vdims=[str(self.ClusterNames[n])]).opts(cmap='Viridis',
                                                                                        color=hv.dim(str(self.ClusterNames[n])),
                                                                                        s=1,
                                                                                        aspect='equal',
                                                                                        bgcolor='black',
                                                                                        fig_inches=20,
                                                                                        title=str(self.ClusterNames[n]),
                                                                                        clim=(pred_labels.min(),ps.max()))
                L.append(scatter)
                hv.save(scatter,"{}/ClusterProbabilities/{}.png".format(self.folder,str(self.ClusterNames[n])))
            
            layout = hv.Layout([x for x in L]).cols(2)
            logging.info('Plots saved.')

        else:
            import gc
            import scanpy as sc
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline

            if type(multigraph_classifier) == type(None):
                random_sample_train = np.random.choice(
                                        len(self.latent_unlabelled.detach().numpy()), 
                                        np.min([len(self.latent_unlabelled),5000]), 
                                        replace=False)
                training_latents =self.latent_unlabelled.detach().numpy()[random_sample_train,:]
                adata = sc.AnnData(X=training_latents)
                logging.info('Building neighbor graph for clustering...')
                sc.pp.neighbors(adata, n_neighbors=15)
                logging.info('Running Leiden clustering...')
                sc.tl.leiden(adata, random_state=42, resolution=1.4)
                logging.info('Leiden clustering done.')
                clusters= adata.obs['leiden'].values

                clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
                clf.fit(training_latents, clusters)
                clusters = clf.predict(self.latent_unlabelled.detach().numpy()).astype('uint16')

                unique_clusters = np.unique(clusters)
                dic = dict(zip(unique_clusters, np.arange(unique_clusters.shape[0])))
                clusters = np.array([dic[i] for i in clusters])

                molecules_id = self.g.ndata['indices']
                import gc
                gc.collect()
                if not os.path.isdir(os.path.join(self.folder,'Clusters')):
                    os.mkdir('{}/Clusters'.format(self.folder))

                merged_clusters= ClusterCleaner(
                    genes=self.data.unique_genes[np.where(self.g.ndata['gene'].numpy())[1]],
                    clusters=clusters
                    ).merge()

                unique_clusters = np.unique(merged_clusters)
                dic = dict(zip(unique_clusters, np.arange(unique_clusters.shape[0])))
                merged_clusters = np.array([dic[i] for i in merged_clusters])

                self.clusters = np.array(merged_clusters)
                new_labels = np.zeros(self.data.shape[0]) -1
                for i,l in zip(molecules_id, self.clusters):
                    new_labels[i] = l

            else:
                from joblib import load
                clf = load(multigraph_classifier)
                clusters = clf.predict(self.latent_unlabelled.detach().numpy()).astype('uint16')

                molecules_id = self.g.ndata['indices']
                import gc
                gc.collect()
                if not os.path.isdir(os.path.join(self.folder,'Clusters')):
                    os.mkdir('{}/Clusters'.format(self.folder))

                self.clusters = np.array(clusters)
                new_labels = np.zeros(self.data.shape[0]) -1
                for i,l in zip(molecules_id, self.clusters):
                    new_labels[i] = l

            self.data.add_dask_attribute('Clusters',new_labels.astype('str'),include_genes=True)

            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=eps,min_samples=min_samples) #*self.data.pixel_size.magnitude
            self.data.segment('Clusters',save_to=os.path.join(self.folder),func=db)
            gc.collect()

            with loompy.connect(os.path.join(self.folder,self.data.filename.split('.')[0]+'_cells.loom'),'r+') as ds:
                enrich = enrich_(labels_attr = ds.ca.Clusters)
                sparse_tmp = ds.sparse().tocsr()
                clusters_ = ds.ca['Clusters'].astype(float)
                cell_unique_clusters = np.unique(clusters_)
                r = enrich._fit(sparse_tmp,permute=False)
                ds.ra['enrichment'] = r

                dic = dict(zip(cell_unique_clusters, np.arange(cell_unique_clusters.shape[0])))
                self.clusters = np.array([i if i in dic else -1 for i in self.clusters])
                cell_clusters = clusters_#np.array([dic[i] for i in clusters_])
                ds.ca['Clusters'] = cell_clusters.astype('str')
                self.cell_unique_clusters = np.unique(cell_clusters)

            self.cell_clusters = self.clusters[self.clusters != -1]

            enriched_genes = {}
            self.enrichment = r
            logging.info('Enrichment shape: {}'.format(self.enrichment.shape))
            enrichment = r.argsort(axis=0)[::-1]
            for c in range(np.unique(clusters_).shape[0]):
                en_genes = enrichment[:,c][:10]
                enriched_genes[c] = self.data.unique_genes[en_genes]
            self.g.ndata['GSclusters'] = th.tensor(self.clusters,dtype=th.int64)
            np.save(self.folder+'/clusters',self.clusters)

            ### Add data to shoji ###
            try:
                import shoji
                loom_filename = os.path.join(self.folder,self.data.filename.split('.')[0]+'_cells.loom')
                logging.info('Adding {} to shoji'.format(loom_filename))
                analysis_name = loom_filename.split('/')[-2]
                self.add_graphicalcells_2shoji(
                    loom_filename,
                    analysis_name,
                    )
            except ImportError:
                logging.info('Shoji not installed. Please install from')

            ### PCI Seq ###
            if type(pci_file) != type(None):
                GPCI = GraphPCI(pci_file)
                GPCI.load_segmentation(
                    segmentation_path=os.path.join(self.folder,'Segmentation/*.parquet'),
                    output_name = os.path.join(self.folder,analysis_name)
                    )
                GPCI.run(self.folder, analysis_name)

                cellData= GPCI.cellData
                db = shoji.connect()
                filename =  loom_filename.split('RNA_transformed')[0].split('_')[-3]
                ws = db.eel.glioblastoma.graphicalCells[filename]
                cell_ID = ws.ID[:]
                gene_order = ws.Gene[:]

                pci_expression = np.zeros_like(ws.Expression[:])
                probs_classes = np.zeros([ws.Expression[:].shape[0], GPCI.ref_clusters.shape[0]],dtype=np.float32)

                for cell, i in tqdm(zip(cell_ID, range(cell_ID.shape[0]))):
                    cell_i= cellData[cellData.Cell_Num == cell]
                    genes_i = cell_i.Genenames.values[0]
                    values = cell_i.CellGeneCount.values[0]
                    
                    where = np.where(np.isin(gene_order,genes_i))[0]
                    expression = np.zeros([gene_order.shape[0]])
                    for w, e in zip(where, values):
                        expression[w] = e
                    pci_expression[i,:] = expression

                    predicted_classes = np.array(cell_i.ClassName.values[0])
                    p = np.array(cell_i.Prob.values[0])
                    order = np.argsort(p)[::-1]

                    predicted_classes = predicted_classes[order]
                    p = p[order]

                    dic_p = dict(zip(predicted_classes,p))
                    probs_cell_i = np.zeros([GPCI.ref_clusters.shape[0]])
                    for c, n  in zip(GPCI.ref_clusters, range(GPCI.ref_clusters.shape[0])):
                        if c in dic_p.keys():
                            probs_cell_i[n] = dic_p[c]
                    probs_classes[i,:] = probs_cell_i
                
                ws.ExpressionPCI =shoji.Tensor("float32", ("cells", "genes"), inits=pci_expression.astype('float32'))  
                ws.pciClusters = shoji.Dimension(GPCI.ref_clusters.shape[0])
                ws.ClustersPCI = shoji.Tensor("string", ("pciClusters",), inits=GPCI.ref_clusters.astype(object))  
                ws.ProbabilitiesPCI = shoji.Tensor("float32", ("cells","pciClusters"), inits=probs_classes.astype('float32'))

            #### Plotting ####
            logging.info('Clustering done.')
            logging.info('Generating umap embedding...')
            gc.collect()
            
            colors = colorize(np.arange(np.unique(self.clusters).shape[0]))
            color_dic = {}
            for x in np.unique(self.clusters):
                c = colors[x,:].tolist()
                color_dic[x] = (c[0],c[1],c[2])
            logging.info(color_dic)
            clusters_colors = np.array([color_dic[x] for x in self.clusters])

            some = np.random.choice(np.arange(self.latent_unlabelled.shape[0]),np.min([random_n, self.latent_unlabelled.shape[0]]),replace=False)
            umap_embedding = reducer.fit_transform(self.latent_unlabelled[some])
            #embedding = umap_embedding.transform(self.latent_unlabelled)
            Y_umap = umap_embedding
            Y_umap -= np.min(Y_umap, axis=0)
            Y_umap /= np.max(Y_umap, axis=0)

            fig=plt.figure(figsize=(7,4),dpi=500)
            cycled = [0,2,1,0]
            for i in range(3):
                plt.subplot(1,3,i+1)
                plt.scatter(Y_umap[:,cycled[i]], Y_umap[:,cycled[i+1]], c=clusters_colors[some],  s=0.5, marker='.', linewidths=0, edgecolors=None)
                plt.xlabel("Y"+str(cycled[i]))
                plt.ylabel("Y"+str(cycled[i+1]))
            plt.tight_layout()
            plt.savefig("{}/umap.png".format(self.folder), bbox_inches='tight', dpi=500)

            fig=plt.figure(figsize=(2,2),dpi=1000,)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_facecolor("black")
            #plt.scatter(DS.df.x.values.compute()[self.cells], DS.df.y.values.compute()[self.cells], c=th.argmax(pred.softmax(dim=-1),dim=-1).numpy(), s=0.2,marker='.',linewidths=0, edgecolors=None,cmap='rainbow')
            plt.scatter(self.data.df.x.values.compute()[molecules_id.numpy()], self.data.df.y.values.compute()[molecules_id.numpy()], c=clusters_colors, s=0.005,marker='.',linewidths=0, edgecolors=None)
            plt.xticks(fontsize=2)
            plt.yticks(fontsize=2)
            plt.axis('scaled')
            plt.savefig("{}/spatial_umap_embedding.png".format(self.folder), bbox_inches='tight', dpi=5000)

            import holoviews as hv
            from holoviews import opts
            hv.extension('matplotlib')
            molecules_y = self.data.df.y.values.compute()[molecules_id.numpy()]
            molecules_x = self.data.df.x.values.compute()[molecules_id.numpy()]
            nd_dic = {}
            allm = 0
            logging.info('Generating plots for cluster assigned to molecules...')
            lay = []
            for cl in np.unique(self.clusters):
                try:
                    x, y = molecules_x[self.clusters == cl], molecules_y[self.clusters == cl]
                    allm += x.shape[0]

                    scatter =  hv.Scatter(np.array([x,y]).T).opts(
                        bgcolor='black',
                        aspect='equal',
                        fig_inches=50,
                        s=1,
                        title=str(cl),
                        color=color_dic[cl])
                    nd_dic[cl] = scatter.opts(title=' - '.join(enriched_genes[float(cl)]), fontsize={'title':24})
                    lay.append(nd_dic[cl])
                    hv.save(scatter,"{}/Clusters/{}.png".format(self.folder,str(cl)), )   
                except:
                    logging.info('Could not get cluster {}'.format(cl))   

            layout = hv.Layout(lay).cols(5).opts(opts.Scatter(s=0.1,fontsize={'title':8}))
            hv.save(layout,"{}/molecule_prediction.png".format(self.folder))
            import holoviews as hv
            hv.notebook_extension('bokeh')
            hmap = hv.HoloMap(kdims=['Enrichment - Cluster'])
            for k in nd_dic:
                hmap[str(k) + ' {}'.format(enriched_genes[float(k)])] = nd_dic[k].opts(bgcolor='black',width=1000,data_aspect=1,size=1,color=color_dic[k])
                #hmap = hmap.opts(opts.Scatter(bgcolor='black',width=1000,data_aspect=1,size=1))


            hv.save(hmap, "{}/Clusters.html".format(self.folder),fmt='html')
            self.save_graph()
            '''except:
                logging.info('Could not generate html file')'''
    
    def add_graphicalcells_2shoji(self,filepath, analysis_name):
        import shoji
        db = shoji.connect()

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

        synonym_dic = {'CD40Li':'CD40LGi',
                    'CD133i':'PROM1i', 
                    'CD134Li':'TNFSF4i',
                    'CD134i': 'TNFRSF4i',
                    'CD137Li':'TNFSF9i'}        

        new_unspliced = []
        for g in unspliced:
            if g in synonym_dic:
                new_unspliced.append(synonym_dic[g])
            else:
                new_unspliced.append(g)
        unspliced = new_unspliced

        for g in unspliced:
            if g[:-1] not in spliced:
                spliced.append(g[:-1])
        unspliced, spliced = np.array(unspliced), np.array(spliced)

        with loompy.connect(filepath, 'r') as ds:
            
            print('Row attributes',ds.ra.keys())
            print('Column Attributes',ds.ca.keys())
            genes = ds.ra.Gene[:]
            control_remove = np.array([True if g.count('Control') + g.count('CONTROL') == 0  else False for g in genes])
            genes_ = genes[control_remove]
            data = ds[:, :].T 
            data = data[:,control_remove]
            filter_cells = (data.sum(axis=1) >= 5) & (data.sum(axis=1) < 500) #& ((data >= 2).sum(axis=1) > 0)
            print('Remaining cells: {}'.format(filter_cells.sum()))

            genes_spliced = genes_[np.isin(genes_, spliced)]
            genes_unspliced = genes_[np.isin(genes_, unspliced)]

            data_s = data[filter_cells,:][:,np.isin(genes_, spliced)]
            data_u = data[filter_cells,:][:,np.isin(genes_, unspliced)]

            order_s, order_u = [], []
            for g in genes_spliced:
                idx_s = np.where(spliced == g)[0][0]
                order_s.append(idx_s)

            for g in genes_unspliced:
                try:
                    idx_u = np.where(spliced == g[:-1])[0][0]
                    order_u.append(idx_u)
                except:
                    print(g)

            order_s, order_u = np.array(order_s), np.array(order_u)

            Expression = np.zeros([data_s.shape[0], spliced.shape[0]],dtype=np.uint16)
            Unspliced = np.zeros([data_s.shape[0], spliced.shape[0]],dtype=np.uint16)

            Expression[:,order_s] = data_s
            if data_u.shape[1] > 0:
                Unspliced[:,order_u] = data_u
            data = Expression + Unspliced
            filename = filepath.split('RNA_transformed')[0].split('_')[-3]
            
            del db.eel.glioblastoma.graphicalCells[filename]
            db.eel.glioblastoma.graphicalCells[filename] = shoji.Workspace()
            ws = db.eel.glioblastoma.graphicalCells[filename]
            
            ws.AnalysisName = shoji.Tensor("string", (), inits=analysis_name)  

            ws.cells = shoji.Dimension(data.shape[0])
            ws.genes = shoji.Dimension(data.shape[1])

            ws.Expression = shoji.Tensor("uint16", ("cells", "genes"), inits=data.astype('uint16'))  
            ws.Unspliced = shoji.Tensor("uint16", ("cells", "genes"), inits=Unspliced.astype('uint16'))  
            ws.Gene = shoji.Tensor("string", ("genes",), inits=spliced.astype('object'))  
            ws.Accession = shoji.Tensor("string", ("genes",), inits=spliced.astype('object'))  
            
            ws.NGenes = shoji.Tensor("uint16", ("cells",), inits = (data >0).sum(axis=1).astype('uint16'))

            ws.SelectedFeatures = shoji.Tensor("bool", ("genes",), inits=np.ones(ws.genes.length, dtype="bool"))
            ws.TotalUMIs = shoji.Tensor("uint32", ("cells",), inits=data.sum(axis=1).astype("uint32"))

            ws.GeneTotalUMIs = shoji.Tensor("uint32", ("genes",), inits=data.sum(axis=0).astype("uint32"))
            ws.OverallTotalUMIs = shoji.Tensor("uint64", (), inits=data.sum().astype("uint64"))
            ws.X = shoji.Tensor("float32", ("cells",), inits=ds.ca.Centroid[:,0][filter_cells].astype('float32')) # Load the spatial X and Y coordinates 
            ws.Y = shoji.Tensor("float32", ("cells",), inits=ds.ca.Centroid[:,1][filter_cells].astype('float32'))
            ws.GraphCluster = shoji.Tensor("uint16", ("cells",), inits=ds.ca.Clusters[filter_cells].astype('uint16'))
            #ws.NucleusArea_um = shoji.Tensor("float32", ("cells",), inits=ds.ca.Nucelus_area_um2[:].astype('float32')) # Load the spatial X and Y coordinates 
            #ws.NucleusArea_px = shoji.Tensor("float32", ("cells",), inits=ds.ca.Nucleus_area_px[:][filter_cells].astype('float32'))

            ws.Species = shoji.Tensor("string", (), inits='Homo sapiens')
            ws.SelectedFeatures = shoji.Tensor("bool", ('genes',), inits=np.ones([ws.genes.length]).astype(bool))

            ws.ValidGenes = shoji.Tensor("bool", ('genes',), inits=np.ones([ws.genes.length]).astype(bool))
            ws.Sample = shoji.Tensor("string", ("cells",), inits= np.array([filename]*filter_cells.sum(),dtype='object' ))
            ws.ID = shoji.Tensor("uint64", ("cells",), inits= ds.ca.Segmentation[:][filter_cells].astype("uint64"))


    def execute(self, c, nodes,att1, att2):
        g1,bg1, counts_cl1 = self.plot_cluster(c,nodes,att1)
        bg1.to_parquet('{}/attention/SyntaxNGH1_Cluster{}.parquet'.format(self.folder,c))
        hv.save(g1, '{}/attention/AttentionNGH1_{}.html'.format(self.folder, c))
        g2,bg2, counts_cl2 = self.plot_cluster(c,nodes,att2)
        bg2.to_parquet('{}/attention/SyntaxNGH2_Cluster{}.parquet'.format(self.folder,c))
        hv.save(g2, '{}/attention/AttentionNGH2_{}.html'.format(self.folder, c))
        return bg1, bg2, counts_cl1, counts_cl2

    def plot_networkx(self):
        import shutil

        gene_ = self.g.ndata['gene'].numpy()
        result = np.where(gene_==1)
        rg = [self.data.unique_genes[r] for r in result[1]]
        self.dic_ = dict(zip(result[0],rg))
        self.dic_clusters = dict(zip([int(n) for n in self.g.nodes()] ,self.clusters))

        if os.path.exists(path.join(self.folder,'attention')):
            shutil.rmtree(path.join(self.folder,'attention'))
            os.mkdir(path.join(self.folder,'attention'))
        else:
             os.mkdir(path.join(self.folder,'attention'))

        bible1 = np.zeros([self.data.unique_genes.shape[0], self.data.unique_genes.shape[0]])
        bible2 = np.zeros([self.data.unique_genes.shape[0], self.data.unique_genes.shape[0]])

        #from joblib import Parallel, delayed
        result = []

        counts_cluster = {}
        for c in tqdm(np.unique(self.cell_unique_clusters)):
            nodes= self.g.nodes()[self.clusters == c]
            att1, att2 = self.get_attention_nodes(nodes=nodes)
            #logging.info(att1.shape,att2.shape)
            bg1,bg2, counts_cl1, counts_cl2 = self.execute(c, nodes,att1,att2)
            result.append((bg1,bg2)) 

            counts_cluster[c] = [counts_cl1, counts_cl2]

        inter_graph, df = self.plot_intercluster(counts_cluster)
        hv.save(inter_graph, '{}/attention/ClusterConnectivity.html'.format(self.folder, c))

        df = pd.DataFrame(index=np.unique(self.clusters[self.clusters != -1]).astype('str'),columns=np.unique(self.clusters[self.clusters != -1]).astype('str'), data=df)
        df.to_parquet('{}/attention/ClusterConnectivity.parquet'.format(self.folder,c))

        for b in result:
            bible1 += b[0].values
            bible2 += b[1].values

        bible1 = pd.DataFrame(index=self.data.unique_genes, columns=self.data.unique_genes, data=bible1)
        bible1.to_parquet('{}/attention/GrammarNGH1.parquet'.format(self.folder))
        bible2 = pd.DataFrame(index=self.data.unique_genes, columns=self.data.unique_genes, data=bible2)
        bible2.to_parquet('{}/attention/GrammarNGH2.parquet'.format(self.folder))

    def bible_grammar(self, e0, e1, att):
        network_grammar = []
        
        for g in tqdm(self.data.unique_genes):
            filter1 = e0 == g
            probs_gene = []
            for g2 in self.data.unique_genes:
                filter2 = e1 == g2
                probs = att[filter1 & filter2].sum()
                probs_gene.append(probs)
            
            pstack = np.stack(probs_gene)
            pstack = pstack/pstack.sum()
            network_grammar.append(pstack)
        logging.info('Syntax learned')
        network_grammar = np.stack(network_grammar)
        #bible_network_ngh = pd.DataFrame(index=self.data.unique_genes, columns= self.data.unique_genes ,data=network_grammar)
        return network_grammar
    
    def bible_grammar2(self, e0, e1, att):
        df = pd.DataFrame({'0':e0,'1':e1, 'w':att})
        df2 = df.pivot_table(
            index='0', 
            columns='1',
            aggfunc='sum',
            fill_value=0,
            dropna=False
            ).reindex(self.data.unique_genes, axis=0)
            
        df2.columns = self.data.unique_genes
        return df2

    def plot_cluster(self,cluster,nodes_cluster_i,att):
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import itertools

        edges = dgl.in_subgraph(self.g,nodes_cluster_i).edges()

        e0_cluster_genes = [self.dic_[e] for e in edges[0].numpy()]
        e1_cluster_genes = [self.dic_[e] for e in edges[1].numpy()]

        ### This part is for the cluster interconnectivity
        Q = np.quantile(att[:,0],0.75)
        filt_edges_cluster1 = edges[0][att[:,0] >  Q]
        filt_edges_cluster2 = edges[1][att[:,0] > Q]
        edges_cluster1 = [self.dic_clusters[e] for e in filt_edges_cluster1.numpy()]
        edges_cluster2 = [self.dic_clusters[e] for e in filt_edges_cluster2.numpy()]
        edges_cluster = edges_cluster1 + edges_cluster2
        cluster_, counts = np.unique(edges_cluster, return_counts=True)
        counts_cluster = dict(zip(cluster_[cluster_ != -1], counts[cluster_ != -1]))
        ###

        a = itertools.combinations(self.data.unique_genes,2)
        att_add = []
        for x in a:
            e0_cluster_genes.append(x[0])
            e1_cluster_genes.append(x[1])

            e0_cluster_genes.append(x[1])
            e1_cluster_genes.append(x[0])
            att_add += [0,0]

        e0_cluster_genes = np.array(e0_cluster_genes)
        e1_cluster_genes = np.array(e1_cluster_genes)
        edges = np.array([e0_cluster_genes,e1_cluster_genes])
        att_add = np.array(att_add)
        att = np.concatenate([att[:,0],att_add])

        bg = self.bible_grammar2(e0_cluster_genes, e1_cluster_genes, att).fillna(0)
        graph_edges1 = []
        graph_edges2 = []
        graph_weights = []

        a = itertools.combinations(self.data.unique_genes,2)
        for x in a:
            graph_edges1.append(x[0])
            graph_edges2.append(x[1])
            graph_weights.append(bg[x[0]][x[1]])
        
        graph_edges1 = np.array(graph_edges1)
        graph_edges2 = np.array(graph_edges2)
        graph_weights = np.array(graph_weights)

        graph_edges1 = graph_edges1[graph_weights > np.quantile(graph_weights,0.5)]
        graph_edges2 = graph_edges2[graph_weights > np.quantile(graph_weights,0.5)]
        graph_weights = graph_weights[graph_weights > np.quantile(graph_weights,0.5)]
        graph_weights = np.array(graph_weights)/graph_weights.sum(axis=0)
        
        node_frequency = np.unique(np.array([graph_edges1,graph_edges2]),return_counts=True)
        genes_present,node_frequency = node_frequency[0],node_frequency[1]
        node_frequency = node_frequency#/node_frequency.sum()

        graph = hv.Graph(((graph_edges1,graph_edges2, graph_weights),),vdims='Attention').opts(
            opts.Graph(edge_cmap='viridis', edge_color='Attention'),
            )#, edge_cmap='viridis', edge_color='Attention')

        df = graph.nodes.data
        enrichment =  self.enrichment[:,cluster]
        enrichmentQ = np.quantile(enrichment,0.5)
        enriched_genes = self.data.unique_genes[enrichment > enrichmentQ]

        
        genes1 = graph_edges1[np.isin(graph_edges1,enriched_genes)| np.isin(graph_edges2,enriched_genes)]
        genes2 = graph_edges2[np.isin(graph_edges1,enriched_genes)| np.isin(graph_edges2,enriched_genes)]
        enriched_genes_connected= np.unique(np.array([genes1,genes2]))

        filter_enrichment = np.isin(graph_edges1,enriched_genes)| np.isin(graph_edges2,enriched_genes)
        graph_edges1 = graph_edges1[filter_enrichment]
        graph_edges2 = graph_edges2[filter_enrichment]
        graph_weights = graph_weights[filter_enrichment]

        node_freq =  node_frequency[np.isin(df['index'].values,enriched_genes_connected)]
        node_enrich = enrichment[np.isin(self.data.unique_genes,enriched_genes_connected)]

        df = df[np.isin(df['index'].values,enriched_genes_connected)]
        df.loc[:,'Frequency'] = node_freq
        df.loc[:,'Enrichment'] = node_enrich

        graph = hv.Graph(((graph_edges1,graph_edges2, graph_weights),df),vdims='Attention').opts(
            opts.Graph(
                edge_cmap='viridis', edge_color='Attention',node_color='Frequency',
                cmap='plasma', edge_line_width=hv.dim('Attention')*20,node_size=hv.dim('Enrichment')*5,
                edge_nonselection_alpha=0, width=1500,height=1500)
                )

        labels = hv.Labels(graph.nodes, ['x', 'y'],'index')
        #graph = graph #* labels.opts(text_font_size='8pt', text_color='white', bgcolor='grey')
        graph = graph*labels.opts(text_font_size='5pt', text_color='white', bgcolor='grey')
        '''graph = datashade(graph, normalization='linear', width=1000, height=1000).opts(
            opts.Graph(
                    edge_cmap='viridis', edge_color='Attention',node_color='Frequency',
                    cmap='plasma', edge_line_width=hv.dim('Attention')*10,
                    edge_nonselection_alpha=0, width=2000,height=2000)
            )'''
        return graph, bg, counts_cluster

    def _intercluster_df(self, dic):
        intercluster_network = []
        for c in dic:
            counts_cl_i = []
            dic1_cluster_i = dic[c][0]
            dic2_cluster_i = dic[c][1]

            for c2 in np.unique(self.clusters[self.clusters != -1]):
                counts = 0
                if c2 in dic1_cluster_i:
                    counts += dic1_cluster_i[c2]
                if c2 in dic2_cluster_i:
                    counts += dic2_cluster_i[c2]
                counts_cl_i.append(counts)
            intercluster_network.append(counts_cl_i)

        return np.stack(intercluster_network)


    def plot_intercluster(self, dic):
        import itertools
        df = self._intercluster_df(dic)
        node_frequency = df.sum(axis=1)
        df = (df.T/df.sum(axis=1)).T

        a = itertools.combinations(np.arange(len(df)),2)
        graph_edges1 = []
        graph_edges2 = []
        graph_weights = []

        for x in a:
            graph_edges1.append(x[0])
            graph_edges2.append(x[1])
            graph_weights.append(df[x[0]][x[1]])
        
        graph_edges1 = np.array(graph_edges1)
        graph_edges2 = np.array(graph_edges2)
        graph_weights = np.array(graph_weights)

        graph = hv.Graph(((graph_edges1,graph_edges2, graph_weights),),vdims='Attention').opts(
            opts.Graph(edge_cmap='viridis', edge_color='Attention'),
            )

        df_info = graph.nodes.data
        df_info.loc[:,'Frequency'] = node_frequency

        graph = hv.Graph(((graph_edges1,graph_edges2, graph_weights),df_info),vdims='Attention').opts(
            opts.Graph(
                edge_cmap='viridis', edge_color='Attention',node_color='Frequency',
                cmap='plasma', edge_line_width=hv.dim('Attention')*20,
                edge_nonselection_alpha=0, width=1500,height=1500)
                )

        labels = hv.Labels(graph.nodes, ['x', 'y'],'index')
        #graph = graph #* labels.opts(text_font_size='8pt', text_color='white', bgcolor='grey')
        inter_graph = graph*labels.opts(text_font_size='5pt', text_color='white', bgcolor='grey')
        return inter_graph, df

        

class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n*self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst

class enrich_: #sparese
   
    def __init__(self, 
                 labels_attr: np.array) -> None:
        self.labels_attr = labels_attr
        self.permute_labs = None
        self.sizes = None
        self.nnz = None
        self.means = None
        self.f_nnz = None
        
    def _shuffle(self):
        
        permute_labs = np.random.permutation(self.labels_attr)

        self.permute_labs = permute_labs
        
    def _sort_col(self,arr,ordering):
        from scipy import sparse
        
        arr_list =[]
        # arr_ = sparse_tmp.copy()
        chunksize = 100000000 // arr.shape[1]
        start = 0
        while start < arr.shape[0]:
            submatrix = arr[start:start + chunksize, :]
            arr_list.append(submatrix[:, ordering])
            start = start + chunksize
            
        return sparse.vstack(arr_list)

    def _fit(self, mtx,permute:bool=False):
        
            if permute:
                enrich_._shuffle()
                labels = self.permute_labs
                logging.info(f'permute{labels}')
            
            else: 
                labels = self.labels_attr 
            labels = labels.astype(float)
            
            # Need to sort out through labels first before doing the split 
            idx = np.unique(labels[np.argsort(labels)], return_index=True)
            idx_ = np.concatenate([idx[1],[mtx.shape[1]]])
            
            mtx_ = self._sort_col(mtx,np.argsort(labels))
    
            alist = []
            for i in range(len(idx_)-1):

                alist.append(mtx_[:,idx_[i]:idx_[i+1]])   

            n_labels = max(labels) + 1

            n_cells = mtx_.shape[1]

            sizes = np.zeros(len(idx[0]))
            nnz=np.zeros([mtx_.shape[0],len(idx[0])])
            means=np.zeros([mtx_.shape[0],len(idx[0])])
            for i in np.arange(len(alist)):

                nnz[:,i] = alist[i].getnnz(axis=1)
                means[:,i] = np.squeeze((alist[i].mean(axis=1).A))
                sizes[i] = alist[i].shape[1]
                
            self.sizes, self.nnz, self.means = sizes, nnz, means

            # Non-zeros and means over all cells
            (nnz_overall, means_overall) = mtx_.getnnz(axis=1),np.squeeze((mtx_.mean(axis=1).A))

            # Scale by number of cells
            f_nnz = nnz / sizes
            f_nnz_overall = nnz_overall / n_cells
            
            self.f_nnz = f_nnz


            # Means and fraction non-zero values in other clusters (per cluster)
            means_other  = ((means_overall * n_cells)[None].T - (means * sizes)) / (n_cells - sizes)
            f_nnz_other = ((f_nnz_overall * n_cells)[None].T - (f_nnz * sizes)) / (n_cells - sizes)

            # enrichment = (f_nnz + 0.1) / (f_nnz_overall[None].T + 0.1) * (means + 0.01) / (means_overall[None].T + 0.01)
            enrichment = (f_nnz + 0.1) / (f_nnz_other + 0.1) * (means + 0.01) / (means_other + 0.01)

            return enrichment


