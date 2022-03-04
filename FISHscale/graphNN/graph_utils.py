import torch as th
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from scipy import sparse
import pytorch_lightning as pl
import os
from annoy import AnnoyIndex
import networkx as nx
from tqdm import trange
import dgl
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import loompy

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
        sm = sparse.csr_matrix((data,(rows,cols))).T
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

        Computes the distance at which 97 percentile molecules are connected to
        at least one other molecule. Like PArtel & Wahlby

        Args:
            omega ([type]): [description]
            tau ([type]): [description]
        """        
        if type(tau) == type(None):
            from scipy.spatial import cKDTree as KDTree
            x,y = self.data.df.x.values.compute(),self.data.df.y.values.compute()
            kdT = KDTree(np.array([x,y]).T)
            d,i = kdT.query(np.array([x,y]).T,k=2)
            d_th = np.percentile(d[:,1],97)*omega
            self.distance_threshold = d_th
            print('Chosen dist: {}'.format(d_th))
        else:
            self.distance_threshold = tau
            print('Chosen dist: {}'.format(tau))

    def compute_library_size(self):
        data= self.g.ndata['ngh'].T
        sum_counts = data.sum(axis=1)
        masked_log_sum = np.ma.log(sum_counts)
        log_counts = masked_log_sum.filled(0)
        local_mean = (np.mean(log_counts).reshape(-1, 1)).astype(np.float32)
        local_var = (np.var(log_counts).reshape(-1, 1)).astype(np.float32)
        return local_mean, local_var

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
        print('Building graph...')
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
        print('Chosen dist: {}'.format(self.distance_threshold))
        
        def find_nn_distance(coords,tree,distance):
            print('Find neighbors below distance: {}'.format(d_th))
            res,nodes,ngh_ = [],[],[]
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
                ngh_.append(n_)
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
        g.ndata['gene'] = th.tensor(d.toarray(), dtype=th.float32)#[self.g.ndata['indices'].numpy(),:]
        nghs = []
        for n in tqdm(ngh_):
            nghs.append(th.tensor(g.ndata['gene'][n,:].sum(axis=0)))
        nghs = th.stack(nghs)
        g.ndata['ngh'] = nghs

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

        sum_nodes_connected = g.ndata['ngh'].sum(axis=1)
        molecules_connected = molecules[sum_nodes_connected >= self.minimum_nodes_connected]
        remove = molecules[sum_nodes_connected.numpy() < self.minimum_nodes_connected]
        g.remove_nodes(th.tensor(remove))
        g.ndata['indices'] = th.tensor(molecules_connected)
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
                print(ds.ca.keys())
                try:
                    k = list(self.exclude_clusters.keys())[0]
                    v = self.exclude_clusters[k]
                    region_filt = np.isin(ds.ca[k], v, invert=True)
                    self.ClusterNames = ds.ca[k][region_filt]
                    print('Selected clusters: {}'.format(self.ClusterNames))
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
                print('Reference dataset shape: {}'.format(self.ref_celltypes.shape))
            
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
    def get_umap(self,random_n=50000,n_clusters=50):
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
                n_jobs=-1
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
            #plt.scatter(DS.df.x.values.compute()[GD.cells], DS.df.y.values.compute()[GD.cells], c=th.argmax(pred.softmax(dim=-1),dim=-1).numpy(), s=0.2,marker='.',linewidths=0, edgecolors=None,cmap='rainbow')
            plt.scatter(self.data.df.x.values.compute()[self.g.ndata['indices'].numpy()][some], self.data.df.y.values.compute()[self.g.ndata['indices'].numpy()][some], c=Y_umap, s=0.05,marker='.',linewidths=0, edgecolors=None)
            plt.xticks(fontsize=2)
            plt.yticks(fontsize=2)
            plt.axis('scaled')
            plt.savefig("{}/spatial_umap_embedding.png".format(self.folder), bbox_inches='tight', dpi=5000)

            clusters= self.prediction_unlabelled.argsort(axis=-1)[:,-1]
            import random
            r = lambda: random.randint(0,255)
            color_dic = {}
            for x in range(self.ClusterNames.shape[0]):
                color_dic[x] = (r()/255,r()/255,r()/255)
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
            print('Generating plots for cluster assigned to molecules...')
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
            print('Generating plots for molecule cluster probabilities...')
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
                                                                                        fig_inches=50,
                                                                                        title=str(self.ClusterNames[n]))
                L.append(scatter)


                hv.save(scatter,"{}/ClusterProbabilities/{}.png".format(self.folder,str(self.ClusterNames[n])))
            layout = hv.Layout([x for x in L]).cols(5)
            hv.save(layout,"{}/molecule_probabilities.png".format(self.folder))

            print('Plots saved.')

        else:
            import scanpy as sc
            from sklearn.cluster import MiniBatchKMeans
            print('Running leiden clustering from scanpy...')
            adata = sc.AnnData(X=self.latent_unlabelled.detach().numpy())
            sc.pp.neighbors(adata, n_neighbors=25)
            sc.tl.leiden(adata, random_state=42)
            self.clusters= adata.obs['leiden'].values
            #kmeans = MiniBatchKMeans(n_clusters=n_clusters)
            #self.clusters = kmeans.fit_predict(self.latent_unlabelled.detach().numpy())
            
            np.save(self.folder+'/clusters',self.clusters)
            print('Clustering done.')
            print('Generating umap embedding...')
            
            import random
            r = lambda: random.randint(0,255)
            color_dic = {}
            for x in np.unique(self.clusters):
                color_dic[x] = (r()/255,r()/255,r()/255)
            clusters_colors = np.array([color_dic[x] for x in self.clusters])

            some = np.random.choice(np.arange(self.latent_unlabelled.shape[0]),random_n,replace=False)
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
            #plt.scatter(DS.df.x.values.compute()[GD.cells], DS.df.y.values.compute()[GD.cells], c=th.argmax(pred.softmax(dim=-1),dim=-1).numpy(), s=0.2,marker='.',linewidths=0, edgecolors=None,cmap='rainbow')
            plt.scatter(self.data.df.x.values.compute()[self.g.ndata['indices'].numpy()], self.data.df.y.values.compute()[self.g.ndata['indices'].numpy()], c=clusters_colors, s=0.05,marker='.',linewidths=0, edgecolors=None)
            plt.xticks(fontsize=2)
            plt.yticks(fontsize=2)
            plt.axis('scaled')
            plt.savefig("{}/spatial_umap_embedding.png".format(self.folder), bbox_inches='tight', dpi=5000)

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