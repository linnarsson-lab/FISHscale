import networkx as nx
from networkx.algorithms.traversal import edgedfs
from numpy.core.fromnumeric import size
import torch as th
import numpy as np
import torch
from tqdm import tqdm
from annoy import AnnoyIndex
from tqdm import trange
import os
import pytorch_lightning as pl
from typing import Optional
from scipy import sparse
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import h5py
import sklearn.linear_model as lm
import sklearn.metrics as skm
import dgl

class UnsupervisedClassification(Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        node_emb = th.cat(self.val_outputs, 0)
        g = trainer.datamodule.g
        labels = g.ndata['labels']
        f1_micro, f1_macro = compute_acc_unsupervised(
            node_emb, labels, trainer.datamodule.train_nid,
            trainer.datamodule.val_nid, trainer.datamodule.test_nid)
        pl_module.log('val_f1_micro', f1_micro)

def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test

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

class GraphData(pl.LightningDataModule):
    """
    Class to prepare the data for GraphSAGE

    """    
    def __init__(self,
        data, # Data as numpy array of shape (Genes, Cells)
        model, # GraphSAGE model
        analysis_name:str,
        cells=None, # Array with cell_ids of shape (Cells)
        distance_threshold = 250,
        minimum_nodes_connected = 25,
        ngh_sizes = [20, 10],
        train_p = 0.25,
        batch_size= 1024,
        num_workers=1,
        save_to = '',
        subsample=1,
        ref_celltypes=None,
        smooth:bool=False,
        negative_samples:int=5,
        ):
        """
        Initialize GraphData class

        Args:
            data (FISHscale.utils.dataset.Dataset): Dataset object.
            model (FISHscale.graphNN.models.SAGE): GraphSAGE model.
            analysis_name (str): Filename for data and analysis.
            distance_threshold (int, optional): Maximum distance to consider to molecules neighbors. Defaults to 250um.
            minimum_nodes_connected (int, optional): Nodes with less will be eliminated. Defaults to 5.
            ngh_sizes (list, optional): Neighborhood sizes that will be aggregated. Defaults to [20, 10].
            train_p (float, optional): Training size, as percentage. Defaults to 0.75.
            batch_size (int, optional): Batch size. Defaults to 1024.
            num_workers (int, optional): Workers for sampling. Defaults to 1.
            save_to (str, optional): Path to save network edges and nn tree. Defaults to current path.
            subsample (int,optional): Subsample part of the input data if it is to large.
            ref_celltypes (np.array, optional): Cell types for decoder. Shape (genes,cell types)       
        """        

        super().__init__()

        self.model = model
        self.analysis_name = analysis_name
        self.ngh_sizes = ngh_sizes
        self.data = data
        self.cells = cells
        self.distance_threshold = distance_threshold
        self.minimum_nodes_connected = minimum_nodes_connected
        self.train_p = train_p
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_to = save_to
        self.ref_celltypes = ref_celltypes 
        self.smooth = smooth
        self.negative_samples = negative_samples

        self.folder = self.analysis_name+ '_' +datetime.now().strftime("%Y-%m-%d-%H%M%S")
        os.mkdir(self.folder)

        self.subsample = subsample
        self.subsample_xy()

        if type(self.ref_celltypes) != type(None):
            self.cluster_nghs, self.cluster_edges, self.cluster_labels = self.cell_types_to_graph(self.ref_celltypes)
        
        # Save random cell selection
        edges = self.buildGraph(self.distance_threshold)
        self.compute_size()
        self.setup()
        if self.smooth:
            self.knn_smooth()

        self.g= dgl.graph((edges[0,:],edges[1,:]))
        self.g.ndata['gene'] = th.tensor(self.d.toarray(),dtype=th.float32)


        self.sampler = dgl.dataloading.MultiLayerNeighborSampler([int(_) for _ in self.ngh_sizes])
        self.device = th.device('cpu')

        self.checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=self.folder,
            filename=self.analysis_name+'-{epoch:02d}-{train_loss:.2f}',
            save_top_k=2,
            mode='min',
            )
        self.early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.1,
            patience=50,
            verbose=True,
            mode='min',
            )
    
    def prepare_data(self):
        # do-something
        pass

    def setup(self, stage: Optional[str] = None):
        #self.d = th.tensor(self.molecules_df(),dtype=th.float32) #works
        self.d = self.molecules_df()

    def compute_size(self):
        cells = self.cells
        if self.smooth:
            cells = self.molecules_connected
        np.save(self.folder +'/cells.npy', cells)

        self.train_size = int((cells.shape[0])*self.train_p)
        self.test_size = cells.shape[0]-int(cells.shape[0]*self.train_p)  
        random_state = np.random.RandomState(seed=0)
        permutation = random_state.permutation(cells.shape[0])
        self.indices_test = th.tensor(permutation[:self.test_size])
        self.indices_train = th.tensor(permutation[self.test_size : (self.test_size + self.train_size)])
        self.indices_validation = th.tensor(np.arange(cells.shape[0]))

    def train_dataloader(self):
        return dgl.dataloading.EdgeDataLoader(
                        self.g,
                        self.indices_train,
                        self.sampler,
                        negative_sampler=NegativeSampler(self.g, self.negative_samples, False),
                        #device=self.device,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=self.num_workers)

    def val_dataloader(self):
        # Note that the validation data loader is a NodeDataLoader
        # as we want to evaluate all the node embeddings.
        return dgl.dataloading.NodeDataLoader(
            self.g,
            np.arange(self.g.num_nodes()),
            self.sampler,
            #device=self.device,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers)
        '''if type(self.ref_celltypes) != type(None):
            #self.indices_labelled = th.tensor(np.random.randint(0,self.cluster_nghs.shape[0],size=self.indices_train.shape[0]))
            self.indices_labelled = th.tensor(np.random.choice(self.cluster_edges.unique().numpy(),size=self.indices_train.shape[0]))
        labelled = None

        if type(labelled) != type(None):
            return {'unlabelled':unlabelled,'labelled':labelled}
        else:'''
        

    def train(self,max_epochs=5,gpus=-1):     
        trainer = pl.Trainer(gpus=gpus,callbacks=[self.checkpoint_callback],max_epochs=max_epochs)
        trainer.fit(self.model, train_dataloader=self.train_dataloader())

    def get_latent(self, deterministic=True,run_clustering=False,make_plot=False):
        print('Training done, generating embedding...')
        import matplotlib.pyplot as plt
        self.model.eval()
        embedding = []
        for bs, x,  adjs, ref in self.latent_dataloader():
            z,qm,_ = self.model.neighborhood_forward(x,adjs)
            if deterministic and self.model.apply_normal_latent:
                z = qm
            embedding.append(z.detach().numpy())
            
        self.embedding = np.concatenate(embedding)
        np.save(self.folder+'/loadings.npy',self.embedding)

        if run_clustering:
            self.data.clustering_scanpy(self.embedding)
    
        if make_plot:
            ### Plot spatial dots with assigned cluster
            fig=plt.figure(figsize=(6,6),dpi=1000)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_facecolor("black")
            width_cutoff = 1640 # um
            plt.scatter(self.data.df.x.compute(), self.data.df.y.compute(), c=self.data.dask_attrs['leiden'].compute().astype('int64'), s=0.2,marker='.',linewidths=0, edgecolors=None, cmap='rainbow')
            plt.xticks(fontsize=4)
            plt.yticks(fontsize=4)
            plt.axis('scaled')
            plt.savefig("{}/spatial_umap_embedding.png".format(self.folder), bbox_inches='tight', dpi=500)

    def make_umap(self,make_plot=True):
        print('Embedding done, generating umap and plots...')
        import matplotlib.pyplot as plt
        import umap

        reducer = umap.UMAP(
            n_neighbors=150,
            n_components=3,
            n_epochs=250,
            init='spectral',
            min_dist=0.1,
            spread=1,
            random_state=1,
            verbose=True,
            n_jobs=-1
        )
        umap_embedding = reducer.fit_transform(self.embedding)
        np.save(self.folder+'/umap.npy',umap_embedding)

        if make_plot:
            Y_umap = umap_embedding
            Y_umap -= np.min(Y_umap, axis=0)
            Y_umap /= np.max(Y_umap, axis=0)
            Y_umap.shape

            fig=plt.figure(figsize=(7,4),dpi=500)
            cycled = [0,1,2,0]
            for i in range(3):
                plt.subplot(1,3,i+1)
                plt.scatter(Y_umap[:,cycled[i]], Y_umap[:,cycled[i+1]], c=Y_umap,  s=5, marker='.', linewidths=0, edgecolors=None)
                plt.xlabel("Y"+str(cycled[i]))
                plt.ylabel("Y"+str(cycled[i+1]))
            plt.tight_layout()
            plt.savefig("{}/umap_embedding.png".format(self.folder), bbox_inches='tight', dpi=500)

    def molecules_df(self):
        rows,cols = [],[]
        filt = self.data.df.g.values.compute()[self.cells]
        for r in trange(self.data.unique_genes.shape[0]):
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
        if type(self.cells) == type(None):
            #self.cells = self.data.df.index.compute()
            self.cells = np.arange(self.data.shape[0])
        if type(self.subsample) == float and self.subsample < 1:
            self.cells = np.random.randint(0,self.data.shape[0], int(self.subsample*self.data.shape[0]))
        elif type(self.subsample) == dict:
            filt_x =  ((self.data.df.x > self.subsample['x'][0]) & (self.data.df.x < self.subsample['x'][1])).values.compute()
            filt_y =  ((self.data.df.y > self.subsample['y'][0]) & (self.data.df.y < self.subsample['y'][1])).values.compute()
            self.cells = self.cells[filt_x & filt_y]
            #self.cells = np.random.choice(self.data.df.index.compute(),size=int(subsample*self.data.shape[0]),replace=False)

    def buildGraph(self, d_th,coords=None):
        print('Building graph...')
        if type(coords)  == type(None):
            supervised = False
            edge_file = os.path.join(self.save_to,'Edges-{}Nodes-Ngh{}-{}-dst{}'.format(self.cells.shape[0],self.ngh_sizes[0],self.ngh_sizes[1],self.distance_threshold))
            tree_file = os.path.join(self.save_to,'Tree-{}Nodes-Ngh{}-{}-dst{}.ann'.format(self.cells.shape[0],self.ngh_sizes[0],self.ngh_sizes[1],self.distance_threshold))
            coords = np.array([self.data.df.x.values.compute()[self.cells], self.data.df.y.values.compute()[self.cells]]).T
            neighborhood_size = self.ngh_sizes[0] + 100
        else:
            supervised=True
            edge_file = os.path.join(self.save_to,'Supervised-Edges-{}Nodes-Ngh{}-{}-dst{}'.format(coords.shape[0],self.ngh_sizes[0],self.ngh_sizes[1],self.distance_threshold))
            tree_file = os.path.join(self.save_to,'Supervised-Tree-{}Nodes-Ngh{}-{}-dst{}.ann'.format(coords.shape[0],self.ngh_sizes[0],self.ngh_sizes[1],self.distance_threshold))
            neighborhood_size = self.ngh_sizes[0]

        if not os.path.isfile(edge_file):
            t = AnnoyIndex(2, 'euclidean')  # Length of item vector that will be indexed
            for i in trange(coords.shape[0]):
                v = coords[i,:]
                t.add_item(i, v)

            t.build(10) # 10 trees
            t.save(tree_file)
        
            def find_nn_distance(coords,tree,distance):
                print('Find neighbors below distance: {}'.format(d_th))
                res = []
                for i in trange(coords.shape[0]):
                    # 100 sets the number of neighbors to find for each node
                    #  it is set to 100 since we usually will compute neighbors
                    #  [20,10]
                    search = tree.get_nns_by_item(i, neighborhood_size, include_distances=True)
                    pair = [(i,n) for n,d in zip(search[0],search[1]) if d < distance]
                    if len(pair) > self.minimum_nodes_connected:
                        res += pair
                res= np.array(res)
                return res
            edges = find_nn_distance(coords,t,d_th)
                    
            with h5py.File(edge_file, 'w') as hf:
                hf.create_dataset("edges",  data=edges)

        else:
            with h5py.File(edge_file, 'r+') as hf:
                edges = hf['edges'][:]

        G = nx.Graph()
        G.add_nodes_from(np.arange(coords.shape[0]))
            # Add edges
        G.add_edges_from(edges)

        node_removed = []
        for component in tqdm(list(nx.connected_components(G))):
            if len(component) < self.minimum_nodes_connected:
                for node in component:
                    node_removed.append(node)
                    G.remove_node(node)

        edges = th.tensor(list(G.edges)).T
        cells = th.tensor(list(G.nodes))

        if supervised==False:
            #self.edges_tensor = edges
            self.molecules_connected = cells
            return edges
        else:
            return edges

    def cell_types_to_graph(self, data):
        """
        cell_types_to_graph [summary]

        Transform data (Ncells, genes) into fake molecule neighborhoods

        Args:
            data ([type]): [description]
            Ncells ([type]): [description]

        Returns:
            [type]: [description]
        """        
        all_molecules = []
        all_coords = []
        all_cl = []
        data = data/data.sum(axis=0)
        #data = (data*1000).astype('int')

        print('Converting clusters into simulated molecule neighborhoods...')
        for i in trange(data.shape[1]):
            molecules = []
            # Reduce number of cells by Ncells.min() to avoid having a huge dataframe, since it is actually simulated data
            cl_i = data[:,i]#*(Ncells[i]/(Ncells.min()*100)).astype('int')
            random_molecules = np.random.choice(data.shape[0],size=5000,p=cl_i)

            '''            
            for x in range(cl_i.shape[0]):
                dot = np.zeros_like(cl_i)
                dot[x] = 1
                try:
                    dot = np.stack([dot]*int(cl_i[x]))
                    molecules.append(dot)

                except:
                    pass
            '''

            for x in random_molecules:
                dot = np.zeros_like(cl_i)
                dot[x] = 1
                try:
                    #dot = np.stack([dot]*int(cl_i[x]))
                    molecules.append(dot)
                except:
                    pass

            molecules = np.stack(molecules)
            #molecules = np.concatenate(molecules)
            all_molecules.append(molecules)
            
            all_coords.append(np.random.normal(loc=i*1000,scale=25,size=[molecules.shape[0],2]))
            #all_coords.append(np.ones_like(molecules)*50*i)
            all_cl.append(np.ones(molecules.shape[0])*i)

        all_molecules = sparse.csr_matrix(np.concatenate(all_molecules))
        all_coords = np.concatenate(all_coords)
        all_cl = np.concatenate(all_cl)
        edges = self.buildGraph(5,coords=all_coords)
        print('Fake Molecules: ',all_molecules.shape)
        return all_molecules, edges, all_cl

    def knn_smooth(self,neighborhood_size=75):
        print('Smoothing neighborhoods with kernel size: {}'.format(neighborhood_size))
        
        u = AnnoyIndex(2, 'euclidean')
        u.load(os.path.join(self.save_to,'Tree-{}Nodes-Ngh{}-{}-dst{}.ann'.format(self.cells.shape[0],self.ngh_sizes[0],self.ngh_sizes[1],self.distance_threshold)))
        smoothed_dataframe = []
        molecules_connected = []
        for i in trange(self.d.shape[0]):
            search = u.get_nns_by_item(i, neighborhood_size, include_distances=True)
            neighbors = [n for n,d in zip(search[0],search[1]) if d < self.distance_threshold]

            try:
                rnd_neighbors = np.random.choice(neighbors, size=neighborhood_size,replace=False)
                smoothed_nn = self.d[rnd_neighbors,:].sum(axis=0)
                smoothed_dataframe.append(smoothed_nn)
                molecules_connected.append(i)
            except:
                smoothed_dataframe.append(self.d[i,:].toarray())

        smoothed_dataframe= np.concatenate(smoothed_dataframe)
        self.d = sparse.csr_matrix(smoothed_dataframe)
        self.molecules_connected = np.array(molecules_connected)