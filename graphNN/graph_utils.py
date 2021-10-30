import networkx as nx
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
import dgl.function as fn
from FISHscale.graphNN.models import SAGELightning

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
        model=None, # GraphSAGE model
        analysis_name:str='',
        cells=None, # Array with cell_ids of shape (Cells)
        ngh_size = 40,
        minimum_nodes_connected = 5,
        ngh_sizes = [20, 10],
        train_p = 0.25,
        batch_size= 512,
        num_workers=0,
        save_to = '',
        subsample=1,
        ref_celltypes=None,
        exclude_clusters:dict={},
        smooth:bool=True,
        negative_samples:int=1,
        distance_factor:int=4,
        device='cpu',
        lr=1e-3,
        ):
        """
        GraphData prepared the FISHscale dataset to be analysed in a supervised
        or unsupervised manner by non-linear GraphSage.

        [extended_summary]

        Args:
            data (FISHscale.dataset): FISHscale DataSet object. Contains 
                x,y coordinates and gene identity.
            model (graphNN.model): GraphSage model to be used. If None GraphData
                will generate automatically a model with 24 latent variables.
                Defaults to None.
            cells (np.array, optional): Array of molecules to be kept for 
                analysis. Defaults to None.
            minimum_nodes_connected (int, optional): Minimum molecules connected
                in the network. Defaults to 5.
            ngh_sizes (list, optional): GraphSage sampling size. 
                Defaults to [20, 10].
            train_p (float, optional): train_size is generally small if number of 
                molecules is large enough that information would be redundant 
                especially if smoothing is performed. Defaults to 0.25.
            batch_size (int, optional): Mini-batch for training. Defaults to 512.
            num_workers (int, optional): Dataloader parameter. Defaults to 0.
            save_to (str, optional): Folder where analysis will be saved. 
                Defaults to ''.
            subsample (float/dict, optional): Float between 0 and 1 to subsample
                a fraction of molecules that will be used to make the network. 
                Or dictionary with the squared region to subsample.
                Defaults to 1.
            ref_celltypes (loomfile, optional): Rerence data of cell types for 
                supervised analysis. Contains mean expression for gene/cell type
                and must contain ds.ca['Ncells'] and ds.ca['ClusterNames'] 
                attributes. Defaults to None.
            exclude_clusters (list, optional): List of clusters present in 
                ds.ca['ClusterNames'] to exclude. Defaults to [''].
            smooth (bool, optional): Smooth knn in the network data. Improves 
                both supervised and unsupervised network performance.
                Defaults to True.
            negative_samples (int, optional): Negative samples taken by 
                GraphSage sampler. Defaults to 5.
            distance_factor (float, optional): Distance selected to construct 
                network graph will be multiplied by the distance factor. 
                Defaults to 3.
            device (str, optional): Device where pytorch is run. Defaults to 'cpu'.
            lr (float, optional): learning rate .Defaults to 1e-3.
        """
        super().__init__()

        self.model = model
        self.analysis_name = analysis_name
        self.ngh_sizes = ngh_sizes
        self.data = data
        self.cells = cells
        self.minimum_nodes_connected = minimum_nodes_connected
        self.train_p = train_p
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_to = save_to
        self.ref_celltypes = ref_celltypes
        self.exclude_clusters = exclude_clusters
        self.smooth = smooth
        self.negative_samples = negative_samples
        self.ngh_size = ngh_size
        self.lr = lr
        self.subsample = subsample

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prepare_reference()
        if type(self.model) == type(None):
            self.model = SAGELightning(in_feats=self.data.unique_genes.shape[0], 
                                        n_hidden=24,
                                        n_layers=2,
                                        n_classes=self.ref_celltypes.shape[1],
                                        lr=self.lr,
                                        supervised=self.supervised,
                                        Ncells=self.ncells*self.ref_celltypes.sum(axis=0),
                                        reference=self.ref_celltypes,
                                        device=self.device.type
                                    )
        self.model.to(self.device)
        print('model is in: ', self.model.device)

        self.folder = self.save_to+self.analysis_name+ '_' +datetime.now().strftime("%Y-%m-%d-%H%M%S")
        os.mkdir(self.folder)
        if not os.path.isdir(self.save_to+'graph'):
            os.mkdir(self.save_to+'graph')
        #print('Device is: ',self.device)
    
        self.compute_distance_th(distance_factor)
        self.subsample_xy()
        self.compute_size()
        self.setup()
        # Save random cell selection
        
        dgluns = self.save_to+'graph/{}Unsupervised_smooth{}.graph'.format(self.cells.shape[0],self.smooth)
        if not os.path.isfile(dgluns):
            d = self.molecules_df()
            edges = self.buildGraph(self.distance_threshold)
            self.g= dgl.graph((edges[0,:],edges[1,:]))
            self.g.ndata['gene'] = th.tensor(d.toarray(), dtype=th.float32)
            graph_labels = {"UnsupervisedDGL": th.tensor([0])}
            if self.smooth:
                #self.g.update_all(fn.u_add_v('gene','gene','a'),fn.sum('a','gene'))
                self.g.update_all(fn.copy_u('gene', 'm'), fn.sum('m', 'gene'))
            #self.g.update_all(fn.copy_u('gene', 'g2'), fn.sum('g2', 'gene'))
            dgl.data.utils.save_graphs(dgluns, [self.g], graph_labels)
            #self.g = self.g.to(self.device)
        else:
            glist, _ = dgl.data.utils.load_graphs(dgluns) # glist will be [g1, g2]
            self.g = glist[0]
            #self.g = self.g.to(self.device)

        if self.model.supervised:
            dglsup =self.save_to+'graph/{}Supervised_smooth{}.graph'.format(self.cells.shape[0],self.smooth)
            if not os.path.isfile(dglsup):
                molecules_labelled, edges_labelled, labels = self.cell_types_to_graph(smooth=self.smooth)
                self.g_lab= dgl.graph((edges_labelled[0,:],edges_labelled[1,:]))
                self.g_lab.ndata['gene'] = th.tensor(molecules_labelled.toarray(),dtype=th.float32)
                self.g_lab.ndata['label'] = th.tensor(labels, dtype=th.long)
                graph_labels = {"SupervisedDGL": th.tensor([0])}
                dgl.data.utils.save_graphs(dglsup, [self.g_lab], graph_labels)
                #self.g_lab = self.g_lab.to(self.device)
            else:
                glist, _ = dgl.data.utils.load_graphs(dglsup) # glist will be [g1, g2]
                self.g_lab = glist[0]
                #self.g_lab = self.g_lab.to(self.device)
        
        print(self.g)

    def prepare_data(self):
        # do-something
        pass

    def setup(self, stage: Optional[str] = None):
        #self.d = th.tensor(self.molecules_df(),dtype=th.float32) #works
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler([int(_) for _ in self.ngh_sizes])

        self.checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=self.folder,
            filename=self.analysis_name+'-{epoch:02d}-{train_loss:.2f}',
            save_top_k=5,
            mode='min',
            )
        self.early_stop_callback = EarlyStopping(
            monitor='balance',
            patience=3,
            verbose=True,
            mode='min',
            stopping_threshold=0.35,
            )

    def compute_size(self):
        cells = self.cells
        #if self.smooth:
        #    cells = self.molecules_connected
        np.save(self.folder +'/cells.npy', cells)
        self.train_size = int((cells.shape[0])*self.train_p)
        self.test_size = cells.shape[0]-int(cells.shape[0]*self.train_p)  
        random_state = np.random.RandomState(seed=0)
        permutation = random_state.permutation(cells.shape[0])
        self.indices_test = th.tensor(permutation[:self.test_size])
        self.indices_train = th.tensor(permutation[self.test_size : (self.test_size + self.train_size)])
        self.indices_validation = th.tensor(np.arange(cells.shape[0]))

    def train_dataloader(self):
        edges = np.arange(self.g.num_edges())
        random_edges = torch.tensor(np.random.choice(edges,int(edges.shape[0]*(self.train_p/(self.g.num_edges()/self.g.num_nodes()))),replace=False))
        unlab = dgl.dataloading.EdgeDataLoader(
                        self.g,
                        random_edges,
                        self.sampler,
                        negative_sampler=dgl.dataloading.negative_sampler.Uniform(self.negative_samples), # NegativeSampler(self.g, self.negative_samples, False),
                        device=self.device,
                        #exclude='self',
                        #reverse_eids=th.arange(self.g.num_edges()) ^ 1,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=self.num_workers,
                        )

        if self.model.supervised:
            edges = np.arange(self.g_lab.num_edges())
            random_edges = torch.tensor(np.random.choice(edges,random_edges.shape[0],replace=True))
            lab = dgl.dataloading.EdgeDataLoader(
                            self.g_lab,
                            random_edges,
                            self.sampler,
                            negative_sampler=dgl.dataloading.negative_sampler.Uniform(self.negative_samples), # NegativeSampler(self.g, self.negative_samples, False),
                            device=self.device,
                            #exclude='self',
                            #reverse_eids=th.arange(self.g.num_edges()) ^ 1,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=self.num_workers)
        else:
            lab = None
        if type(lab) != type(None):
            return {'unlabelled':unlab,'labelled':lab}
        else:
            return {'unlabelled':unlab}

    def train(self,max_epochs=5,gpus=0):
        if self.device.type == 'cuda':
            gpus=1
        if self.model.supervised: 
            trainer = pl.Trainer(gpus=gpus,callbacks=[self.checkpoint_callback], max_epochs=max_epochs)
        else:
            trainer = pl.Trainer(gpus=gpus,callbacks=[self.checkpoint_callback], max_epochs=max_epochs)
        trainer.fit(self.model, train_dataloaders=self.train_dataloader())

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

    def compute_distance_th(self,omega):
        from scipy.spatial import cKDTree as KDTree
        from matplotlib import pyplot as plt
        x,y = self.data.df.x.values.compute(),self.data.df.y.values.compute()
        kdT = KDTree(np.array([x,y]).T)
        d,i = kdT.query(np.array([x,y]).T,k=2)
        d_th = np.percentile(d[:,1],97)*omega
        self.distance_threshold = d_th
        print('Chosen dist: {}'.format(d_th))

    def buildGraph(self, d_th,coords=None):
        print('Building graph...')
        if type(coords)  == type(None):
            supervised = False
            edge_file = os.path.join(self.save_to,'graph/DGL-Edges-{}Nodes-dst{}'.format(self.cells.shape[0],self.distance_threshold))
            tree_file = os.path.join(self.save_to,'graph/DGL-Tree-{}Nodes-dst{}.ann'.format(self.cells.shape[0],self.distance_threshold))
            coords = np.array([self.data.df.x.values.compute()[self.cells], self.data.df.y.values.compute()[self.cells]]).T
            neighborhood_size = self.ngh_size
        else:
            supervised=True
            edge_file = os.path.join(self.save_to,'graph/DGL-Supervised-Edges-{}Nodes-dst{}'.format(coords.shape[0],self.distance_threshold))
            tree_file = os.path.join(self.save_to,'graph/DGL-Supervised-Tree-{}Nodes-dst{}.ann'.format(coords.shape[0],self.distance_threshold))
            neighborhood_size = self.ngh_size

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
            self.molecules_connected = cells
            return edges
        else:
            return edges

    def cell_types_to_graph(self,smooth=False):
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
        data = self.ref_celltypes#/data.sum(axis=0)

        #data = (data*1000).astype('int')
        print('Converting clusters into simulated molecule neighborhoods...')
        for i in trange(data.shape[1]):
            molecules = []
            # Reduce number of cells by Ncells.min() to avoid having a huge dataframe, since it is actually simulated data
            cl_i = data[:,i]#*(Ncells[i]/(Ncells.min()*100)).astype('int')
            if smooth == False:
                random_molecules = np.random.choice(data.shape[0],size=2500,p=cl_i/cl_i.sum())
                for x in random_molecules:
                    dot = np.zeros_like(cl_i)
                    dot[x] = 1
                    try:
                        #dot = np.stack([dot]*int(cl_i[x]))
                        molecules.append(dot)
                    except:
                        pass
            else:
                for x in range(2500):
                    p = np.random.poisson(cl_i,size=(1,cl_i.shape[0]))[0,:]
                    p[p < 0] = 0
                    p = p/p.sum()
                    random_molecules = np.random.choice(data.shape[0],size=25,p=p)
                    dot = np.zeros_like(cl_i)
                    for x in random_molecules:
                        dot[x] = dot[x]+1
                    molecules.append(dot)

            molecules = np.stack(molecules)
            #molecules = np.concatenate(molecules)
            all_molecules.append(molecules)
            
            all_coords.append(np.random.normal(loc=i*1000,scale=25,size=[molecules.shape[0],2]))
            #all_coords.append(np.ones_like(molecules)*50*i)
            all_cl.append(np.ones(molecules.shape[0])*i)

        all_molecules = sparse.csr_matrix(np.concatenate(all_molecules))
        all_coords = np.concatenate(all_coords)
        all_cl = np.concatenate(all_cl)
        edges = self.buildGraph(75,coords=all_coords)
        print('Fake Molecules: ',all_molecules.shape)
        return all_molecules, edges, all_cl

    def prepare_reference(self):
        if type(self.ref_celltypes) != type(None):
            self.supervised = True
            import loompy
            with loompy.connect(self.ref_celltypes,'r') as ds:
                print(ds.ca.keys())
                k = list(self.exclude_clusters.keys())[0]
                v = self.exclude_clusters[k]
                region_filt = np.isin(ds.ca[k], v, invert=True)
                self.ClusterNames = ds.ca[k][region_filt]
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
        else:
            self.supervised = False
            self.ref_celltypes = np.array([[0],[0]])
            self.ncells = 0

        #do something

    '''def knn_smooth(self,neighborhood_size=75):
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
        self.molecules_connected = np.array(molecules_connected)'''

    #### plotting and latent factors #####

    def get_latents(self,labelled=True):
        self.model.eval()
        latent_unlabelled = self.model.module.inference(self.g,self.g.ndata['gene'],'cpu',512,0)#.detach().numpy()
        
        if self.model.supervised:
            if labelled:
                latent_labelled = self.model.module.inference(self.g_lab,self.g_lab.ndata['gene'],'cpu',512,0)#.detach().numpy()
                self.prediction_labelled = self.model.module.encoder.encoder_dict['CF'](latent_labelled).detach().numpy()
                #np.save(self.folder+'/probabilities_labelled',self.prediction_labelled)
                self.latent_labelled = latent_labelled.detach().numpy()
                #np.save(self.folder+'/latent_labelled',self.latent_labelled)
            self.prediction_unlabelled = self.model.module.encoder.encoder_dict['CF'](latent_unlabelled).detach().numpy()
            np.save(self.folder+'/labels_unlabelled',self.prediction_unlabelled.argsort(axis=-1)[:,-1].astype('str'))
            np.save(self.folder+'/probabilities_unlabelled',self.prediction_unlabelled)

        self.latent_unlabelled = latent_unlabelled.detach().numpy()
        np.save(self.folder+'/latent_unlabelled',latent_unlabelled)


    def get_umap(self,random_n=50000):
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

            mixed = np.concatenate([self.latent_unlabelled,self.latent_labelled])
            batch = np.concatenate([np.zeros(self.latent_unlabelled.shape[0]),np.ones(self.latent_labelled.shape[0])])
            some_mixed = np.random.choice(np.arange(mixed.shape[0]),int(random_n/2),replace=False)
            print(some_mixed.shape,batch.shape)
            umap_embedding = reducer.fit(mixed[some_mixed])
            embedding = umap_embedding.transform(mixed)
            Y_umap_mixed = embedding
            Y_umap_mixed -= np.min(Y_umap_mixed, axis=0)
            Y_umap_mixed /= np.max(Y_umap_mixed, axis=0)

            fig=plt.figure(figsize=(7,4),dpi=500)
            cycled = [0,1,2,0]
            for i in range(3):
                plt.subplot(1,3,i+1)
                plt.scatter(Y_umap_mixed[:,cycled[i]], Y_umap_mixed[:,cycled[i+1]], c=batch,  s=0.25, marker='.', linewidths=0, edgecolors=None)
                plt.xlabel("Y"+str(cycled[i]))
                plt.ylabel("Y"+str(cycled[i+1]))
            plt.tight_layout()
            plt.savefig("{}/umap_supervised.png".format(self.folder), bbox_inches='tight', dpi=500)

            some = np.random.choice(np.arange(self.latent_unlabelled.shape[0]),random_n,replace=False)
            #umap_embedding = reducer.fit_transform(self.latent_unlabelled[some])
            umap_embedding = reducer.fit(self.latent_unlabelled[some])
            embedding = umap_embedding.transform(self.latent_unlabelled)
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
            #plt.scatter(DS.df.x.values.compute()[GD.cells], DS.df.y.values.compute()[GD.cells], c=torch.argmax(pred.softmax(dim=-1),dim=-1).numpy(), s=0.2,marker='.',linewidths=0, edgecolors=None,cmap='rainbow')
            plt.scatter(self.data.df.x.values.compute()[self.cells], self.data.df.y.values.compute()[self.cells], c=Y_umap, s=0.05,marker='.',linewidths=0, edgecolors=None)
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
                plt.scatter(Y_umap[:,cycled[i]], Y_umap[:,cycled[i+1]], c=clusters_colors,  s=2, marker='.', linewidths=0, edgecolors=None,cmap='rainbow')
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
            plt.scatter(self.data.df.x.values.compute()[self.cells], self.data.df.y.values.compute()[self.cells], c=clusters_colors, alpha=0.9,s=0.05,marker='.',linewidths=0, edgecolors=None)
            plt.xticks(fontsize=2)
            plt.yticks(fontsize=2)
            plt.axis('scaled')
            plt.savefig("{}/spatial_umap_embedding_clusters.png".format(self.folder), bbox_inches='tight', dpi=5000)

            import holoviews as hv
            hv.extension('matplotlib')
            molecules_y = self.data.df.y.values.compute()[self.cells]
            molecules_x = self.data.df.x.values.compute()[self.cells]
            nd_dic = {}

            allm = 0
            print('Generating plots for cluster assigned to molecules...')
            for cl in range(self.ClusterNames.shape[0]):
                    try:
                        x, y = molecules_x[clusters == cl], molecules_y[clusters == cl]
                        allm += x.shape[0]
                        color = ['red']*x.shape[0]
                        nd_dic[self.ClusterNames[cl]] = hv.Scatter(np.array([x,y]).T).opts(
                            bgcolor='black',
                            aspect='equal',
                            fig_inches=10,
                            s=1,
                            title=self.ClusterNames[cl],
                            color=color_dic[cl])
                    except:
                        pass

            layout = hv.Layout([nd_dic[x] for x in nd_dic]).cols(5)
            hv.save(layout,"{}/molecule_prediction.png".format(self.folder))

            pred_labels = torch.tensor(self.prediction_unlabelled)
            merge = np.concatenate([molecules_x[:,np.newaxis],molecules_y[:,np.newaxis]],axis=1)
            L = []
            print('Generating plots for molecule cluster probabilities...')
            for n in range(self.ClusterNames.shape[0]):
                try:
                    ps = pred_labels.softmax(axis=-1).detach().numpy()[:,n][:,np.newaxis]
                    scatter= hv.Scatter(np.concatenate([merge,ps],axis=1)[:,ps >0.5],
                                        kdims=['x','y'],vdims=[self.ClusterNames[n]]).opts(cmap='Viridis',
                                                                                            color=hv.dim(str(self.ClusterNames[n])),
                                                                                            s=1,
                                                                                            aspect='equal',
                                                                                            bgcolor='black',
                                                                                            fig_inches=10,
                                                                                            title=self.ClusterNames[n])
                    L.append(scatter)
                except:
                    print('Skipping cluster: ', self.ClusterNames[n])
                    pass
            layout = hv.Layout(L).cols(5)
            print('Saving cluster probabilities...')
            hv.save(layout,"{}/cluster_probabilities.png".format(self.folder))
            print('Plots saved.')

        else:
            import scanpy as sc
            print('Running leiden clustering from scanpy...')
            adata = sc.AnnData(X=self.latent_unlabelled)
            sc.pp.neighbors(adata, n_neighbors=15)
            sc.tl.leiden(adata, random_state=42)
            #self.data.add_dask_attribute('leiden',adata.obs['leiden'].values.tolist())
            self.clusters= adata.obs['leiden'].values.astype('int')
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
            umap_embedding = reducer.fit(self.latent_unlabelled[some])
            embedding = umap_embedding.transform(self.latent_unlabelled)
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
            #plt.scatter(DS.df.x.values.compute()[GD.cells], DS.df.y.values.compute()[GD.cells], c=torch.argmax(pred.softmax(dim=-1),dim=-1).numpy(), s=0.2,marker='.',linewidths=0, edgecolors=None,cmap='rainbow')
            plt.scatter(self.data.df.x.values.compute()[self.cells], self.data.df.y.values.compute()[self.cells], c=clusters_colors, s=0.05,marker='.',linewidths=0, edgecolors=None)
            plt.xticks(fontsize=2)
            plt.yticks(fontsize=2)
            plt.axis('scaled')
            plt.savefig("{}/spatial_umap_embedding.png".format(self.folder), bbox_inches='tight', dpi=5000)
