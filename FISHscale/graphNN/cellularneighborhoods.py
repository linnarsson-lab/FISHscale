from pyexpat import model
import torch as th
import numpy as np
from tqdm import trange, tqdm
import os
from typing import Optional
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import pandas as pd
import dgl
from FISHscale.graphNN.models import SAGELightning
from FISHscale.graphNN.graph_utils import GraphUtils, GraphPlotting
from FISHscale.graphNN.graph_decoder import GraphDecoder

from pyro.distributions.util import broadcast_shape
from torch.optim import Adam
from pyro.optim import Adam as AdamPyro
from pyro.infer.autoguide import init_to_mean
from pyro.infer import SVI, config_enumerate, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoNormal, AutoDelta
import logging
from annoy import AnnoyIndex
import scanpy as sc


class CellularNeighborhoods(pl.LightningDataModule, GraphPlotting, GraphDecoder):
    """
    Class to prepare the data for GraphSAGE

    """    
    def __init__(self,
        anndata, # Data as FISHscale Dataset (Molecules, Genes)
        genes,
        features_name='Expression',
        distance=500,
        model=None, # GraphSAGE model
        analysis_name:str='',
        ngh_sizes = [10, 5],
        minimum_nodes_connected = 2,
        train_p = 0.75,
        batch_size= 512,
        num_workers=0,
        save_to = '',
        subsample=1,
        lr=1e-3,
        aggregator='attentional',
        supervised=True,#'supervised'
        n_epochs=5,
        label_name='MolecularNgh',
        id_name = 'ID',
        distance_factor=5,
        normalize=True,
        ):
        """
        GraphData prepared the FISHscale dataset to be analysed in a supervised
        or unsupervised manner by non-linear GraphSage.

        Args:
            anndata (AnnData): AnnData with X and Y coords.
            genes (list): List of genes to be used as features.
            model (graphNN.model): GraphSage model to be used. If None GraphData
                will generate automatically a model with 24 latent variables.
                Defaults to None.
            analysis_name (str,optional): Name of the analysis folder.
            ngh_sizes (list, optional): GraphSage sampling size.
                Defaults to [20, 10].
            minimum_nodes_connected (int, optional): Minimum molecules connected
                in the network. Defaults to 5.
            train_p (float, optional): train_size is generally small if number 
                of molecules is large enough that information would be redundant. 
                Defaults to 0.25.
            batch_size (int, optional): Mini-batch for training. Defaults to 512.
            num_workers (int, optional): Dataloader parameter. Defaults to 0.
            save_to (str, optional): Folder where analysis will be saved. 
                Defaults to ''.
            device (str, optional): Device where pytorch is run. Defaults to 'gpu'.
            lr (float, optional): learning rate .Defaults to 1e-3.
            aggregator (str, optional). Aggregator type for SAGEConv. Choose 
                between 'pool','mean','min','lstm'. Defaults to 'pool'.
            celltype_distribution (str, optional). Supervised cell_type/
                molecules distribution. Choose between 'uniform','ascending'
                or 'cell'. Defaults to 'uniform'.

        """
        super().__init__()

        self.genes = genes
        self.features_name = features_name
        self.distance = distance
        self.model = model
        self.analysis_name = analysis_name
        self.ngh_sizes = ngh_sizes
        self.minimum_nodes_connected = minimum_nodes_connected
        self.train_p = train_p
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_to = save_to
        self.lr = lr
        self.subsample = subsample
        self.aggregator = aggregator
        self.n_epochs= n_epochs
        self.unique_samples = np.unique(anndata.obs['Sample'].values)
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.label_name = label_name
        self.supervised = supervised
        self.id_name = id_name
        self.distance_factor = distance_factor

        self.unique_labels = np.unique(anndata.obs[self.label_name].values)
        anndata = anndata[(anndata[:, self.genes].X.sum(axis=1) > 5), :]
        anndata.raw = anndata
        if normalize:
            sc.pp.normalize_total(anndata, target_sum=1e4)
            sc.pp.log1p(anndata)
        self.anndata = anndata

        ### Model hyperparameters
        if supervised:
            in_feats= len(self.genes)
            n_latents = self.unique_labels.shape[0]
            n_hidden = 128
            loss_type = 'supervised'
        
        elif supervised == False and self.features_name == 'Expression':
            in_feats= len(self.genes)
            n_latents = 24
            n_hidden = 128
            loss_type = 'unsupervised'
        else:
            in_feats= self.unique_labels.max()+1
            n_latents = 24
            n_hidden = 64
            loss_type = 'unsupervised'
        self.n_latents = n_latents
        ###

        ### Prepare data
        self.folder = self.save_to+self.analysis_name+ '_' +datetime.now().strftime("%Y-%m-%d-%H%M%S")
        os.mkdir(self.folder)
        if not os.path.isdir(self.save_to+'graph'):
            os.mkdir(self.save_to+'graph')
        self.setup()
        
        dgluns = self.save_to+'graph/CellularNeighborhoods{}_features{}_dst{}_mNodes{}_nlatents{}.graph'.format(
            self.anndata.shape[0],
            self.features_name,
            self.distance_factor,
            self.minimum_nodes_connected,
            self.n_latents)
        if not os.path.isfile(dgluns):
            subgraphs = []
            for sample in tqdm(self.unique_samples):
                adata = self.anndata[self.anndata.obs['Sample']==sample,:]
                if self.features_name == 'Expression':
                    if type(adata.X) == np.ndarray:
                        features = adata[:,self.genes].X#.toarray()
                    else: #is sparse
                        features = adata[:,self.genes].X.toarray()
                
                else:
                    features = adata.obs[self.features_name].values

                labels = adata.obs[self.label_name].values

                g = self.buildGraph(adata, labels, features, d_th =self.distance)
                g.ndata['sample'] = th.ones(g.ndata[self.features_name].shape[0]) * np.where(self.unique_samples==sample)[0][0]
                
                
                subgraphs.append(g)
            
            graph_labels = {"Multigraph": th.arange(len(subgraphs))}
            logging.info('Saving graph...')
            dgl.data.utils.save_graphs(dgluns, subgraphs, graph_labels)
            logging.info('Graph saved.')
        else:
            logging.info('Loading graph.')
            subgraphs, graph_labels = dgl.data.utils.load_graphs(dgluns) # glist will be [g1, g2]
            logging.info('Graph data loaded.')

        self.g = dgl.batch(subgraphs)
        
        if self.aggregator == 'attentional':
            # Remove all self loops because data will be saved, otherwise the 
            # number of self-loops will be doubled with each new run.
            remove = dgl.RemoveSelfLoop()
            self.g = remove(self.g)
            self.g = dgl.add_self_loop(self.g)

        logging.info(self.g)
        #self.make_train_test_validation()
        l_loc,l_scale= 0,1
        
        ### Prepare Model

        
        if type(self.model) == type(None):
            self.model = SAGELightning(
                                        in_feats=in_feats, 
                                        n_latent=n_latents,
                                        n_layers=len(self.ngh_sizes),
                                        n_classes=n_latents,
                                        n_hidden=n_hidden,
                                        lr=self.lr,
                                        supervised=self.supervised,
                                        device=self.device.type,
                                        aggregator=self.aggregator,
                                        loss_type=loss_type,
                                        features_name=self.features_name,
                                        
                                    
                                    )

        self.model.to(self.device)

        if 'clusters' in self.g.ndata.keys():
            self.clusters = self.g.ndata['clusters']

    def prepare_data(self):
        # do-something
        pass
    
    def save_graph(self):
        dgluns = self.save_to+'graph/CellularNeighborhoods{}_features{}_dst{}_mNodes{}_nlatents{}.graph'.format(
            self.anndata.shape[0],
            self.features_name,
            self.distance_factor,
            self.minimum_nodes_connected,
            self.n_latents)
        subgraphs = dgl.unbatch(self.g)
        graph_labels = {"Multigraph": th.arange(len(subgraphs))}
        logging.info('Saving graph...')
        dgl.data.utils.save_graphs(dgluns, subgraphs, graph_labels)
        logging.info('Graph saved.')


    def setup(self, stage: Optional[str] = None):
        #self.d = th.tensor(self.molecules_df(),dtype=th.float32) #works
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler([int(_) for _ in self.ngh_sizes])

        self.checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=self.save_to,
            filename=self.analysis_name+'-{epoch:02d}-{train_loss:.2f}',
            save_top_k=1,
            mode='min',
            )
        self.early_stop_callback = EarlyStopping(
            monitor='balance',
            patience=3,
            verbose=True,
            mode='min',
            stopping_threshold=0.35,
            )

    def train_dataloader(self):
        """
        train_dataloader

        Prepare dataloader

        Returns:
            dgl.dataloading.EdgeDataLoader: Deep Graph Library dataloader.
        """        
        if self.supervised:
            train_p_nodes = int(self.g.num_nodes()*(self.train_p))
            train_nodes = th.randperm(self.g.num_nodes())[:train_p_nodes]
            
            node_sampler = dgl.dataloading.NeighborSampler(
                [int(_) for _ in self.ngh_sizes],
                prefetch_node_feats=[self.features_name],
                prefetch_labels=['label']
            )

            unlab = dgl.dataloading.DataLoader(
                            self.g,
                            train_nodes,
                            node_sampler,
                            device=self.device,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=self.num_workers,
                            )
            return [unlab]

        else:
            train_p_edges = int(self.g.num_edges()*(self.train_p))
            train_edges = th.randperm(self.g.num_edges())[:train_p_edges]
            negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)
            
            edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
                    dgl.dataloading.NeighborSampler(
                        [int(_) for _ in self.ngh_sizes],
                        prefetch_node_feats=[self.features_name],
                        prefetch_labels=['label']
                        
                        ),
                    negative_sampler=negative_sampler,
                )

            unlab = dgl.dataloading.DataLoader(
                            self.g,
                            train_edges,
                            edge_sampler,
                            device=self.device,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=self.num_workers,
                            )
            return [unlab]


    def train(self,gpus=0, model_file=None, continue_training=False):
        """
        train

        Pytorch-Lightning trainer

        Args:
            max_epochs (int, optional): Maximum epochs. Defaults to 5.
            gpus (int, optional): Whether to use GPU. Defaults to 0.
        """        
        if self.device.type == 'cuda':
            gpus=1
        trainer = pl.Trainer(gpus=gpus,
                            log_every_n_steps=50,
                            callbacks=[self.checkpoint_callback], 
                            max_epochs=self.n_epochs,)

        is_trained = 0
        for File in os.listdir(os.path.join(self.folder, '..')):
            if File.count('.ckpt'):
                is_trained = 1
                model_path = os.path.join(self.save_to, File)
                break
        
        if type(model_file) != type(None):
            is_trained = 1
            model_path = model_file

        if is_trained:
            logging.info('Pretrained model exists in folder {}, loading model parameters. If you wish to re-train, delete .ckpt file.'.format(model_path))
            
            if self.supervised:
                in_feats= len(self.genes)
                n_latents = self.unique_labels.shape[0]
                n_hidden = 128
                loss_type = 'supervised'
        
            elif self.supervised == False and self.features_name == 'Expression':
                in_feats= len(self.genes)
                n_latents = 10
                n_hidden = 128
                loss_type = 'unsupervised'
            else:
                in_feats= self.unique_labels.max()+1
                n_latents = 10
                n_hidden = 24
                loss_type = 'unsupervised'
            
            self.model = self.model.load_from_checkpoint(
                                        model_path,
                                        in_feats=in_feats, 
                                        n_latent=n_latents,
                                        n_layers=len(self.ngh_sizes),
                                        n_classes=n_latents,
                                        n_hidden=n_hidden,
                                        lr=self.lr,
                                        supervised=self.supervised,
                                        device=self.device.type,
                                        aggregator=self.aggregator,
                                        loss_type=loss_type,
                                        features_name=self.features_name,
                                        )
            
        if continue_training or not is_trained:
            trainer.fit(self.model, train_dataloaders=self.train_dataloader())#,val_dataloaders=self.test_dataloader())

    def get_saved_latents(self):
        if type(self.g.ndata['h']) != type(None):
            logging.info('Latents already computed. Loading...')
            self.latent_unlabelled = self.g.ndata['h']
        else:
            self.get_latents()


    def get_latents(self):
        """
        get_latents: get the new embedding for each molecule
        
        Passes the validation data through the model to generatehe neighborhood 
        embedding. If the model is in supervised version, the model will also
        output the predicted cell type.

        Args:
            labelled (bool, optional): [description]. Defaults to True.
        """        
        self.model.eval()

        self.latent_unlabelled, prediction_unlabelled = self.model.module.inference(self.g,
                        self.model.device,
                        10*512,
                        0)


        self.save_graph()

    def get_attention(self):
        """
        get_latents: get the new embedding for each molecule
        
        Passes the validation data through the model to generate the neighborhood 
        embedding. If the model is in supervised version, the model will also
        output the predicted cell type.

        Args:
            labelled (bool, optional): [description]. Defaults to True.
        """        
        self.model.eval()
        self.attention_ngh1, self.attention_ngh2 = self.model.module.inference_attention(self.g,
                        self.model.device,
                        5*512,
                        0,
                        buffer_device=self.g.device)#.detach().numpy()
        self.g.edata['attention1'] = self.attention_ngh1
        self.g.edata['attention2'] = self.attention_ngh2
        self.save_graph()

    def get_attention_nodes(self,nodes=None):
        """
        get_latents: get the new embedding for each molecule
        
        Passes the validation data through the model to generate the neighborhood 
        embedding. If the model is in supervised version, the model will also
        output the predicted cell type.

        Args:
            labelled (bool, optional): [description]. Defaults to True.
        """        
        self.model.eval()
        att1,att2 =  self.model.module.inference_attention(self.g,
                        self.model.device,
                        5*512,
                        0,
                        nodes=nodes,
                        buffer_device=self.g.device)#.detach().numpy()
        return att1, att2

    def compute_distance_th(self,coords):
        """
        compute_distance_th: deprecated, now inside BuildGraph

        Computes the distance at which 95 percentile molecules are connected to
        at least 2 other molecules. Like PArtel & Wahlby

        Args:
            omega ([type]): [description]
            tau ([type]): [description]
        """        

        from scipy.spatial import cKDTree as KDTree
        kdT = KDTree(coords)
        d,i = kdT.query(coords,k=3)
        d_th = np.percentile(d[:,-1],95)*self.distance_factor
        logging.info('Chosen dist to connect molecules into a graph: {}'.format(d_th))
        print('Chosen dist to connect molecules into a graph: {}'.format(d_th))
        return d_th
        

    def buildGraph(self, adata, labels, features='Expression', d_th=500, coords=None):
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
        coords = np.array([adata.obs.X.values, adata.obs.Y.values]).T
        neighborhood_size = 100

        t = AnnoyIndex(2, 'euclidean')  # Length of item vector that will be indexed
        for i in trange(coords.shape[0]):
            v = coords[i,:]
            t.add_item(i, v)

        t.build(10) # 10 trees
        d_th = self.compute_distance_th(coords)
        distance_threshold = d_th
        #distance_threshold = d_th
        logging.info('Chosen dist: {}'.format(distance_threshold))
        
        def find_nn_distance(coords,tree,distance):
            logging.info('Find neighbors below distance: {}'.format(d_th))
            res,nodes,ngh_, ncoords = [],[],[], []
            for i in range(coords.shape[0]):
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

        d = features
        edges, molecules, ngh_ = find_nn_distance(coords, t, distance_threshold)
        d= d[molecules]
        #d = self.molecules_df(molecules)
        g= dgl.graph((edges[0,:],edges[1,:]),)
        #g = dgl.to_bidirected(g)]
        g.ndata[self.features_name] = th.tensor(d) #, dtype=th.int16)#
        g.ndata['label'] = th.tensor(labels[molecules], dtype=th.uint8)
        g.ndata[self.id_name] = th.tensor(adata.obs[self.id_name].values.astype(np.int64))

        sum_nodes_connected = th.tensor(np.array(ngh_,dtype=np.uint8))
        print('Number of edges for sample: {}'.format(g.num_edges()))
        molecules_connected = molecules[sum_nodes_connected >= self.minimum_nodes_connected]
        remove = molecules[sum_nodes_connected < self.minimum_nodes_connected]
        g.remove_nodes(th.tensor(remove))
        g.ndata['indices'] = th.tensor(molecules_connected.clone().detach())
        g.ndata['coords'] = th.tensor(coords[molecules_connected])
        return g

    def cluster(self, n_clusters=10):
        """
        cluster: cluster the latent space.

        Clusters the latent space using MiniBatchKMeans. The number of clusters
        is set to 10 by default.

        Args:
            training_latents ([type]): [description]

        Returns:
            [type]: [description]
        """
        from sklearn.cluster import MiniBatchKMeans
        
        clusters = MiniBatchKMeans(n_clusters=n_clusters).fit_predict(self.latent_unlabelled)
        self.g.ndata['CellularNgh'] = th.tensor(clusters)
        self.save_graph()
        return clusters

                
