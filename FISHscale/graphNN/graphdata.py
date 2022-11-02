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


class GraphData(pl.LightningDataModule, GraphUtils, GraphPlotting, GraphDecoder):
    """
    Class to prepare the data for GraphSAGE

    """    
    def __init__(self,
        data, # Data as FISHscale Dataset (Molecules, Genes)
        model=None, # GraphSAGE model
        analysis_name:str='',
        molecules=None, # Array with molecules_ids of shape (molecules)
        ngh_size = 100,
        ngh_sizes = [20, 10],
        minimum_nodes_connected = 5,
        fraction_edges = 10,
        train_p = 0.75,
        batch_size= 512,
        num_workers=0,
        save_to = '',
        subsample=1,
        ref_celltypes=None,
        exclude_clusters:dict={},
        smooth:bool=False,
        negative_samples:int=5,
        distance_factor:int=2,
        lr=1e-3,
        aggregator='attentional',
        celltype_distribution='uniform',
        inference_type='deterministic',
        model_type='unsupervised',#'supervised'
        n_epochs=5,
        ):
        """
        GraphData prepared the FISHscale dataset to be analysed in a supervised
        or unsupervised manner by non-linear GraphSage.

        Args:
            data (FISHscale.dataset): FISHscale DataSet object. Contains 
                x,y coordinates and gene identity.
            model (graphNN.model): GraphSage model to be used. If None GraphData
                will generate automatically a model with 24 latent variables.
                Defaults to None.
            analysis_name (str,optional): Name of the analysis folder.
            molecules (np.array, optional): Array of molecules to be kept for 
                analysis. Defaults to None.
            ngh_size (int, optional): Maximum number of molecules per 
                neighborhood. Defaults to 100.
            ngh_sizes (list, optional): GraphSage sampling size.
                Defaults to [20, 10].
            minimum_nodes_connected (int, optional): Minimum molecules connected
                in the network. Defaults to 5.
            fraction_edges (int, optional): Number to divide the number of edges
                for training. Defaults to 10.
            train_p (float, optional): train_size is generally small if number 
                of molecules is large enough that information would be redundant. 
                Defaults to 0.25.
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
                and must contain ds.ca['Ncells'] and ds.ca['ClusterName'] 
                attributes. Defaults to None.
            exclude_clusters (list, optional): List of clusters present in 
                ds.ca['ClusterName'] to exclude. Defaults to [''].
            smooth (bool, optional): Smooth knn in the network data. Improves 
                both supervised and unsupervised network performance.
                Defaults to True.
            negative_samples (int, optional): Negative samples taken by 
                GraphSage sampler. Defaults to 5.
            distance_factor (float, optional): Distance selected to construct 
                network graph will be multiplied by the distance factor. 
                Defaults to 3.
            device (str, optional): Device where pytorch is run. Defaults to 'gpu'.
            lr (float, optional): learning rate .Defaults to 1e-3.
            aggregator (str, optional). Aggregator type for SAGEConv. Choose 
                between 'pool','mean','min','lstm'. Defaults to 'pool'.
            celltype_distribution (str, optional). Supervised cell_type/
                molecules distribution. Choose between 'uniform','ascending'
                or 'cell'. Defaults to 'uniform'.

        """
        super().__init__()

        self.model = model
        self.analysis_name = analysis_name
        self.ngh_sizes = ngh_sizes
        self.data = data
        self.molecules = molecules
        self.minimum_nodes_connected = minimum_nodes_connected
        self.fraction_edges = fraction_edges
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
        self.distance_factor = distance_factor
        self.subsample = subsample
        self.aggregator = aggregator
        self.celltype_distribution = celltype_distribution
        self.inference_type = inference_type
        self.n_epochs= n_epochs
        self.model_type = model_type

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.prepare_reference()

        ### Prepare data
        self.folder = self.save_to+self.analysis_name+ '_' +datetime.now().strftime("%Y-%m-%d-%H%M%S")
        os.mkdir(self.folder)
        if not os.path.isdir(self.save_to+'graph'):
            os.mkdir(self.save_to+'graph')

        #logging.info('Device is: ',self.device)
        #self.compute_distance_th(distance_factor,max_distance_nodes)
        self.subsample_xy()
        self.setup()
        
        dgluns = self.save_to+'graph/{}Unsupervised_smooth{}_dst{}_mNodes{}.graph'.format(self.molecules.shape[0],self.smooth,self.distance_factor,self.minimum_nodes_connected)
        if not os.path.isfile(dgluns):
            self.g = self.buildGraph()
            graph_labels = {"UnsupervisedDGL": th.tensor([0])}
            logging.info('Saving model...')
            dgl.data.utils.save_graphs(dgluns, [self.g], graph_labels)
            logging.info('Model saved.')
        else:
            logging.info('Loading model.')
            glist, _ = dgl.data.utils.load_graphs(dgluns) # glist will be [g1, g2]
            self.g = glist[0]
            logging.info('Graph data loaded.')
        
        if self.aggregator == 'attentional':
            # Remove all self loops because data will be saved, otherwise the 
            # number of self-loops will be doubled with each new run.
            remove = dgl.RemoveSelfLoop()
            self.g = remove(self.g)
            self.g = dgl.add_self_loop(self.g)

        logging.info(self.g)
        self.make_train_test_validation()
        l_loc,l_scale= self.compute_library_size()
        
        ### Prepare Model
        if type(self.model) == type(None):
            self.model = SAGELightning(in_feats=self.data.unique_genes.shape[0], 
                                        n_latent=48,
                                        n_layers=len(self.ngh_sizes),
                                        n_classes=2,
                                        n_hidden=64,
                                        lr=self.lr,
                                        supervised=self.supervised,
                                        reference=self.ref_celltypes,
                                        device=self.device.type,
                                        smooth=self.smooth,
                                        aggregator=self.aggregator,
                                        celltype_distribution=self.dist,
                                        ncells=self.ncells,
                                        inference_type=self.inference_type,
                                        l_loc=l_loc,
                                        l_scale= l_scale,
                                        scale_factor=1/(batch_size*self.data.unique_genes.shape[0]),
                                        warmup_factor=int(self.edges_train.shape[0]/self.batch_size)*self.n_epochs,
                                    )

        #self.g.ndata['gene'] = th.log(1+self.g.ndata['gene'])
        self.model.to(self.device)

        if 'clusters' in self.g.ndata.keys():
            self.clusters = self.g.ndata['clusters']

    def prepare_data(self):
        # do-something
        pass
    
    def save_graph(self):
        dgluns = self.save_to+'graph/{}Unsupervised_smooth{}_dst{}_mNodes{}.graph'.format(self.molecules.shape[0],self.smooth,self.distance_factor,self.minimum_nodes_connected)
        graph_labels = {"UnsupervisedDGL": th.tensor([0])}
        logging.info('Saving model...')
        dgl.data.utils.save_graphs(dgluns, [self.g], graph_labels)
        logging.info('Model saved.')


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

    def pyro_train(self, n_epochs=100):
        # Training loop.
        # We train for 80 epochs, although this isn't enough to achieve full convergence.
        # For optimal results it is necessary to tweak the optimization parameters.
        # For our purposes, however, 80 epochs of training is sufficient.
        # Training should take about 8 minutes on a GPU-equipped Colab instance.
        
        # Setup a variational objective for gradient-based learning.
        # Note we use TraceEnum_ELBO in order to leverage Pyro's machinery
        # for automatic enumeration of the discrete latent variable y.
        #self.guide = AutoGuideList(self.model)
        #self.guide.append(AutoNormal(poutine.block(self.model,expose_all=True, hide_all=False, hide=['test'],)
                   #,init_loc_fn=init_to_mean))
      
        self.guide = self.model.guide
        self.elbo = Trace_ELBO()
        svi = SVI(self.model.model, self.guide, AdamPyro({'lr':1e-3}), self.elbo)
        dl = self.train_dataloader()

        logging.info('Training')
        for epoch in range(n_epochs):
            losses = []

            # Take a gradient step for each mini-batch in the dataset
            for batch_idx, batch in enumerate(dl):
                loss = svi.step(batch)
                losses.append(loss)

            # Tell the scheduler we've done one epoch.
            #scheduler.step()
            logging.info("[Epoch %02d]  Loss: %.5f" % (epoch, np.mean(losses)))
        logging.info("Finished training!")

    def make_train_test_validation(self):
        """
        make_train_test_validation: only self.indices_validation is used.

        Splits the data into train, test and validation. Test data is not used for
        because at the moment the model performance cannot be checked against labelled
        data.
        """        
        indices_train = []
        indices_test = []
        import torch as th

        random_state = np.random.RandomState(seed=0)

        for gene in range(self.g.ndata['gene'].shape[1]):
            molecules_g = self.g.ndata['gene'][:,gene] == 1

            if molecules_g.sum() >= 20:
                indices_g = self.g.ndata['indices'][molecules_g]
                train_size = int(indices_g.shape[0]*self.train_p)
                test_size = indices_g.shape[0]-train_size
                permutation = random_state.permutation(indices_g)
                test_g = th.tensor(permutation[:test_size])
                train_g = th.tensor(permutation[test_size:(train_size+test_size)])   
                
            indices_train += train_g
            indices_test += test_g
        indices_train, indices_test = th.stack(indices_train), th.stack(indices_test)
        edges_bool_train =  th.isin(self.g.edges()[0],indices_train) & th.isin(self.g.edges()[1],indices_train) 
        edges_bool_test =  th.isin(self.g.edges()[0],indices_test) & th.isin(self.g.edges()[1],indices_test)

        self.edges_train  = np.random.choice(np.arange(edges_bool_train.shape[0])[edges_bool_train],int(edges_bool_train.sum()*(self.train_p/self.fraction_edges)),replace=False)
        self.edges_test  = np.random.choice(np.arange(edges_bool_test.shape[0])[edges_bool_test],int(edges_bool_test.sum()*(self.train_p/self.fraction_edges)),replace=False)

        logging.info('Training on {} edges.'.format(self.edges_train.shape[0]))
        logging.info('Testing on {} edges.'.format(self.edges_test.shape[0]))

    def train_dataloader(self):
        """
        train_dataloader

        Prepare dataloader

        Returns:
            dgl.dataloading.EdgeDataLoader: Deep Graph Library dataloader.
        """        
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(self.negative_samples)
        edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
            dgl.dataloading.NeighborSampler([int(_) for _ in self.ngh_sizes]),
            negative_sampler=negative_sampler,
            )

        unlab = dgl.dataloading.DataLoader(
                        self.g,
                        self.edges_train,
                        edge_sampler,
                        #negative_sampler=negative_sampler,
                        device=self.device,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=self.num_workers,
                        )
        return [unlab]

    def test_dataloader(self):
        """
        train_dataloader

        Prepare dataloader

        Returns:
            dgl.dataloading.EdgeDataLoader: Deep Graph Library dataloader.
        """        
        
        unlab = dgl.dataloading.EdgeDataLoader(
                        self.g,
                        self.edges_test,
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
        return unlab

    def validation_dataloader(self):
        """
        train_dataloader

        Prepare dataloader

        Returns:
            dgl.dataloading.EdgeDataLoader: Deep Graph Library dataloader.
        """        
        
        validation = dgl.dataloading.NodeDataLoader(
                        self.g,
                        th.arange(self.g.num_nodes(),),
                        dgl.dataloading.MultiLayerNeighborSampler([-1,-1]),
                        device=self.device,
                        batch_size=self.batch_size*1,
                        shuffle=False,
                        drop_last=False,
                        num_workers=self.num_workers,
                        )
        return validation

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
            self.model = self.model.load_from_checkpoint(model_path,
                                        in_feats=self.model.in_feats,
                                        n_latent=self.model.n_latent,
                                        n_classes=self.model.module.n_classes,
                                        n_layers=self.model.module.n_layers,
                                        )

        
        if continue_training or not is_trained:
            trainer.fit(self.model, train_dataloaders=self.train_dataloader())#,val_dataloaders=self.test_dataloader())
            

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
        molecules_id = self.g.ndata['indices']
        pd.DataFrame({
            'x':self.data.df.x.values.compute()[molecules_id.numpy()], 
            'y':self.data.df.y.values.compute()[molecules_id.numpy()],
            'g':self.data.df.g.values.compute()[molecules_id.numpy()],
            }
        ).to_parquet(self.folder+'/molecules_latent.parquet')
        
        np.save(self.folder+'/latent',self.latent_unlabelled)
        if self.supervised:
            self.prediction_unlabelled = prediction_unlabelled.softmax(dim=-1).detach().numpy()
            np.save(self.folder+'/labels',self.prediction_unlabelled)
            np.save(self.folder+'/probabilities',prediction_unlabelled)

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
                

class MultiGraphData(pl.LightningDataModule):
    """
    Class to prepare multiple Graphs for GraphSAGE

    """    
    def __init__(self,
        filepaths:list,
        num_nodes_per_graph:int=500,
        ngh_sizes:list=[20,10],
        train_percentage:float=0.75,
        batch_size:int=1024,
        num_workers:int=1,
        analysis_name:str='MultiGraph',
        n_epochs:int=3,
        save_to:str ='',
        lr = 1e-3,
        ):
        
        self.filepaths = filepaths
        self.ngh_sizes = ngh_sizes
        self.train_percentage = train_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.analysis_name = analysis_name
        self.save_to = save_to
        self.folder = self.save_to+self.analysis_name+ '_' +datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.n_epochs = n_epochs
        self.num_nodes_per_graph = num_nodes_per_graph
        self.lr = lr
        os.mkdir(self.folder)
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        batchgraph_filename = os.path.join(self.save_to, '{}BatchGraph.graph'.format(len(self.filepaths)))

        if not os.path.isfile(batchgraph_filename):
            self.load_graphs()

            graph_labels = {"Multigraph": th.arange(len(self.filepaths))}
            logging.info('Saving model...')
            dgl.data.utils.save_graphs(batchgraph_filename, self.sub_graphs, graph_labels)
        else:
            logging.info('Loading model...')
            self.sub_graphs, graph_labels = dgl.data.utils.load_graphs(batchgraph_filename)

        self.training_dataloaders = []
        for sg in self.sub_graphs:
            logging.info('Number of genes in graph: {}'.format(sg.ndata['gene'].shape[1]))
            #self.training_dataloaders.append(self.wrap_train_dataloader(sg))
        self.sub_graphs = dgl.batch(self.sub_graphs)
        self.training_dataloaders.append(self.wrap_train_dataloader_batch())

        self.model = SAGELightning(in_feats=self.sub_graphs.ndata['gene'].shape[1], 
                                        n_latent=48,
                                        n_layers=len(self.ngh_sizes),
                                        n_classes=2,
                                        n_hidden=64,
                                        lr=self.lr,
                                    )
        self.model.to(self.device)
        self.setup()

    def load_graphs(self):
        """
        load_graphs

        Load graphs from filepaths

        Returns:
            list: List of graphs
        """        
        self.sub_graphs = []
        self.training_dataloaders = []
        #self.validation_dataloaders = []

        for filepath in self.filepaths:
            logging.info('Subsampling graph from {}.'.format(filepath))
            #subsample 1M edges from the graph for training:
            g = dgl.load_graphs(filepath)[0][0]
            to_delete = [x for x in g.ndata.keys() if x != 'gene']
            for x in to_delete:
                del g.ndata[x]
            random_train_nodes = th.randperm(g.nodes().size(0))[:self.num_nodes_per_graph]
            sg = dgl.khop_in_subgraph(g, g.nodes()[random_train_nodes], k=2)[0]
            logging.info('Training sample with {} nodes and {} edges.'.format(sg.num_nodes(), sg.num_edges()))
            
            self.sub_graphs.append(sg)
            #self.validation_dataloaders.append(self.validation_dataloader(sg))
        #self.batch_graph = dgl.batch(graphs)
    
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
        return self.training_dataloaders

    #def validation_dataloader(self):
    #    return self.validation_dataloaders

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
            self.model = self.model.load_from_checkpoint(model_path,
                                        in_feats=self.sub_graphs.ndata['gene'].shape[1],
                                        n_latent=48,
                                        n_layers=2,
                                        n_classes=2,
                                        n_hidden=64,
                                        )

        
        if continue_training or not is_trained:
            trainer.fit(self.model, train_dataloaders=self.train_dataloader())#,val_dataloaders=self.test_dataloader())
            
    def wrap_train_dataloader(self, batch_graph):
        """
        train_dataloader

        Prepare dataloader

        Returns:
            dgl.dataloading.EdgeDataLoader: Deep Graph Library dataloader.
        """        
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(3)
        edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
            dgl.dataloading.NeighborSampler([int(_) for _ in self.ngh_sizes]),
            negative_sampler=negative_sampler,
            )

        #edges = batch_graph.edges()
        train_p_edges = int(batch_graph.num_edges()*(self.train_percentage/5))
        train_edges = th.randperm(batch_graph.num_edges())[:train_p_edges]
        #train_edges = self.make_train_test_validation(batch_graph)

        unlab = dgl.dataloading.DataLoader(
                        batch_graph,
                        train_edges,
                        edge_sampler,
                        #negative_sampler=negative_sampler,
                        device=self.device,
                        use_uva=True,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=self.num_workers,
                        )
        return unlab

    def wrap_train_dataloader_batch(self):
        """
        train_dataloader

        Prepare dataloader

        Returns:
            dgl.dataloading.EdgeDataLoader: Deep Graph Library dataloader.
        """        
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(3)
        edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
            dgl.dataloading.NeighborSampler([int(_) for _ in self.ngh_sizes]),
            negative_sampler=negative_sampler,
            )

        #edges = batch_graph.edges()
        #train_p_edges = int(self.sub_graphs.num_edges()*(self.train_percentage/10))
        #train_edges = th.randperm(self.sub_graphs.num_edges())[:train_p_edges]
        train_edges = self.make_train_test_validation(self.sub_graphs)

        unlab = dgl.dataloading.DataLoader(
                        self.sub_graphs,
                        train_edges,
                        edge_sampler,
                        #negative_sampler=negative_sampler,
                        device=self.device,
                        use_uva=True,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=self.num_workers,
                        )
        return unlab
    
    def validation_dataloader(self, graph):
        """
        train_dataloader

        Prepare dataloader

        Returns:
            dgl.dataloading.EdgeDataLoader: Deep Graph Library dataloader.
        """        
        
        validation = dgl.dataloading.NodeDataLoader(
                        graph,
                        th.arange(graph.num_nodes(),),
                        dgl.dataloading.MultiLayerNeighborSampler([-1,-1]),
                        device=self.device,
                        batch_size=self.batch_size*1,
                        shuffle=False,
                        drop_last=False,
                        num_workers=self.num_workers,
                        )
        return validation

    def make_train_test_validation(self, g):
        """
        make_train_test_validation: only self.indices_validation is used.

        Splits the data into train, test and validation. Test data is not used for
        because at the moment the model performance cannot be checked against labelled
        data.
        """        
        indices_train = []
        indices_test = []
        random_state = np.random.RandomState(seed=0)
        nodes = th.arange(g.num_nodes())

        for gene in range(g.ndata['gene'].shape[1]):
            molecules_g = g.ndata['gene'][:,gene] == 1

            #if molecules_g.sum() >= 20:
            indices_g = nodes[molecules_g]
            train_size = int(indices_g.shape[0]*self.train_percentage)
            if train_size == 0:
                train_size = 1
            test_size = indices_g.shape[0]-train_size
            permutation = random_state.permutation(indices_g)
            test_g = th.tensor(permutation[:test_size])
            train_g = th.tensor(permutation[test_size:(train_size+test_size)])   
                
            indices_train += train_g
            indices_test += test_g
        indices_train, indices_test = th.stack(indices_train), th.stack(indices_test)
        edges_bool_train =  th.isin(g.edges()[0],indices_train) & th.isin(g.edges()[1],indices_train) 
        edges_train = np.random.choice(np.arange(edges_bool_train.shape[0])[edges_bool_train],int(edges_bool_train.sum()*(self.train_percentage/10)),replace=False)
        #self.edges_test  = np.random.choice(np.arange(edges_bool_test.shape[0])[edges_bool_test],int(edges_bool_test.sum()*(self.train_p/self.fraction_edges)),replace=False)
        logging.info('Training sample on {} edges.'.format(edges_train.shape[0]))
        #logging.info('Testing on {} edges.'.format(self.edges_test.shape[0]))
        return edges_train

    def get_latents(self):
        """
        get_latents: get the new embedding for each molecule
        
        Passes the validation data through the model to generatehe neighborhood 
        embedding. If the model is in supervised version, the model will also
        output the predicted cell type.

        Args:
            labelled (bool, optional): [description]. Defaults to True.
        """        
        import scanpy as sc
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from joblib import dump, load

        self.model.eval()
        latent_unlabelled = []

        #self.sub_graphs = dgl.unbatch(self.sub_graphs)
        #for g in tqdm(self.sub_graphs):
        lu, _ = self.model.module.inference(
                                self.sub_graphs,
                                self.model.device,
                                10*512,
                                0)

        print(lu.shape)
        latent_unlabelled = lu.detach().cpu().numpy()
        self.latent_unlabelled = np.concatenate(latent_unlabelled)
        logging.info('Latent embeddings generated for {} molecules'.format(self.latent_unlabelled.shape[0]))
        
        np.save(self.folder+'/latent',self.latent_unlabelled)

        random_sample_train = np.random.choice(
                                len(self.latent_unlabelled), 
                                np.min([len(self.latent_unlabelled),500000]), 
                                replace=False)

        training_latents =self.latent_unlabelled[random_sample_train,:]
        adata = sc.AnnData(X=training_latents)
        logging.info('Building neighbor graph for clustering...')
        sc.pp.neighbors(adata, n_neighbors=15)
        logging.info('Running Leiden clustering...')
        sc.tl.leiden(adata, random_state=42, resolution=1.8)
        logging.info('Leiden clustering done.')
        clusters= adata.obs['leiden'].values
        logging.info('Total of {} found'.format(len(np.unique(clusters))))
        clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3))
        clf.fit(training_latents, clusters)
        clusters = clf.predict(self.latent_unlabelled).astype('int8')

        clf_total = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3))
        clf_total.fit(self.latent_unlabelled, clusters)
        clusters = clf.predict(self.latent_unlabelled).astype('int8')
        dump(clf, 'miniMultiGraphNeighborhoodClassifier.joblib') 
        self.sub_graphs.ndata['label'] = th.tensor(clusters)


class MultiGraphDataPredictor(pl.LightningDataModule):
    """
    Class to prepare multiple Graphs for GraphSAGE

    """    
    def __init__(self,
        filepaths:list,
        num_nodes_per_graph:int=500,
        ngh_sizes:list=[20,10],
        train_percentage:float=0.75,
        batch_size:int=1024,
        num_workers:int=1,
        analysis_name:str='MultiGraph',
        n_epochs:int=3,
        save_to:str ='',
        lr = 1e-3,
        ):
        
        self.filepaths = filepaths
        self.ngh_sizes = ngh_sizes
        self.train_percentage = train_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.analysis_name = analysis_name
        self.save_to = save_to
        self.folder = self.save_to+self.analysis_name+ '_' +datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.n_epochs = n_epochs
        self.num_nodes_per_graph = num_nodes_per_graph
        self.lr = lr
        os.mkdir(self.folder)
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        batchgraph_filename = os.path.join(self.save_to, '{}BatchGraphLabelled.graph'.format(len(self.filepaths)))
        self.sub_graphs, graph_labels = dgl.data.utils.load_graphs(batchgraph_filename)

        self.training_dataloaders = []
        for sg in self.sub_graphs:
            logging.info('Number of genes in graph: {}'.format(sg.ndata['gene'].shape[1]))
            #self.training_dataloaders.append(self.wrap_train_dataloader(sg))
        self.sub_graphs = dgl.batch(self.sub_graphs)
        self.training_dataloaders.append(self.wrap_train_dataloader_batch())

        self.model = SAGELightning(in_feats=self.sub_graphs.ndata['gene'].shape[1], 
                                        n_latent=len(self.sub_graphs.ndata['label'].unique()),
                                        n_layers=len(self.ngh_sizes),
                                        n_classes=len(self.sub_graphs.ndata['label'].unique()),
                                        n_hidden=64,
                                        lr=self.lr,
                                    )
        self.model.to(self.device)
        self.setup()
    
    def setup(self, stage: Optional[str] = None):
        #self.d = th.tensor(self.molecules_df(),dtype=th.float32) #works
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler([int(_) for _ in self.ngh_sizes])

        self.checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=self.save_to,
            filename=self.analysis_name+'-{epoch:02d}-{train_loss:.2f}-labelprediction',
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
        return self.training_dataloaders

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
            if File.count('labelprediction.ckpt'):
                is_trained = 1
                model_path = os.path.join(self.save_to, File)
                break
        
        if type(model_file) != type(None):
            is_trained = 1
            model_path = model_file

        if is_trained:
            logging.info('Pretrained model exists in folder {}, loading model parameters. If you wish to re-train, delete .ckpt file.'.format(model_path))
            self.model = self.model.load_from_checkpoint(model_path,
                                        in_feats=self.sub_graphs.ndata['gene'].shape[1],
                                        n_latent=48,
                                        n_layers=2,
                                        n_classes=2,
                                        n_hidden=64,
                                        )

        
        if continue_training or not is_trained:
            trainer.fit(self.model, train_dataloaders=self.train_dataloader())#,val_dataloaders=self.test_dataloader())
            
    def wrap_train_dataloader_batch(self):
        """
        train_dataloader

        Prepare dataloader

        Returns:
            dgl.dataloading.EdgeDataLoader: Deep Graph Library dataloader.
        """        
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(3)
        edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
            dgl.dataloading.NeighborSampler([int(_) for _ in self.ngh_sizes]),
            negative_sampler=negative_sampler,
            )

        #edges = batch_graph.edges()
        #train_p_edges = int(self.sub_graphs.num_edges()*(self.train_percentage/10))
        #train_edges = th.randperm(self.sub_graphs.num_edges())[:train_p_edges]
        train_edges = self.make_train_test_validation(self.sub_graphs)

        unlab = dgl.dataloading.DataLoader(
                        self.sub_graphs,
                        train_edges,
                        edge_sampler,
                        #negative_sampler=negative_sampler,
                        device=self.device,
                        use_uva=True,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=self.num_workers,
                        )
        return unlab

    def make_train_test_validation(self, g):
        """
        make_train_test_validation: only self.indices_validation is used.

        Splits the data into train, test and validation. Test data is not used for
        because at the moment the model performance cannot be checked against labelled
        data.
        """        
        indices_train = []
        indices_test = []
        random_state = np.random.RandomState(seed=0)
        nodes = th.arange(g.num_nodes())

        for gene in range(g.ndata['gene'].shape[1]):
            molecules_g = g.ndata['gene'][:,gene] == 1

            #if molecules_g.sum() >= 20:
            indices_g = nodes[molecules_g]
            train_size = int(indices_g.shape[0]*self.train_percentage)
            if train_size == 0:
                train_size = 1
            test_size = indices_g.shape[0]-train_size
            permutation = random_state.permutation(indices_g)
            test_g = th.tensor(permutation[:test_size])
            train_g = th.tensor(permutation[test_size:(train_size+test_size)])   
                
            indices_train += train_g
            indices_test += test_g
        indices_train, indices_test = th.stack(indices_train), th.stack(indices_test)
        edges_bool_train =  th.isin(g.edges()[0],indices_train) & th.isin(g.edges()[1],indices_train) 
        edges_train = np.random.choice(np.arange(edges_bool_train.shape[0])[edges_bool_train],int(edges_bool_train.sum()*(self.train_percentage/10)),replace=False)
        #self.edges_test  = np.random.choice(np.arange(edges_bool_test.shape[0])[edges_bool_test],int(edges_bool_test.sum()*(self.train_p/self.fraction_edges)),replace=False)
        logging.info('Training sample on {} edges.'.format(edges_train.shape[0]))
        #logging.info('Testing on {} edges.'.format(self.edges_test.shape[0]))
        return edges_train

    def get_latents(self):
        """
        get_latents: get the new embedding for each molecule
        
        Passes the validation data through the model to generatehe neighborhood 
        embedding. If the model is in supervised version, the model will also
        output the predicted cell type.

        Args:
            labelled (bool, optional): [description]. Defaults to True.
        """        
        import scanpy as sc
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from joblib import dump, load

        self.model.eval()
        latent_unlabelled = []

        #self.sub_graphs = dgl.unbatch(self.sub_graphs)
        #for g in tqdm(self.sub_graphs):
        lu, _ = self.model.module.inference(
                                self.sub_graphs,
                                self.model.device,
                                10*512,
                                0)
