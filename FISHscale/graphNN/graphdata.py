import torch as th
import numpy as np
import torch
from tqdm import trange
import os
import pytorch_lightning as pl
from typing import Optional
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

import dgl
from FISHscale.graphNN.models import SAGELightning
from FISHscale.graphNN.graph_utils import GraphUtils, GraphPlotting

from pyro.distributions.util import broadcast_shape
from torch.optim import Adam
from pyro.optim import Adam as AdamPyro
from pyro.infer.autoguide import init_to_mean
from pyro.infer import SVI, config_enumerate, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoGuideList, AutoNormal, AutoDelta

class GraphData(pl.LightningDataModule, GraphUtils, GraphPlotting):
    """
    Class to prepare the data for GraphSAGE

    """    
    def __init__(self,
        data, # Data as FISHscale Dataset (Molecules, Genes)
        model=None, # GraphSAGE model
        analysis_name:str='',
        molecules=None, # Array with molecules_ids of shape (molecules)
        ngh_size = 500,
        ngh_sizes = [20, 10],
        minimum_nodes_connected = 5,
        train_p = 0.25,
        batch_size= 512,
        num_workers=0,
        save_to = '',
        subsample=1,
        ref_celltypes=None,
        exclude_clusters:dict={},
        smooth:bool=False,
        negative_samples:int=1,
        distance_factor:int=4,
        max_distance_nodes=None,
        device='cpu',
        lr=1e-3,
        aggregator='pool',
        celltype_distribution='uniform',
        inference_type='deterministic',
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
            device (str, optional): Device where pytorch is run. Defaults to 'cpu'.
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prepare_reference()

        ### Prepare data
        self.folder = self.save_to+self.analysis_name+ '_' +datetime.now().strftime("%Y-%m-%d-%H%M%S")
        os.mkdir(self.folder)
        if not os.path.isdir(self.save_to+'graph'):
            os.mkdir(self.save_to+'graph')

        #print('Device is: ',self.device)
        #self.compute_distance_th(distance_factor,max_distance_nodes)
        self.subsample_xy()
        self.setup()
        
        dgluns = self.save_to+'graph/{}Unsupervised_smooth{}_dst{}_mNodes{}.graph'.format(self.molecules.shape[0],self.smooth,self.distance_factor,self.minimum_nodes_connected)
        if not os.path.isfile(dgluns):
            self.g = self.buildGraph()
            graph_labels = {"UnsupervisedDGL": th.tensor([0])}
            print('Saving model...')
            dgl.data.utils.save_graphs(dgluns, [self.g], graph_labels)
            print('Model saved.')
        else:
            print('Loading model.')
            glist, _ = dgl.data.utils.load_graphs(dgluns) # glist will be [g1, g2]
            self.g = glist[0]
            print('Model loaded.')
        
        if self.aggregator == 'attentional':
            self.g = dgl.add_self_loop(self.g)

        print(self.g)
        self.make_train_test_validation()
        l_loc,l_scale= self.compute_library_size()

                ### Prepare Model
        if type(self.model) == type(None):
            self.model = SAGELightning(in_feats=self.data.unique_genes.shape[0], 
                                        n_latent=24,
                                        n_layers=len(self.ngh_sizes),
                                        n_classes=self.ref_celltypes.shape[1],
                                        n_hidden=48,
                                        lr=self.lr,
                                        supervised=self.supervised,
                                        reference=self.ref_celltypes,
                                        device=self.device.type,
                                        smooth=self.smooth,
                                        aggregator=self.aggregator,
                                        celltype_distribution=self.dist,
                                        ncells=self.ncells,
                                        inference_type=self.inference_type,
                                        l_loc=l_loc[0][0],
                                        l_scale= l_scale[0][0],
                                        scale_factor=1/(batch_size*self.data.unique_genes.shape[0]),
                                    )
        self.model.to(self.device)

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

        print('Training')
        for epoch in range(n_epochs):
            losses = []

            # Take a gradient step for each mini-batch in the dataset
            for batch_idx, batch in enumerate(dl):
                loss = svi.step(batch)
                losses.append(loss)

            # Tell the scheduler we've done one epoch.
            #scheduler.step()
            print("[Epoch %02d]  Loss: %.5f" % (epoch, np.mean(losses)))
        print("Finished training!")

    def train(self,max_epochs=15,gpus=0):
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
                            max_epochs=max_epochs)
        trainer.fit(self.model, train_dataloaders=self.train_dataloader())

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
        self.indices_train, self.indices_test = th.stack(indices_train), th.stack(indices_test)
        edges_train =  th.isin(self.g.edges()[0],indices_train) & th.isin(self.g.edges()[1],indices_train) 
        edges_test =  th.isin(self.g.edges()[0],indices_test) & th.isin(self.g.edges()[1],indices_test)
        
        print('Training on {} edges.'.format(self.random_edges.shape[0]))

    def train_dataloader(self):
        """
        train_dataloader

        Prepare dataloader

        Returns:
            dgl.dataloading.EdgeDataLoader: Deep Graph Library dataloader.
        """        
        unlab = dgl.dataloading.EdgeDataLoader(
                        self.g,
                        self.random_edges,
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

    def validation_dataloader_pyro(self):
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


    def train(self,max_epochs=5,gpus=0):
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
                            max_epochs=max_epochs)
        trainer.fit(self.model, train_dataloaders=self.train_dataloader())

    #### plotting and latent factors #####

    def get_latents_pyro(self,labelled=True):
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
                        self.g.ndata['gene'], self.g.ndata['ngh'],
                        self.device,
                        10*512,
                        0)#.detach().numpy()

        self.prediction_unlabelled = prediction_unlabelled.softmax(dim=-1).detach().numpy()