import networkx as nx
from networkx.algorithms.traversal import edgedfs
from numpy.core.fromnumeric import size
import torch
import numpy as np
import torch
from tqdm import tqdm
from annoy import AnnoyIndex
from tqdm import trange
import pickle
import os
import pytorch_lightning as pl
from typing import Optional, List, NamedTuple
from torch import Tensor
from torch_sparse import SparseTensor
from torch_cluster import random_walk
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
from torch import Tensor
from torch_sparse import SparseTensor
from scipy import sparse
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import h5py


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
        minimum_nodes_connected = 5,
        ngh_sizes = [20, 10],
        train_p = 0.25,
        batch_size= 1024,
        num_workers=1,
        save_to = '',
        subsample=1,
        ref_celltypes=None,
        Ncells = None,
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

        self.folder = self.analysis_name+ '_' +datetime.now().strftime("%Y-%m-%d-%H%M%S")
        os.mkdir(self.folder)

        self.subsample = subsample
        self.subsample_xy()

        '''        
        if type(ref_celltypes) != type(None):
            self.ref_celltypes = torch.tensor(ref_celltypes,dtype=torch.float32)
        else:
            self.ref_celltypes = None
        '''
        
        if type(ref_celltypes) != type(None):
            if type(Ncells) == type(None):
                Ncells = np.ones(ref_celltypes.shape[1])
            self.cluster_nghs, self.cluster_edges, self.cluster_labels = self.cell_types_to_graph(ref_celltypes, Ncells)
        
        # Save random cell selection
        self.buildGraph(self.distance_threshold)
        self.compute_size()
        self.setup()

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
            patience=3,
            verbose=True,
            mode='min',
            )
    
    def prepare_data(self):
        # do-something
        pass

    def setup(self, stage: Optional[str] = None):
        print('Loading dataset...')
        #self.d = torch.tensor(self.molecules_df(),dtype=torch.float32) #works
        self.d = self.molecules_df()
        
        print('tensor',self.d.shape)

    def compute_size(self):
        self.train_size = int((self.cells.shape[0])*self.train_p)
        self.test_size = self.cells.shape[0]-int(self.cells.shape[0]*self.train_p)  
        random_state = np.random.RandomState(seed=0)
        permutation = random_state.permutation(self.cells.shape[0])
        self.indices_test = torch.tensor(permutation[:self.test_size])
        self.indices_train = torch.tensor(permutation[self.test_size : (self.test_size + self.train_size)])
        self.indices_validation = torch.tensor(np.arange(self.cells.shape[0]))

    def train_dataloader(self):
        return NeighborSampler2(self.edges_tensor, node_idx=self.indices_train,data=self.d,
                               sizes=self.ngh_sizes, return_e_id=False,
                               batch_size=self.batch_size,
                               shuffle=True, num_workers=self.num_workers)

    def validation_dataloader(self):
        # set a big batch size, not all will be loaded in memory but it will loop relatively fast through large dataset
        return NeighborSampler2(self.edges_tensor, node_idx=self.indices_validation,data=self.d,
                               sizes=self.ngh_sizes, return_e_id=False,
                               batch_size=self.batch_size*1,
                               shuffle=False)

    def train(self,max_epochs=5,gpus=-1):     
        print('Saving random cells used during training...')
        np.save(self.folder +'/random_cells.npy',self.cells)
        trainer = pl.Trainer(gpus=gpus,callbacks=[self.checkpoint_callback,self.early_stop_callback],max_epochs=max_epochs)
        trainer.fit(self.model, train_dataloader=self.train_dataloader(),val_dataloaders=self.validation_dataloader())

    def get_latent(self, deterministic=True,run_clustering=True,make_plot=True):
        print('Training done, generating embedding...')
        import matplotlib.pyplot as plt
        embedding = []
        for x,pos,neg,adjs,ref in self.validation_dataloader():
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
            n_neighbors=15,
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
            plt.savefig("{}/umap_embedding.png".format(GD.folder), bbox_inches='tight', dpi=500)

    def molecules_df(self):
        rows,cols = [],[]
        #filt = self.data.df.map_partitions(lambda x: x[x.index.isin(self.cells)]).g.values.compute()
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
        if type(self.subsample) == int and self.subsample < 1:
            self.cells = np.random.randint(0,self.data.shape[0], int(self.subsample*self.data.shape[0]))
        elif type(self.subsample) == dict:
            filt_x =  ((self.data.df.x > self.subsample['x'][0]) & (self.data.df.x < self.subsample['x'][1])).values.compute()
            filt_y =  ((self.data.df.y > self.subsample['y'][0]) & (self.data.df.x < self.subsample['y'][1])).values.compute()
            self.cells = self.cells[filt_x & filt_y]
            #self.cells = np.random.choice(self.data.df.index.compute(),size=int(subsample*self.data.shape[0]),replace=False)

    def buildGraph(self, d_th,coords=None):
        print('Building graph...')
        if type(coords)  == type(None):
            edge_file = os.path.join(self.save_to,'Edges-{}Nodes-Ngh{}-{}-dst{}'.format(self.cells.shape[0],self.ngh_sizes[0],self.ngh_sizes[1],self.distance_threshold))
            tree_file = os.path.join(self.save_to,'Tree-{}Nodes-Ngh{}-{}-dst{}.ann'.format(self.cells.shape[0],self.ngh_sizes[0],self.ngh_sizes[1],self.distance_threshold))
        
            coords = np.array([self.data.df.x.values.compute()[self.cells], self.data.df.y.values.compute()[self.cells]]).T
            neighborhood_size = self.ngh_sizes[0]
        else:
            edge_file = os.path.join(self.save_to,'Supervised-Edges-{}Nodes-Ngh{}-{}-dst{}'.format(self.cells.shape[0],self.ngh_sizes[0],self.ngh_sizes[1],self.distance_threshold))
            tree_file = os.path.join(self.save_to,'Supervised-Tree-{}Nodes-Ngh{}-{}-dst{}.ann'.format(self.cells.shape[0],self.ngh_sizes[0],self.ngh_sizes[1],self.distance_threshold))
        
            coords = coords
            neighborhood_size = self.ngh_sizes[0]

        if not os.path.isfile(edge_file):
            t = AnnoyIndex(2, 'euclidean')  # Length of item vector that will be indexed
            for i in trange(coords.shape[0]):
                v = coords[i,:]
                t.add_item(i, v)

            print('Building tree...')
            t.build(10) # 10 trees
            print('Built tree.')
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
            res = find_nn_distance(coords,t,d_th)

            with h5py.File(edge_file, 'w') as hf:
                hf.create_dataset("edges",  data=res)

        else:
            print('Edges file exists, loading...')
            with h5py.File(edge_file, 'r+') as hf:
                res = hf['edges'][:]
        print('Edges',res.shape)
        self.edges_tensor = torch.tensor(res.T)

    def cell_types_to_graph(self, data, Ncells):
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

        print('Converting clusters into simulated molecule neighborhoods...')
        for i in trange(data.shape[1]):
            molecules = []
            # Reduce number of cells by Ncells.min() to avoid having a huge dataframe, since it is actually simulated data
            cl_i = data[:,i]*(Ncells[i]/Ncells.min())
            for x in range(cl_i.shape[0]):
                dot = np.zeros_like(cl_i)
                dot[i] = 1
                try:
                    dot = np.stack([dot]*int(cl_i[x]))
                    molecules.append(dot)

                except:
                    pass
            molecules = np.concatenate(molecules)
            all_molecules.append(molecules)
            
            all_coords.append(np.random.normal(loc=i*100,scale=25,size=[molecules.shape[0],2]))
            #all_coords.append(np.ones_like(molecules)*50*i)
            all_cl.append(np.ones(molecules.shape[0])*i)

        all_molecules = np.concatenate(all_molecules)
        all_coords = np.concatenate(all_coords)
        all_cl = np.concatenate(all_cl)
        edges = self.buildGraph(2.5,coords=all_coords)
        return all_molecules, edges, all_cl

'''
def compute_library_size(data):   
    sum_counts = data.sum(axis=1)
    masked_log_sum = np.ma.log(sum_counts)
    log_counts = masked_log_sum.filled(0)
    local_mean = np.mean(log_counts).astype(np.float32)
    local_var = np.var(log_counts).astype(np.float32)
    return local_mean, local_var
'''

class NeighborSampler2(torch.utils.data.DataLoader):
    r"""The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.

    Given a GNN with :math:`L` layers and a specific mini-batch of nodes
    :obj:`node_idx` for which we want to compute embeddings, this module
    iteratively samples neighbors and constructs bipartite graphs that simulate
    the actual computation flow of GNNs.

    More specifically, :obj:`sizes` denotes how much neighbors we want to
    sample for each node in each layer.
    This module then takes in these :obj:`sizes` and iteratively samples
    :obj:`sizes[l]` for each node involved in layer :obj:`l`.
    In the next layer, sampling is repeated for the union of nodes that were
    already encountered.
    The actual computation graphs are then returned in reverse-mode, meaning
    that we pass messages from a larger set of nodes to a smaller one, until we
    reach the nodes for which we originally wanted to compute embeddings.

    Hence, an item returned by :class:`NeighborSampler` holds the current
    :obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the
    computation, and a list of bipartite graph objects via the tuple
    :obj:`(edge_index, e_id, size)`, where :obj:`edge_index` represents the
    bipartite edges between source and target nodes, :obj:`e_id` denotes the
    IDs of original edges in the full graph, and :obj:`size` holds the shape
    of the bipartite graph.
    For each bipartite graph, target nodes are also included at the beginning
    of the list of source nodes so that one can easily apply skip-connections
    or add self-loops.

    .. note::

        For an example of using :obj:`NeighborSampler`, see
        `examples/reddit.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        reddit.py>`_ or
        `examples/ogbn_products_sage.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        ogbn_products_sage.py>`_.

    Args:
        edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
            :obj:`torch_sparse.SparseTensor` that defines the underlying graph
            connectivity/message passing flow.
            :obj:`edge_index` holds the indices of a (sparse) symmetric
            adjacency matrix.
            If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
            must be defined as :obj:`[2, num_edges]`, where messages from nodes
            :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
            (in case :obj:`flow="source_to_target"`).
            If :obj:`edge_index` is of type :obj:`torch_sparse.SparseTensor`,
            its sparse indices :obj:`(row, col)` should relate to
            :obj:`row = edge_index[1]` and :obj:`col = edge_index[0]`.
            The major difference between both formats is that we need to input
            the *transposed* sparse adjacency matrix.
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to :obj:`sizes[l] = -1`, all neighbors are included
            in layer :obj:`l`.
        node_idx (LongTensor, optional): The nodes that should be considered
            for creating mini-batches. If set to :obj:`None`, all nodes will be
            considered.
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)
        return_e_id (bool, optional): If set to :obj:`False`, will not return
            original edge indices of sampled edges. This is only useful in case
            when operating on graphs without edge features to save memory.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, edge_index: Union[Tensor, SparseTensor], data,
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, supervised_data:Optional[Tensor]=None,**kwargs):

        edge_index = edge_index.to('cpu')

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for Pytorch Lightning...
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None
        self.data = data
        self.supervised_data = supervised_data

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            value = torch.arange(edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super(NeighborSampler2, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        n_id,pos,neg = self.sample_pos_neg(n_id)

        #out = (self.data[n_id],self.data[pos],self.data[neg], adjs) #v1 no sparse tensor

        #out v2 using sparse tensor
        out = (torch.tensor(self.data[n_id].toarray(),dtype=torch.float32),
                torch.tensor(self.data[pos].toarray(),dtype=torch.float32),
                torch.tensor(self.data[neg].toarray(),dtype=torch.float32),
                adjs,
                self.supervised_data)


        out = self.transform(*out) if self.transform is not None else out
        return out

    def sample_pos_neg(self, batch):
        #batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                dtype=torch.long)

        return batch,pos_batch,neg_batch

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)

class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)



    


        
