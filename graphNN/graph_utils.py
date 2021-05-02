import networkx as nx
import loompy
from scipy.spatial import cKDTree as KDTree
import torch
import numpy as np
import torch_geometric
import torch
from tqdm import tqdm
from torch_geometric.data import NeighborSampler
from torch_geometric.data import Data
from annoy import AnnoyIndex
import random
from tqdm import trange
import pickle
import os
import pytorch_lightning as pl
from typing import Optional, List, NamedTuple
from torch import Tensor
from torch_sparse import SparseTensor
from torch_cluster import random_walk
import timeit
import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
from torch import Tensor
from torch_sparse import SparseTensor




def compute_library_size(data):
    sum_counts = data.sum(axis=1)
    masked_log_sum = np.ma.log(sum_counts)
    
    log_counts = masked_log_sum.filled(0)
    
    local_mean = np.mean(log_counts).astype(np.float32)
    local_var = np.var(log_counts).astype(np.float32)

    return local_mean, local_var

class GraphData(pl.LightningDataModule):
    def __init__(self,
        data, # Data as numpy array of shape (Genes, Cells)
        genes, # List of Genes
        coords, # Array of shape (Cells, 2), coordinates for the cells
        cells=None, # Array with cell_ids of shape (Cells)
        with_background=False,
        distance_threshold = 250,
        minimum_nodes_connected = 3,
        ngh_sizes = [5, 10],
        train_p = 0.75,
        batch_size= 128,
        num_workers=1
        ):

        super().__init__()
        
        self.ngh_sizes = ngh_sizes
        self.data = data
        self.genes = genes
        self.coords = coords
        self.cells = cells
        self.distance_threshold = distance_threshold
        self.minimum_nodes_connected = minimum_nodes_connected
        self.train_p = train_p
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.local_mean,self.local_var = compute_library_size(self.data.T)

        if type(self.cells) == type(None):
            self.cells = np.arange(self.data.shape[1])

        self.G = self.buildGraph(self.distance_threshold)

        self.cleanGraph()
        self.compute_size()
        self.setup()

    def buildGraph(self, d_th):
        print('Building graph...')
        G = nx.Graph()


        if not os.path.isfile('Edges-{}Nodes.pkl'.format(self.data.shape[1])):
            t = AnnoyIndex(2, 'euclidean')  # Length of item vector that will be indexed
            for i in trange(self.coords.shape[0]):
                v = self.coords[i,:]
                t.add_item(i, v)

            print('Building tree...')
            t.build(10) # 10 trees
            print('Built tree.')
            t.save('test.ann')

            u = AnnoyIndex(2, 'euclidean')
            u.load('test.ann') # super fast, will just mmap the file
            print('Find neighbors below distance: {}'.format(d_th))

            def find_pairs(k,nghs,tree,distance):
                pair = [(k,n) for n in nghs if tree.get_distance(k,n) < distance]
                return pair

            def find_nn_distance(coords,tree,distance):
                res = []
                for i in trange(coords.shape[0]):
                    # 100 sets the number of neighbors to find for each node
                    #  it is set to 100 since we usually will compute neighbors
                    #  [20,10]
                    nghs = t.get_nns_by_item(i, self.ngh_sizes[0])
                    pair = find_pairs(i,nghs,u,distance)
                    res += pair
                return res
            res = find_nn_distance(self.coords,u,d_th)

            with open('Edges-{}Nodes.pkl'.format(self.data.shape[1]), 'wb') as f:
                pickle.dump(res, f)
        
        else:
            print('Edges file exists, loading...')
            with open('Edges-{}Nodes.pkl'.format(self.data.shape[1]), "rb") as input_file:
                res = pickle.load(input_file)

        # Add nodes to graph
        G.add_nodes_from((self.cells), test=False, val=False, label=0)
        # Add node features to graph
        nx.set_node_attributes(G,dict(zip((self.cells), self.data)), 'expression')
        # Add edges to graph
        G.add_edges_from(res)
        return G

    
    def prepare_data(self):
        # do-something
        pass

    def cleanGraph(self):
        print('Cleaning graph...')
        for component in tqdm(list(nx.connected_components(self.G))):
            if len(component)< self.minimum_nodes_connected:
                for node in component:
                    self.G.remove_node(node)
    
    def compute_size(self):
        self.train_size = int(self.cells.max()*self.train_p)
        self.test_size = self.cells.max()-int(self.cells.max()*self.train_p)
        
        random_state = np.random.RandomState(seed=0)
        permutation = random_state.permutation(self.cells.max())
        
        self.indices_test = torch.tensor(permutation[:self.test_size])
        #self.indices_test = np.array([x in indices_test for x in self.cells])
        self.indices_train = torch.tensor(permutation[self.test_size : (self.test_size + self.train_size)])
        #self.indices_train = np.array([x in indices_train for x in self.cells])    
        self.indices_validation = torch.tensor(np.arange(self.cells.max()))
        #self.indices_validation = np.array([x in indices_validation for x in self.cells])


    def setup(self, stage: Optional[str] = None):
        print('Loading dataset...')
        self.edges_tensor = torch.tensor(np.array(list(self.G.edges)).T)
        #self.dataset = Data(torch.tensor(self.data.T,dtype=torch.float32),edge_index=self.edges_tensor)
        self.dataset = torch.tensor(self.data.T,dtype=torch.float32)


    def train_dataloader(self):
        return NeighborSampler2(self.edges_tensor, node_idx=self.indices_train,data=self.dataset,
                               sizes=self.ngh_sizes, return_e_id=False,
                               batch_size=self.batch_size,
                               shuffle=True, num_workers=self.num_workers)

    def validation_dataloader(self):
        return NeighborSampler2(self.edges_tensor, node_idx=self.indices_validation,data=self.dataset,
                               sizes=self.ngh_sizes, return_e_id=False,
                               batch_size=self.batch_size,
                               shuffle=False)


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
                 transform: Callable = None, **kwargs):

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
        out = (self.data[n_id],self.data[pos],self.data[neg],adjs)
        out = self.transform(*out) if self.transform is not None else out
        return out

    def sample_pos_neg(self, batch):
        batch = torch.tensor(batch)
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



    


        
