
import networkx as nx
import loompy
from scipy.spatial import cKDTree as KDTree
import torch
import numpy as np
import torch_geometric
import torch
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import NeighborSampler


def compute_library_size(data):
    sum_counts = data.sum(axis=1)
    masked_log_sum = np.ma.log(sum_counts)
    
    log_counts = masked_log_sum.filled(0)
    
    local_mean = np.mean(log_counts).astype(np.float32)
    local_var = np.var(log_counts).astype(np.float32)

    return local_mean, local_var

class GraphData:
    def __init__(self,
        data, # Data as numpy array of shape (Genes, Cells)
        genes, # List of Genes
        coords, # Array of shape (Cells, 2), coordinates for the cells
        cells=None, # Array with cell_ids of shape (Cells)
        distance_threshold = 250,
        minimum_nodes_connected = 3,
        ngh_sizes = [5, 10],
        train_p = 0.75,
        batch_size= 128,
        ):

        self.ngh_sizes = ngh_sizes
        self.data = data
        self.genes = genes
        self.coords = coords
        self.cells = cells
        self.distance_threshold = distance_threshold
        self.minimum_nodes_connected = minimum_nodes_connected
        self.train_p = train_p
        self.batch_size = batch_size

        self.local_mean,self.local_var = compute_library_size(self.data.T)

        if type(self.cells) == type(None):
            self.cells = np.arange(self.data.shape[1])

        self.G = self.buildGraph(self.distance_threshold)
        self.cleanGraph()
        self.compute_size()
        self.load_dataset()
        self.load_trainers()


    def buildGraph(self, d_th):
        G = nx.Graph()

        kdT = KDTree(self.coords)
        res = kdT.query_pairs(d_th)
        res = [(x[0],x[1]) for x in list(res)]

        # Add nodes to graph
        G.add_nodes_from((self.cells), test=False, val=False, label=0)
        # Add node features to graph
        nx.set_node_attributes(G,dict(zip((self.cells), self.data)), 'expression')
        # Add edges to graph
        G.add_edges_from(res)
        return G

    def cleanGraph(self):
        for component in tqdm(list(nx.connected_components(self.G))):
            if len(component)< self.minimum_nodes_connected:
                for node in component:
                    self.G.remove_node(node)

    
    def compute_size(self):
        self.train_size = int(self.cells.shape[0]*self.train_p)
        self.test_size = self.cells.shape[0]-int(self.cells.shape[0]*self.train_p)
        
        random_state = np.random.RandomState(seed=0)
        permutation = random_state.permutation(self.cells.shape[0])
        
        self.indices_test = torch.tensor(permutation[:self.test_size])
        #self.indices_test = np.array([x in indices_test for x in self.cells])
        self.indices_train = torch.tensor(permutation[self.test_size : (self.test_size + self.train_size)])
        #self.indices_train = np.array([x in indices_train for x in self.cells])    
        self.indices_validation = torch.tensor(np.arange(self.cells.shape[0]))
        #self.indices_validation = np.array([x in indices_validation for x in self.cells])


    def load_dataset(self):
        self.edges_tensor = torch.tensor(np.array(list(self.G.edges)).T)
        self.dataset = torch_geometric.data.Data(torch.tensor(self.data.T,dtype=torch.float32),edge_index=self.edges_tensor)

    def load_trainers(self):
        data = self.dataset
        self.train_loader = NeighborSampler(data.edge_index, sizes=self.ngh_sizes, batch_size=self.batch_size, shuffle=True,node_idx=self.indices_train,drop_last=True)
        self.test_loader = NeighborSampler(data.edge_index, sizes=self.ngh_sizes, batch_size=self.batch_size, shuffle=True,node_idx=self.indices_test,drop_last=True)
        self.validation_loader = NeighborSampler(data.edge_index, sizes=self.ngh_sizes, batch_size=self.batch_size ,shuffle=False,node_idx=self.indices_validation)

        


        