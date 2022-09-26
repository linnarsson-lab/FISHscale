# 1 HEXBIN data to get a multinomial distribution of genes per HEX. Phg(H, G).
    # This could be replaced by a general Pg(G) multinomial, to make an even 
    # more generalized model
# 2 Lose identity of a certain percentage of all molecules Pl(L). 
# 3 Modify normalized attention probabilities to introduce knowckout (or overexpression).
    # Do N times until full expansion
    # 3.1 Expand expression spatially following the multinomial probabilities by the 
    #    restricted neighbor attention.
# 4 Check the new HEXBIN distributions
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import torch as th
from pyro import distributions as dist
import dgl
import dgl.function as fn
from tqdm import trange, tqdm
import logging

class GraphDecoder:
    def __init__(
        self,
        lose_identity_percentage = 0.9,
        ):
        self.lose_identity_percentage = lose_identity_percentage

       
    def load_attention(
        self, 
        attentionNN1_file:str,
        attentionNN2_file:str,
        ):
        self.attentionNN1_scores = pd.read_parquet(attentionNN1_file)
        self.attentionNN1_scores = self.attentionNN1_scores/self.attentionNN1_scores.sum(axis=0)
        self.attentionNN2_scores = pd.read_parquet(attentionNN2_file)
        self.attentionNN2_scores = self.attentionNN2_scores/self.attentionNN2_scores.sum(axis=0)

    def _multinomial_hexbin(self,spacing=500,
        min_count=0,
        ) -> None:

        df_hex,centroids = self.data.hexbin_make(spacing=spacing, min_count=min_count)
        tree = KDTree(centroids)
        dst, hex_region = tree.query(self.g.ndata['coords'].numpy(), distance_upper_bound=spacing, workers=-1)
        self.g.ndata['hex_region'] = th.tensor(hex_region)

        self.multinomial_region = {}
        for h in np.unique(hex_region):
            freq = self.g.ndata['gene'][hex_region == h].sum(axis=0)
            freq = freq/freq.sum()
            if freq.sum() == 0 or np.isnan(freq.sum()):
                freq = np.ones_like(freq)/freq.shape[0]
            self.multinomial_region[h]= freq

    def simulate_expression(self, ntimes=10, simulation_name='base_simulation'):
        self._multinomial_hexbin()
        simulation = []

        simulation_zeros = np.zeros((self.g.num_nodes(),self.g.ndata['gene'].shape[1]))
        for _ in trange(ntimes):
            self._lose_identity()
            self.random_sampler()
            simulated_expression= self.random_decoder()
            simulation.append(simulated_expression)
            simulation_zeros += self.g.ndata['tmp_gene'].numpy()
            del self.g.ndata['tmp_gene']

        simulation =  np.stack(simulation).T
        expression_by_region_by_simulation = []

        for reg in th.unique(self.g.ndata['hex_region']):
            region_expression = simulation[self.g.ndata['hex_region'] == reg,:]
            expression_simulation = []
            for s in range(region_expression.shape[1]):
                
                sim_exp = [(region_expression[:,s] == g ).sum() for g in self.data.unique_genes]
                expression_simulation.append(sim_exp)
            expression_by_region_by_simulation.append(expression_simulation)
            
        expression_by_simulation_by_region = np.array(expression_by_region_by_simulation)
        self.g.ndata[simulation_name] = th.tensor(simulation_zeros)
        return expression_by_simulation_by_region

    def plot_distribution(self):
        'a'

    def _lose_identity(self):
        self.lost_nodes = th.tensor(
            np.random.choice(np.arange(self.g.num_nodes()) ,
            size=int(self.lose_identity_percentage*self.g.num_nodes()),
            replace=False),
            )

    def random_sampler(self):
        nodes_gene =  self.g.ndata['gene']
        #self.g.ndata['gene'] = th.tensor(self.g.ndata['gene'],dtype=th.float32)
        self.g.ndata['tmp_gene'] = th.tensor(nodes_gene.clone(),dtype=th.uint8)
        self.g.ndata['tmp_gene'][self.lost_nodes,:] = th.zeros_like(self.g.ndata['tmp_gene'][self.lost_nodes,:],dtype=th.uint8)

        logging.info((self.g.ndata['tmp_gene'].sum(axis=1) > 0).sum())

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2,prefetch_node_feats=['tmp_gene'])
        self.decoder_dataloader = dgl.dataloading.DataLoader(
                self.g.to('cpu'), self.lost_nodes.to('cpu'), sampler,
                batch_size=1024, shuffle=True, drop_last=False, num_workers=self.num_workers,
                persistent_workers=(self.num_workers > 0)
                )
                
    def random_decoder(self):
        for _, nodes, blocks in self.decoder_dataloader:
            block_1hop = blocks[1]
            block_2hop = blocks[0]
            
            probmultinomial_region = th.stack([self.multinomial_region[int(n)] for n in self.g.ndata['hex_region'][nodes] ])
            block_1hop.srcdata['tmp_gene']=  block_1hop.srcdata['tmp_gene'].float()
            block_1hop.update_all(fn.copy_u('tmp_gene', 'e'), fn.sum('e', 'h'))
            genes_1hop = block_1hop.dstdata['h']

            logprobs1_hop = np.log((genes_1hop@self.attentionNN1_scores).values+1e-6)

            block_2hop.srcdata['tmp_gene']=  block_2hop.srcdata['tmp_gene'].float()
            repeated_nodes = np.where(np.isin(block_2hop.srcnodes(),block_1hop.srcnodes()))[0]
            block_2hop.srcdata['tmp_gene'][repeated_nodes,:] = th.zeros_like(block_2hop.srcdata['tmp_gene'][repeated_nodes,:]).float()
            block_2hop.update_all(fn.copy_u('tmp_gene', 'e'), fn.sum('e', 'h'))
            
            block_1hop.srcdata['tmp_gene2hop'] = block_2hop.dstdata['h'].float()
            block_1hop.update_all(fn.copy_u('tmp_gene2hop', 'e2'), fn.sum('e2', 'h2'))
            genes_2hop = block_1hop.dstdata['h2']
            logprobs2_hop = np.log((genes_2hop@self.attentionNN2_scores).values+1e-6)

            probabilities = th.log(probmultinomial_region+1e-6) + logprobs1_hop + logprobs2_hop
            M = dist.Multinomial(total_count=1, logits=probabilities).sample()
            
            self.g.ndata['tmp_gene'][nodes,:] = th.tensor(M,dtype=th.uint8)
            #logging.info((self.g.ndata['tmp_gene'].sum(axis=1) > 0).sum())
            
        simulated_genes = self.data.unique_genes[np.where(self.g.ndata['tmp_gene'].numpy() == 1)[1]]
        return simulated_genes
