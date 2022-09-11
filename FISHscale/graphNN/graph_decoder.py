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

    def simulate_expression(self, ntimes=10):
        self._multinomial_hexbin()
        simulation = []
        for _ in trange(ntimes):
            self._lose_identity()
            self.random_sampler()
            simulated_expression= self.random_decoder()
            simulation.append(simulated_expression)

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
        self.g.ndata['simulation'] = simulation
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
        self.g.ndata['gene'] = th.tensor(self.g.ndata['gene'],dtype=th.float32)
        self.g.ndata['tmp_gene'] = nodes_gene.clone().float()
        self.g.ndata['tmp_gene'][self.lost_nodes,:] = th.zeros_like(self.g.ndata['tmp_gene'][self.lost_nodes,:],dtype=th.float32)

        print((self.g.ndata['tmp_gene'].sum(axis=1) > 0).sum())

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2,prefetch_node_feats=['tmp_gene'])
        self.decoder_dataloader = dgl.dataloading.DataLoader(
                self.g.to('cpu'), self.lost_nodes.to('cpu'), sampler,
                batch_size=1024, shuffle=True, drop_last=False, num_workers=self.num_workers,
                #persistent_workers=(self.num_workers > 0)
                )
                
    def random_decoder(self):

        for _, nodes, blocks in tqdm(self.decoder_dataloader):
            block_1hop = blocks[1]
            block_2hop = blocks[0]
            
            probmultinomial_region = th.stack([self.multinomial_region[int(n)] for n in self.g.ndata['hex_region'][nodes] ])
            
            block_1hop.srcdata['tmp_gene']=  block_1hop.srcdata['tmp_gene'].float()
            block_1hop.update_all(fn.copy_u('tmp_gene', 'e'), fn.sum('e', 'h'))
            genes_1hop = block_1hop.dstdata['h']
            logprobs1_hop = np.array([np.array([np.log(self.attentionNN1_scores[:][gene]+1e-6)*counts.numpy() for gene,counts in zip(self.data.unique_genes,center)]).sum(axis=0) for center in genes_1hop])
            
            
            block_2hop = blocks[0]
            block_2hop.srcdata['tmp_gene']=  block_2hop.srcdata['tmp_gene'].float()
            repeated_nodes = np.where(np.isin(block_2hop.srcnodes(),block_1hop.srcnodes()))[0]
            block_2hop.srcdata['tmp_gene'][repeated_nodes,:] = th.zeros_like(block_2hop.srcdata['tmp_gene'][repeated_nodes,:]).float()
            
            block_2hop.update_all(fn.copy_u('tmp_gene', 'e'), fn.sum('e', 'h'))
            
            block_1hop.srcdata['tmp_gene2hop'] = block_2hop.dstdata['h'].float()
            block_1hop.update_all(fn.copy_u('tmp_gene2hop', 'e2'), fn.sum('e2', 'h2'))
            genes_2hop = block_1hop.dstdata['h2']
            logprobs2_hop = np.array([np.array([np.log(self.attentionNN2_scores[:][gene]+1e-6)*counts.numpy() for gene,counts in zip(self.data.unique_genes,center)]).sum(axis=0) for center in genes_2hop])
            
            probabilities = th.log(probmultinomial_region+1e-6) + logprobs1_hop + logprobs2_hop
            M = dist.Multinomial(total_count=1, logits=probabilities).sample()
            
            self.g.ndata['tmp_gene'][nodes,:] = M.float()
            #print((self.g.ndata['tmp_gene'].sum(axis=1) > 0).sum())
            
        simulated_genes = self.data.unique_genes[np.where(self.g.ndata['tmp_gene'].numpy() == 1)][0]
        del self.g.ndata['tmp_gene']
        return simulated_genes

    def random_decoder_deprecated(self):
        resampled_nodes = []
        resampled_genes = []

        for nghs, nodes, _ in tqdm(self.decoder_dataloader):
            for n in range(nodes.shape[0]):
                center_node = nghs[n]
                
                multinomial_region = self.multinomial_region[int(self.g.ndata['hex_region'][nodes[n]])]
                multinomial_region_probs = multinomial_region/(multinomial_region.sum()+1e-6)

                nodes_1hop = th.unique(th.cat(self.g.in_subgraph(center_node).edges()))
                genes_ngh1 = self.data.unique_genes[np.where(self.g.ndata['gene'][nodes_1hop,:] == 1)[1]]

                if genes_ngh1.shape[0] > 0:
                    probs1 = np.stack([self.attentionNN1_scores[:][g].values for g in genes_ngh1])
                    probs1 = np.sum(np.log(probs1+1e-6),axis=0)#probs1/ (probs1.sum() + 1e-6)
                else:
                    probs1 = 1

                nodes_2hop = th.unique(th.cat(self.g.in_subgraph(nodes_1hop).edges()))
                genes_ngh2 = self.data.unique_genes[np.where(self.g.ndata['gene'][nodes_2hop,:] == 1)[1]]
                if genes_ngh2.shape[0] > 0:
                    probs2 = np.stack([self.attentionNN2_scores[:][g].values for g in genes_ngh2])
                    probs2 = np.sum(np.log(probs2+1e-6),axis=0)#probs1/ (probs1.sum() + 1e-6)
                else:
                    probs2 = 1
                    
                probabilities_genes = np.log(multinomial_region_probs+1e-6)+probs1+probs2

                M = dist.Multinomial(total_count=1, logits=probabilities_genes/probabilities_genes.sum()).sample()
                sampled_gene = self.data.unique_genes[np.where(M == 1)][0]
                resampled_nodes.append(nghs[n].numpy().tolist())
                resampled_genes.append(sampled_gene)
                
            self.lost_nodes = self.lost_nodes[th.isin(self.lost_nodes, nghs, invert=True)]

        simulated_expression = self.data.unique_genes[(np.where(self.g.ndata['gene'])[1])]
        simulated_expression[resampled_nodes] = resampled_genes
        return simulated_expression
