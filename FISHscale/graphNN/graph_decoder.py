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
from tqdm import trange

class GraphDecoder:
    def __init__(
        self,
        lose_identity_percentage = 0.25,
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

    def simulate_expression(self, ntimes=100):
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
        self.lost_nodes = th.tensor(np.random.choice(np.arange(self.g.num_nodes()) ,size=int(0.2*self.g.num_nodes())))

    def random_sampler(self):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.decoder_dataloader = dgl.dataloading.DataLoader(
                self.g, th.tensor(self.lost_nodes).to(self.g.device), sampler,
                batch_size=512, shuffle=True, drop_last=False, num_workers=0,
                #persistent_workers=(num_workers > 0)
                )

    def random_decoder(self):
        resampled_nodes = []
        resampled_genes = []

        for nghs, nodes, blocks in self.decoder_dataloader:
            ngh2 = blocks[0]
            ngh1 = blocks[1]
            for n in range(nodes.shape[0]):

                multinomial_region = self.multinomial_region[int(self.g.ndata['hex_region'][nodes[n]])]

                multinomial_region_probs = multinomial_region/(multinomial_region.sum()+1e-6)
                #center_gene = self.data.unique_genes[np.where(self.g.ndata['gene'][nodes[n].numpy(),:] == 1)][0]      
                nodes_ngh1 = ngh1.edges()[0][ngh1.edges()[1] == n]
                nodes_ngh1F = nghs[nodes_ngh1][th.isin(nghs[nodes_ngh1], self.lost_nodes,invert=True)]
                nodes_ngh1F = th.unique(nodes_ngh1F)
                genes_ngh1 = self.data.unique_genes[np.where(self.g.ndata['gene'][nodes_ngh1F.numpy(),:] == 1)[1]]
                #probs1 = np.array([ self.attentionNN1_scores[center_gene][g]*(genes_ngh1 == g).sum() for g in self.data.unique_genes])
                if genes_ngh1.shape[0] > 0:
                    probs1 = np.stack([self.attentionNN1_scores[:][g].values for g in genes_ngh1])
                    probs1 = np.sum(np.log(probs1+1e-6),axis=0)#probs1/ (probs1.sum() + 1e-6)

                else:
                    probs1 = th.ones(self.data.unique_genes.shape[0])/self.data.unique_genes.shape[0]
                
                
                nodes_ngh2 = ngh2.edges()[0][th.isin(ngh2.edges()[1],nodes_ngh1)]
                nodes_ngh2F = nghs[nodes_ngh2][th.isin(nghs[nodes_ngh2], self.lost_nodes,invert=True)]
                nodes_ngh2F = th.unique(nodes_ngh2F)
                genes_ngh2 = self.data.unique_genes[np.where(self.g.ndata['gene'][nodes_ngh2F.numpy(),:] == 1)[1]]
                if genes_ngh2.shape[0] > 0:
                    probs2 = np.stack([self.attentionNN2_scores[:][g].values for g in genes_ngh2])
                    probs2 = np.sum(np.log(probs2+1e-6),axis=0)#probs1/ (probs1.sum() + 1e-6)

                else:
                    probs2 = th.ones(self.data.unique_genes.shape[0])/self.data.unique_genes.shape[0] 

              
                if probs1.sum() == 0:
                    probs1 += 1e-6
                if probs2.sum() == 0:
                    probs2 += 1e-6


                probabilities_genes = np.log(multinomial_region_probs+1e-6)+probs1+probs2
                #print(probabilities_genes)
                
                M = dist.Multinomial(total_count=1, logits=probabilities_genes/probabilities_genes.sum()).sample()
                sampled_gene = self.data.unique_genes[np.where(M.numpy() == 1)][0]
                resampled_nodes.append(nghs[n].numpy().tolist())
                resampled_genes.append(sampled_gene)
                
            self.lost_nodes = self.lost_nodes[th.isin(self.lost_nodes, nghs, invert=True)]

        simulated_expression = self.data.unique_genes[(np.where(self.g.ndata['gene'])[1])]
        simulated_expression[resampled_nodes] = resampled_genes
        return simulated_expression
                

