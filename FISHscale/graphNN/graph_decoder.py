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

class GraphDecoder:
    def __init__(
        self,
        lose_identity_percentage = 0.5,
        ):
        self.lose_identity_percentage = lose_identity_percentage

       
    def load_attention(
        self, 
        attentionNN1_file:str,
        attentionNN2_file:str,
        ):
        self.attentionNN1_scores = pd.read_parquet(attentionNN1_file)
        self.attentionNN2_scores = pd.read_parquet(attentionNN2_file)

    def _multinomial_hexbin(self,spacing=500,
        min_count=10,
        ) -> None:

        df_hex,centroids = self.data.hexbin_make(spacing=spacing, min_count=min_count)
        tree = KDTree(centroids)
        dist, hex_region = tree.query(self.g.ndata['coords'].numpy(), distance_upper_bound=spacing, workers=-1)
        self.g.ndata['hex_region'] = th.tensor(hex_region)

        self.multinomial_region = []
        for h in np.unique(hex_region):
            freq = self.g.ndata['gene'][hex_region == h].sum(axis=0)
            m = dist.Multinomial(total_count=1,probs=freq/freq.sum()).sample()
            self.multinomial_region.append(m)

    def _lose_identity(self):
        self.lost_nodes = th.tensor(np.random.choice(np.arange(self.g.num_nodes()) ,size=int(0.2*self.g.num_nodes())))

    def random_sampler(self):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.decoder_dataloader = dgl.dataloading.DataLoader(
                self.g, th.tensor(self.lost_nodes).to(self.g.device), sampler,
                batch_size=512, shuffle=True, drop_last=False, num_workers=1,
                #persistent_workers=(num_workers > 0)
                )

    def random_decoder(self):
        for nghs, nodes, blocks in self.decoder_dataloader:
            ngh2 = blocks[0]
            ngh1 = blocks[1]
            for n in range(nodes.shape[0]):

                nodes_ngh1 = ngh1.edges()[0][ngh1.edges()[1] == n]
                genes_ngh1 = self.data.unique_genes[np.where(self.g.ndata['gene'][nghs[nodes_ngh1].numpy(),:] == 1)[1]]
                # Generate Probabilities for ngh1
                
                nodes_ngh2 = ngh2.edges()[0][th.isin(ngh2.edges()[1],nodes_ngh1)]
                genes_ngh2 = self.data.unique_genes[np.where(self.g.ndata['gene'][nghs[nodes_ngh2].numpy(),:] == 1)[1]]
                # Generate Probabilities for ngh1

            break

        