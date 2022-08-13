# 1 HEXBIN data to get a multinomial distribution of genes per HEX. Phg(H, G).
    # This could be replaced by a general Pg(G) multinomial, to make an even 
    # more generalized model
# 2 Lose identity of a certain percentage of all molecules Pl(L). 
# 3 Modify normalized attention probabilities to introduce knowckout (or overexpression).
    # Do N times until full expansion
    # 3.1 Expand expression spatially following the multinomial probabilities by the 
    #    restricted neighbor attention.
# 4 Check the new HEXBIN distributions
from scipy.spatial import KDTree
import torch as th
from pyro import distributions as dist

class GraphDecoder:
    def __init__(
        self,
        lose_identity_percentage = 0.5
        ):
        self.lose_identity_percentage = lose_identity_percentage

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
        return 'lose'

    def random_sampler(self):
        return 'a'

    def random_decoder(self):
        return 'b'

        