import numpy as np
import pandas as pd


def correlate_neighbours(hex_bin, coordinates, spacing, neighbours=1):

    def worker(d1, d2):
        corr_neighbours = np.array([spearmanr(d1, d2[j])[0] for j in d2])
        
        
        mean = corr_neighbours.mean()
        std = corr_neighbours.std()
        variance = corr_neighbours.var()
        if corr_neighbours.shape[0] > 2:
            difference = corr_neighbours.max() - corr_neighbours.min()
        else:
            difference = 0

        return np.array([mean, std, variance, difference])
    
    #Make nearest neighbour graph
    radius = neighbours * spacing 
    radius += 0.1 * radius
    Kgraph = radius_neighbors_graph(cent, radius, include_self=False, n_jobs=-1)
    Kgraph_neigh = Kgraph.toarray().astype('bool')

    #Format expression data 
    d1_data = dask.delayed([i for i in hex_bin.to_numpy().T])
    #d2_data = dask.delayed([hex_bin.iloc[:, Kgraph.getrow(i).indices].to_numpy() for i in range(Kgraph.shape[0]) ])
    #d2_data = [hb.loc[:, n].to_numpy() for n in Kgraph_neigh.T]
    
    #neighbour data
    
    #Compute
    results = []
    #for i, (d1, d2) in enumerate(zip(d1_data, d2_data)):
    for i in range(hex_bin.shape[1]):
        
        #d1 = hex_bin.iloc[:,i].to_numpy()
        #d2 = hex_bin.iloc[:, Kgraph.getrow(i).indices].to_numpy()
        y = dask.delayed(worker)(hex_bin.iloc[:,i].to_numpy(), hex_bin.iloc[:, Kgraph.getrow(i).indices].to_numpy())
        results.append(y)

    with ProgressBar():
        results = dask.compute(*results, scheduler='processes')
    
    #Return dataframe with data
    return pd.DataFrame(data = np.vstack(results), columns=['r_mean', 'r_std', 'r_variance', 'r_difference'])


#Need to build a new function, that takes the sprce matrix and for every combination calcualtes the correlation.
#Then it uses this sparce matrix with correlation values to calcualted the mean,std etc of the neighbours.

def:
    #Make nearest neighbour graph
    radius = neighbours * spacing 
    radius += 0.1 * radius
    Kgraph = radius_neighbors_graph(cent, radius, include_self=False, n_jobs=-1)


    #Calculate correlation for each pair


    #Use Kgraph to pool the correlations and calculate metrics.






#corr = correlate_neighbours(hb, cent, 100, neighbours=2)
#d.hexbin_plot(c=corr['r_difference'])