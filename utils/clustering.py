import scanpy as sc

class Clustering:

    def clustering_scanpy(self, factors, n_neighbors=15):
        adata = sc.AnnData(X=factors)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.leiden(adata, random_state=42)
        