from typing import Any
import numpy as np
from sklearn.decomposition import PCA, LatentDirichletAllocation

class Decomposition:

    def PCA(self, data: Any) -> np.ndarray:
            """Calculate principle components

            Args:
                df_hex (pd.DataFrame): Dataframe with samples as rows and 
                    features as columns. 

            Returns:
                [np.array]: Array with principle components as rows.
            """
            pca = PCA()
            return pca.fit_transform(data)
        
    def LDA(self, data: Any, n_components:int = 64, n_jobs:int = -1) -> np.ndarray:
        """Calculate Latent Dirichlet Allocation.

        Args:
            df_hex (pd.DataFrame): Dataframe with samples as rows and 
                features as columns. 
            n_components (int, optional): Number of resulting components.
                Defaults to 64.
            n_jobs (int, optional): Number of jobs. Defaults to -1.

        Returns:
            np.ndarray: Array with components as rows.
        """
        lda = LatentDirichletAllocation(n_components=n_components, random_state=0, n_jobs=n_jobs)
        return lda.fit_transform(data.T)