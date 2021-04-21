import pandas as pd
import numpy as np
from typing import Generator, Tuple

class Iteration:

    def xy_groupby_gene_generator(self) -> Generator[Tuple[str, np.ndarray, np.ndarray], None, None]:
        """Generator function that groups XY coordinates by gene.

        Uses the Pandas groupby() function for speed on unsorted numpy arrays.

        Yields:
            Iterator[Union[str, np.ndarray, np.ndarray]]: Gene name, X 
                coordinates, Y coordinates.

        """
        df = pd.DataFrame(data = np.column_stack((self.x, self.y, self.gene)), columns = [self.x_label, self.y_label, self.gene_label])
        for (g,c) in df.groupby(self.gene_label):
            yield g, c.loc[:, self.x_label].to_numpy(), c.loc[:, self.y_label].to_numpy()



    def make_gene_coordinates(self) -> None:
<<<<<<< HEAD
        """Make a dictionary with point coordinates for each gene.

        Output will be in self.gene_coordinates
        """
=======

>>>>>>> 7572b93950ca1686ebd5b8d347fc772c8f956a58
        self.gene_coordinates = {g: np.column_stack((x, y)) for g, x, y in self.xy_groupby_gene_generator()}
