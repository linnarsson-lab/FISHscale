# FISHscale
Spatial analysis of FISH data

Various functions for spatial analysis of point based data. Developed for image based in situ transcriptomics, where the data are sets of points.  
Many functions will generalize to other point based data.  

# Density along line
`density_1D.py`  
Function to determine the density of points that fall within a certain width along a given line.
<img src="/Images/1D_distribution_example.png" width="800px"/>
  
# Coordinate based colocalization
`coordinate_based_colocalization.py`  
Function to calculate the coordinate based colocalization. Implementation based on this paper:  
Sebastian Malkusch, Ulrike Endesfelder, Justine Mondry, Márton Gelléri, Peter J. Verveer & Mike Heilemann  
Malkusch, S., Endesfelder, U., Mondry, J. et al. Coordinate-based colocalization analysis of single-molecule   
localization microscopy data. Histochem Cell Biol 137, 1–10 (2012). https://doi.org/10.1007/s00418-011-0880-5  
  
[paper](https://link.springer.com/article/10.1007/s00418-011-0880-5)
  
### High spatial correlation
<img src="/Images/CBC_example_high.png" width="800px"/>
  
### Low spatial correlation
<img src="/Images/CBC_example_low.png" width="800px"/>
