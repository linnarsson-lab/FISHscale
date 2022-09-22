from setuptools import setup, find_packages

__version__ = "0.13.0"

setup(
	name="FISHscale",
	python_requires ='>=3.7, <3.10', #open3d does not work in python 3.10
	version=__version__,
	packages=find_packages(),
	install_requires=[
		'loompy',
		'numpy',
		'scikit-learn',
		'scipy',
		'networkx',
		'sklearn',
        'dask',
		'tqdm',
		'umap-learn',  # imported as "umap"
		'torch',
		'torchvision',
        'pytorch-lightning',
		'dgl',
		'open3d',
		'pandas',
        'pint',
        'pyarrow',
        'fastparquet',
        'PyQt5',
        'annoy',
        'geopandas',
        'shapely',
        'numba',
		'h5py',
		'ripleyk',
		'scikit-image',
		'torch-scatter',
		'dask[distributed]',
		'pyro-ppl',
		'scvi-tools',
	],

	author="Linnarsson Lab",
	authors_email=["lars.borm@ki.se","alejandro.mossi.albiach@ki.se"],
	description="Pipeline for large smFISH data",
	license="MIT",
	url="https://github.com/linnarsson-lab/FISHscale",
)
