from setuptools import setup, find_packages

__version__ = "0.12.0"

setup(
	name="FISHscale",
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
		'dgl',
		'open3d',
		'pandas',
	],

	author="Linnarsson Lab",
	authors_email=["lars.borm@ki.se","alejandro.mossi.albiach@ki.se"],
	description="Pipeline for large smFISH data",
	license="MIT",
	url="https://github.com/linnarsson-lab/FISHscale",
)