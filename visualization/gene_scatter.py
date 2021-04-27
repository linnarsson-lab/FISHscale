import matplotlib.pyplot as plt
import numpy as np
from pint import UnitRegistry
from typing import Union, Any, List
from time import strftime


class AxSize:

    def __init__(self):
        self.ureg = UnitRegistry()

    def to_inch(self, x:Union[float, str], unit: str='micrometer') -> float:
        """Convert scale to inches.

        Takes an length in micrometer or length with specified unit and 
        converts to inches using Pint UnitRegistry.

        Args:
            x (Union[float, str]): Length in micrometer or length with 
                specified unit. Like: "15 milimeter".
            unit (str, optional): Unit of x. See Pint documentation for valid
                units. Defaults to 'micrometer'.

        Returns:
            [float]: Converted input in inches. (As number, not as number with
                unit.)
        """

        #Add unit if not present
        if isinstance(x, float) or isinstance(x, int):
            x = f'{x} {unit}'

        #Convert to inch
        x = self.ureg(x)
        inch = x.to(self.ureg.inch)

        return inch.magnitude

    def set_size(self, w:float, h:float, ax: Any=None):
        """Set size of axis, will drive figure size.

        From: https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units

        Args:
            w (float): Width of axes in inch.
            h (float): Hight of axes in inch.
            ax (Any, optional): [description]. Ax to set size. If None will 
                get current axes. Defaults to None.
        """

        if not ax: ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)

    def add_scale_bar(self, ax):

        scale_defaults = np.array([0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000, 
                                   5000, 10000, 50000, 100000, 500000, 1000000])
        scale_names = ['1 nm', '10 nm', '100 nm', '1 µm', '10 µm', '50 µm', '100 µm', '500 µm', 
                       '1 mm', '5 mm', '1 cm', '5 cm', '10 cm', '50 cm', '1 m']
        
        #Get extend of x axis
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_extend = x_max - x_min
        y_extend = y_max - y_min

        #Convert to micrometer
        x_extend_um = (x_extend * self.unit_scale).to(self.ureg.micrometer).magnitude

        #Find closest scale bar
        indx = np.argmin(np.abs(scale_defaults - (x_extend_um * 0.1)))
        if x_extend_um > (scale_defaults[-1] * 5):
            print(f'Scale bar warning: Scale bars longer than {scale_names[-1]} are not supported yet.')
            indx = len(scale_defaults)
        if indx == -1:
            indx = 0        
        bar_size = scale_defaults[indx]
        bar_name = scale_names[indx]

        #Define position
        right = x_max - (0.05 * x_extend) 
        left = right - bar_size
        text_center = right - (0.5 * (right - left))
        hight_bar = y_min + (0.05 * y_extend)
        hight_text = y_min + (0.07 * y_extend)

        #Define linewidth and fontsize
        ax_hight = ax.bbox.transformed(ax.transAxes).height
        lw = ax_hight / 40000
        fs = ax_hight / 8000

        #Set scale bar
        ax.text(text_center, hight_text, bar_name, color='white', ha='center', fontsize=fs,
                bbox=dict(facecolor='black', edgecolor=None), zorder=5)
        ax.hlines(hight_bar, left, right, colors=['white'], linewidth=lw, zorder=10)


class GeneScatter(AxSize):

    def scatter_plot(self, genes: Union[List, np.ndarray], s: float=0.1, ax_scale_factor: int=10, 
                    view: Union[Any, List] = None, scalebar: bool=True, show_axes: bool=False, 
                    save: bool=False, save_name: str='', dpi: int=300, file_format: str='.eps') -> None:
        """Make a scatter plot of the data.

        Uses a black background. Plots in real size if `ax_scale_factor` is 1. 
        If the plot is saved it rasterizes the points because vector plots of
        milions of points get very huge. All other parts of the plot are 
        vectors.

        Args:
            genes (Union[List, np.ndarray]): List of genes to plot. First gene
                will be plotted first and thus be on the bottom of the stack.
            s (float, optional): Size of the points. Defaults to 0.1.
            ax_scale_factor (int, optional): Scale factor of the plot. If 1,
                the plot will be in real size. Carefully scale this for every 
                plot so that the plot does not become too small or too big.
                Defaults to 10.
            view (Union[Any, List], optional): If given it crops the points. 
                Should be a list of list with the the Left Bottom and Top Right
                corner coordinates: [[X_BL, Y_BL], [X_TR, Y_TR]]
                Defaults to None.
            scalebar (bool, optional): If True adds a scalebar.
                Defaults to True.
            show_axes (bool, optional): If True adds the axes to the plot.
                Defaults to False.
            save (bool, optional): If True saves the plot. Defaults to False.
            save_name (str, optional): Name of the plot. If not given will have
                the format: "Scatter_plot_<dataset_name>_<timestamp>"
                Defaults to ''.
            dpi (int, optional): Dots Per Inch (DPI) of the plot.
                Defaults to 300.
            file_format (str, optional): Format of the plot including the 
                point. Even if vector format is given the points will be 
                rasterized. Defaults to '.eps'.
        """
        #Make sure gene coordinate dictionary is present
        self.make_gene_coordinates()

        #Make figure
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)#, rasterized=True)
        ax.set_rasterization_zorder(1)

        #Plot points
        for g in genes:
            x = self.gene_coordinates[g][:, 0]
            y = self.gene_coordinates[g][:, 1]
            if isinstance(view, list):
                filt_x = (x > view[0][0]) & (x < view[1][0])
                filt_y = (y > view[0][1] )& (y < view[1][1])
                filt = filt_x & filt_y
                x = x[filt]
                y = y[filt]
            ax.scatter(x, y, s=s, color=self.color_dict[g], zorder=0)
            
        
        
        #Rescale
        if isinstance(view, list):
            x_extend = view[1][0] - view[0][0]
            y_extend = view[1][1] - view[0][1]
        else:
            x_extend = self.x_extend
            y_extend = self.y_extend

        x_scale = self.to_inch(x_extend * ax_scale_factor)
        y_scale = self.to_inch(y_extend * ax_scale_factor)
        self.set_size(x_scale, y_scale)

        #Add scale bar
        if scalebar:   
            self.add_scale_bar(ax)
        
        #Plot layout
        if not show_axes:
            ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0,0), 1, 1, facecolor=(0,0,0),
                                transform=ax.transAxes, zorder=-1))


        #plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        if save:
            if save_name == '':
                save_name = f'Scatter_plot_{self.dataset_name}_{strftime("%Y-%m-%d_%H-%M-%S")}'
            plt.savefig(f'{save_name}{file_format}', dpi=dpi, bbox_inches='tight', pad_inches=0)


class MultiGeneScatter(AxSize):

    def scatter_plot(self, genes: Union[List, np.ndarray], s: float=0.1, ax_scale_factor: int=10, 
                    scalebar: bool=True, show_axes: bool=False, 
                    save: bool=False, save_name: str='', dpi: int=300, file_format: str='.eps'):
        
        #Make sure gene coordinate dictionary is present
        self.make_multi_gene_coordinates()

        #Make figure
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)#, rasterized=True)
        ax.set_rasterization_zorder(1)

        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0

        #Plot points
        for d in self.datasets:
            for g in genes:
                x = d.gene_coordinates[g][:, 0]
                y = d.gene_coordinates[g][:, 1]

                x_min_g, x_max_g = x.min(), x.max()
                y_min_g, y_max_g = y.min(), y.max()
                x_min = x_min_g if x_min_g < x_min else x_min
                x_max = x_max_g if x_max_g > x_max else x_max
                y_min = y_min_g if y_min_g < y_min else y_min
                y_max = y_max_g if y_max_g > y_max else y_max

                ax.scatter(x, y, s=s, color=self.color_dict[g], zorder=0)
            
        #Rescale
        x_extend = x_max - x_min
        y_extend = y_max - y_min
        x_scale = self.to_inch(x_extend * ax_scale_factor)
        y_scale = self.to_inch(y_extend * ax_scale_factor)
        self.set_size(x_scale, y_scale)

        #Add scale bar
        if scalebar:   
            self.add_scale_bar(ax)
        
        #Plot layout
        if not show_axes:
            ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0,0), 1, 1, facecolor=(0,0,0),
                                transform=ax.transAxes, zorder=-1))


        #plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        if save:
            if save_name == '':
                save_name = f'Scatter_plot_{self.dataset_name}_{strftime("%Y-%m-%d_%H-%M-%S")}'
            plt.savefig(f'{save_name}{file_format}', dpi=dpi, bbox_inches='tight', pad_inches=0)