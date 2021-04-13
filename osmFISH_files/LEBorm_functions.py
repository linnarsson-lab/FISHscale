import pandas as pd
import numpy as np

#Not sure if all this is needed:
from fastcluster import linkage
import polo # 
from polo import optimal_leaf_ordering #from polo import polo
#Paper: http://bioinformatics.oxfordjournals.org/content/17/suppl_1/S22.long

from scipy.spatial.distance import pdist

from scipy.cluster import hierarchy
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

gene_sort_fish = ['Hybridization2_Gad2', 'Hybridization12_Slc32a1', 'Hybridization10_Crhbp', 'Hybridization12_Kcnip2', 'Hybridization13_Cnr1', 'Hybridization6_Vip', 'Hybridization5_Cpne5', 'Hybridization8_Pthlh',  'Hybridization10_Crh', 
'Hybridization1_Tbr1', 'Hybridization9_Lamp5', 'Hybridization7_Rorb', 'Hybridization11_Syt6',
'Hybridization1_Aldoc', 'Hybridization2_Gfap', 'Hybridization8_Serpinf1', 'Hybridization3_Mfge8',
 'Hybridization7_Sox10', 'Hybridization13_Plp1', 'Hybridization8_Pdgfra', 'Hybridization6_Bmp4','Hybridization6_Itpr2','Hybridization11_Tmem2', 'Hybridization7_Ctps','Hybridization5_Klk6','Hybridization9_Anln',
 'Hybridization3_Mrc1','Hybridization3_Hexb', 
 'Hybridization13_Ttr',
 'Hybridization1_Foxj1',
 'Hybridization12_Vtn',
 'Hybridization2_Flt1',
 'Hybridization10_Apln',
 'Hybridization5_Acta2',
 'Hybridization9_Lum']
 
gene_sort = ['Gad2', 'Slc32a1', 'Crhbp', 'Kcnip2', 'Cnr1', 'Vip', 'Cpne5', 'Pthlh', 'Crh',             
             'Tbr1', 'Lamp5', 'Rorb', 'Syt6', 
             'Aldoc', 'Gfap', 'Serpinf1', 'Mfge8', 
             'Sox10', 'Plp1', 'Pdgfra', 'Bmp4', 'Itpr2', 'Tmem2',  'Ctps',  'Klk6', 'Anln',   
             'Mrc1', 'Hexb', 
             'Ttr', 
             'Foxj1', 
             'Vtn', 'Flt1', 'Apln', 'Acta2',  'Lum'] 

def plot_cell_pos(coordinate_df, cell_ids=None, cell_of_interest=None, color='gray', color_highlight1='r', color_highlight2='r', s=5, standalone = True, mode='Highlight'):
    """
    Plot the centroids of all cells in grey. A selection of cells can be made red.
    And one cell of interest can be shown larger with a white border around.
    Input:
    `coordinate_df`(pd df): Pandas df with cells in columns, and 'X' & 'Y' as rows.
    `cell_ids`(list): If selection of cells to plot, enter a list of cell ids. Default = None
    `color`(str): color of cells, default gray
    `s`(float): Size of dots. Default = 5
    `stanalone`(bool): If true it creates a figure. If false it can be used as subplot
    `mode`(str): if 'highlight' it is possible to highlight a group of cells and/or one cell
    `color_highlight1`: color of group of cells. default='r' 
    `color_highlight2`: color of single cells. default='r'
    
    """
    if standalone == True:
        plt.figure(figsize=(7,7))
        
    if mode.lower() == 'tsne':
        plt.scatter(coordinate_df.loc['X',:], coordinate_df.loc['Y',:], linewidths=0, c=color, s=s)
    
    if mode.lower() == 'highlight':
        plt.scatter(coordinate_df.loc['X',:], coordinate_df.loc['Y',:], linewidths=0, c=color, s=s)
        if cell_ids != None:
            plt.scatter(coordinate_df.loc[:,cell_ids].loc['X'], coordinate_df.loc[:,cell_ids].loc['Y'], color=color_highlight1, s=s*2)
        if cell_of_interest != None:
            plt.scatter(coordinate_df.loc[:,cell_of_interest][0], coordinate_df.loc[:,cell_of_interest][1], color=color_highlight2, s=s*8, lw=2, edgecolor='w')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.gca().axis('equal')
    plt.xlim([coordinate_df.loc['X'].max(), coordinate_df.loc['X'].min()])
    plt.ylim([coordinate_df.loc['Y'].max(), coordinate_df.loc['Y'].min()])
    plt.gca().patch.set_facecolor((.9,.9,.9))


def gen_labels(df, model):
    """
    Generate cell labels from model.
    Input:
    `df`: Panda's dataframe that has been used for the clustering. (used to get
    the names of colums and rows)
    `model`(obj): Clustering object
    Returns (in this order):
    `cell_labels` = Dictionary coupling cellID with cluster label
    `label_cells` = Dictionary coupling cluster labels with cellID
    `cellID` = List of cellID in same order as labels
    `labels` = List of cluster labels in same order as cells
    `labels_a` = Same as "labels" but in numpy array
    
    """
    if str(type(model)).startswith("<class 'sklearn.cluster"):
        cell_labels = dict(zip(df.columns, model.labels_))
        label_cells = {}
        for l in np.unique(model.labels_):
            label_cells[l] = []
        for i, label in enumerate(model.labels_):
            label_cells[label].append(df.columns[i])
        cellID = list(df.columns)
        labels = list(model.labels_)
        labels_a = model.labels_
    elif type(model) == np.ndarray:
        cell_labels = dict(zip(df.columns, model))
        label_cells = {}
        for l in np.unique(model):
            label_cells[l] = []
        for i, label in enumerate(model):
            label_cells[label].append(df.columns[i])
        cellID = list(df.columns)
        labels = list(model)
        labels_a = model
    else:
        print('Error wrong input type')
    
    return cell_labels, label_cells, cellID, labels, labels_a

def sort_df(df, labels_a, row_sort=True, sorted_row_names=gene_sort_fish):
    """
    Sort the dataframe columns based on the cluster labels (additional row sort is optional).
    Input:
    `df`: Panda's dataframe that has been used for the clustering. (or a df
        that has the EXACT same order)
    `cluster_model`: Results of the clustering
    `row_sort`(bool): If True it will sort the rows of the dataframe, acording to the probided list
    `sorted_row_names`(list): List of row names. Default = gene_sort_fish
    
    """
    #Sort the dataframe with the new clusters
    #df_sort = pd.DataFrame(data=X, columns=df_fish.columns, index=df_fish.index)
    df_sort = df
    #labels_a = cluster_model.labels_ #Array of cell labels
    new_column_order = df_sort.columns[labels_a.argsort()]
    if row_sort == True:
        df_sort = df_sort.loc[sorted_row_names,new_column_order]
    else:
        df_sort = df_sort.loc[:,new_column_order]
    return df_sort
    

def sort_dataset_df(data, df, labels_a):# , cluster_model):
    """
    Sort a dataset based on the cluster labels. The data can be a normalized 
    np array. 
    Input:
    `data`(np array): Any normalized array, in the EXACT same order as the df on
        which the clustering algoritm ran.
    `df`: Panda's dataframe that has been used for the clustering. (used to get
        the names of colums and rows)
    Uses df_fish as basis.
    #`cluster_model`: Results of the clustering
    
    """    
    #Filter the datasetet to use the cells in the df
    data_filt = np.zeros((len(df.index), len(df.columns)))
    for i, n in enumerate(df.columns):
        #Assumes that data and df_fish are in the same format
        data_filt[:,i] = data[:,df_fish.columns.get_loc(n)]
    
    df_sort = pd.DataFrame(data=data_filt, columns=df.columns, index=df.index)
    new_column_order = df_sort.columns[labels_a.argsort()]
    
    df_sort = df_sort.loc[gene_sort_fish,new_column_order]
    return df_sort

def plot_labels(df, labels_a, standalone=True):
    if standalone == True:
        plt.figure(figsize=(10,10))
    points = np.zeros((len(df.columns),2))
    for i, n in enumerate(df.columns):
        points[i,:] = tSNE_points[df_fish.columns.get_loc(n),:]
    print(len(points))
    #plt.scatter(points[:,0], points[:,1],c=plt.cm.jet(labels_a/max(labels_a)), lw=0, alpha=1, s=15)
        #Color labels: See below, part 10
    plt.scatter(points[:,0], points[:,1],c=label_colors_hex, lw=0, alpha=1, s=15)


def plot_cells_pos_labels(labels_a):
    
    coord_df_sort = sort_df(coord_df.loc[:,cellID], labels_a, row_sort=False)
    color_labels_sort =plt.cm.jet(np.sort(labels_a)/max(np.sort(labels_a)))    
    #plot_cell_pos(coord_df_sort, cell_ids=None, color=color_labels_sort, s=10, standalone=False, mode='tsne')
        #Color labels: See below, part 10
    plot_cell_pos(coord_df_sort, cell_ids=None, color=label_colors_hex, s=10, standalone=False, mode='tsne')
    
    

def tSNE_and_pos(df, labels_a, save=False):
    """
    Plot the tSNE and cell positions with the cluster colors
    
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
    plt.sca(axes[0])
    plot_labels(df, labels_a, standalone=False)
    plt.title('tSNE')
    plt.axis('off')
    plt.sca(axes[1])
    plot_cells_pos_labels(labels_a)
    plt.title('cell positon')
    plt.axis('off')
    if save == True:
        plt.savefig('Cluster_tSNE_and_Position.png', dpi=600)
        
def mean_expression(df, labels):
    """
    Make dataframe with mean expression
    
    """
    #Make df with count averages per cluster
    df_count_average = pd.DataFrame(index=df.index, columns=np.sort(np.unique(labels))) #np.unique(labels_a))
    for l in np.unique(labels):
        filt = np.sort(labels) == l
        mean = np.array(df.loc[:,filt].T.mean())
        #std = np.array(np.std(df_sort.loc[:,filt], axis=1))
        if np.isnan(np.sum(mean)) == False: #In case some clusters do not have cells
            df_count_average[l] = mean
    return df_count_average
    
    
def heat_map(df, labels, sort=None, save=False):
    """
    Plot heat_map of a sorted dataframe
    
    """
    #Find the name of the input df, for logging
    #df_input_name =[x for x in globals() if globals()[x] is df][0]
    #print('DF used for plot: {}'.format(df_input_name))
    
    cell_labels = dict(zip(df.columns, labels))
    label_cells = {}
    for l in np.unique(labels):
        label_cells[l] = []
    for i, label in enumerate(labels):
            label_cells[label].append(df.columns[i])
    
    if sort == None:
        #Make df with count averages per cluster
        df_count_average = mean_expression(df, labels)
        #Make optimal sort on average expression of each cluster
        #Transpose, otherwise you are doing it on genes instead of clusters
        D = pdist(df_count_average.T, 'cityblock') #Working well: 'correlation', 'cityblock', 'seuclidean', 'canberra', 'cosine'
        Z = linkage(D, 'ward')
        optimal_Z = optimal_leaf_ordering(Z, D)
        optimal_o = polo.polo.leaves_list(optimal_Z) 
        #In case some clusters are missing
        optimal_order = []
        for i in optimal_o:
            optimal_order.append(df_count_average.columns[i])
    else:
        optimal_order = sort
    print('Order of clusters: {}'.format(optimal_order))
    
    #Sort the cells according to the optimal cluster order
    optimal_sort_cells = []
    for i in optimal_order:
        optimal_sort_cells.extend(label_cells[i])
    
    
    #Create a list of optimal sorted cell labels
    optimal_sort_labels = []
    for i in optimal_sort_cells:
        optimal_sort_labels.append(cell_labels[i])
    
    fig, axHM = plt.subplots(figsize=(14,6))
    #z = df.loc[:,optimal_sort_cells].values
    #z = z/np.percentile(z, 99.5, 1)[:,None]
    
    df_full = df # _fish_log
    z = df_full.values
    z = z/np.percentile(z, 99, 1)[:,None]
    z = pd.DataFrame(z, index=df_full.index, columns=df_full.columns)
    z = z.loc[:,optimal_sort_cells].values
    print(z.shape)
    
    im = axHM.pcolor(z, cmap='viridis', vmax=1)

    plt.yticks(np.arange(0.5, len(df.index), 1), gene_sort, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlim(xmax=len(labels))
    print(len(labels))
    #plt.title(df_input_name)

    divider = make_axes_locatable(axHM)
    axLabel = divider.append_axes("top", .3, pad=0, sharex=axHM)

    optimal_sort_labels = np.array(optimal_sort_labels)
    axLabel.pcolor(optimal_sort_labels[None,:]/max(optimal_sort_labels), cmap='prism')
        #Colors, see below:
    #axLabel.pcolor(label_colors_rgb) #label_colors_hex
    
    
    
    axLabel.set_xlim(xmax=len(labels))
    axLabel.axis('off')
    
    cax = fig.add_axes([.91, 0.13, 0.01, 0.22])
    colorbar = fig.colorbar(im, cax=cax, ticks=[0,1])
    colorbar.set_ticklabels(['0', 'max'])

    if save == True:
        plt.savefig('/home/lars/storage/Documents/Cortex_FISH/Heatmap_{}clusters_{}.png'.format(len(np.unique(labels_a)), df_input_name), dpi=300)
    
    return optimal_order
