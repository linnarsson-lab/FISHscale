import h5py
import numpy as np
from scipy.sparse import load_npz
import os
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from collections import defaultdict
import logging


# logger = logging.getLogger(__name__)
# logger.disabled = True


def _to_csr_matrix(i, j, n):
    """Using i and j as coo-format coordinates, return csr matrix."""
    n = int(n)
    v = np.ones_like(i)
    mat = coo_matrix((v, (i, j)), shape=(n, n))
    return mat.tocsr()


def get_dir(my_config, tile_id):
    root_dir = my_config['FOV_ROOT']
    return os.path.join(root_dir, 'tile_' + str(tile_id))


def _get_connected_labels(lst):
    '''
    find which positions in the input list have repeated values
    Example:
     If lst = [0, 4, 4] then it returns [1, 2] because 4 appears twice, at positions 1 and 2 of the input list
     if lst = [0,1,2,1,3,4,2,2] then it returns [[1, 3], [2, 6, 7]]
    :param lst:
    :return:
    '''
    output = defaultdict(list)
    # Loop once over lst, store the indices of all unique elements
    for i, el in enumerate(lst):
        output[el].append(i)
    return np.array([np.array(d) for d in output.values() if len(d) > 1])


def load_mat(filepath):
    '''
    reads a Matlab mat file and returns the CellMap
    :param filepath:
    :return:
    '''
    arrays = {}
    f = h5py.File(filepath, 'r')
    for k, v in f.items():
        arrays[k] = np.array(v)

    # CAUTION: TAKE THE TRANSPOSE. MORE DETAILS ON
    # https://www.mathworks.com/matlabcentral/answers/308303-why-does-matlab-transpose-hdf5-data
    logger.info('***** Returning the transpose of the input Matlab array *******')
    return arrays['CellMap'].T


def load_mat_2(filepath):
    x = loadmat(filepath)
    return x['CellMap']


def tilefy(a, p, q):
    """
    Splits array '''a''' into smaller subarrays of size p-by-q
    Parameters
    ----------
    a: Numpy array of size m-by-n
    p: int, the height of the tile
    q: int, the length of the tile

    Returns
    -------
    A list of numpy arrays

    Example:
    a = np.arange(21).reshape(7,3)
    out = tilefy(a, 4, 2)

    will return
     [array([[ 0,  1],
        [ 3,  4],
        [ 6,  7],
        [ 9, 10]]),
     array([[ 2],
            [ 5],
            [ 8],
            [11]]),
     array([[12, 13],
            [15, 16],
            [18, 19]]),
     array([[14],
            [17],
            [20]])]

    """
    m, n = a.shape
    ltr = -(-n//q)  # left to right
    ttb = -(-m//p)  # top to bottom
    out = []
    for j in range(ttb):
        rows = np.arange(p) + j*p
        for i in range(ltr):
            cols = np.arange(q) + i*q
            _slice = a[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
            out.append(_slice.astype(np.int32))
    return out


# def split_CellMap(filepath, p, q=None):
#     if q is None:
#         q = p
#     cellmap = load_mat(filepath)
#     # p = q = 2000
#     out = blockfy(cellmap, p, q)
#     return out


def split_label_img(filepath, p, q):
    """
    Splits the label_image into smaller chunks(tiles) of size p-by-q
    Parameters
    ----------
    filepath: Path to the npy file (the output of the cell segmentation)
    p: height in pixels of the tile
    q: width in pixels of the tile

    Returns
    -------
    """
    label_img = load_npz(filepath).toarray()
    # label_img = np.load(filepath)
    logger.info('label image loaded from %s' % filepath)
    logger.info('Width: %d, height: %d' % (label_img.shape[1], label_img.shape[0]))
    out = tilefy(label_img, p, q)
    return out

# if __name__ == "__main__":
#     p = q = 2000
#     filepath = os.path.join(config.ROOT_DIR, 'CellMap_left.mat')
#     fov_list = split_CellMap(filepath, p, q)
#     print('Done!')

