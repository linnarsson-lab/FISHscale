""" Functions for extracting the boundaries of the cells """

# WARNING -- WARNING -- WARNING
# NOTE: 30-Nov-2020: I am using pydip to get the cell boundaries. It is a lot faster but I need to
# further test this and compare the cell boundaries with the previous way i used to do it

#import cv2
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage import binary_erosion
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
import diplib as dip
import time
import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s:%(levelname)s:%(message)s"
# )
#
# logger = logging.getLogger()



def cell_boundaries(stage, cell_props):
    '''
    calculate the outlines of the cells
    :return:
    '''

    # loop over the self.cell_props
    res_list = []
    for tile in stage.tiles:
        if np.any(tile['label_image'].data):
            df = obj_outline(tile, cell_props)
            res_list.append(df)
        else:
            logger.info('tile:%d empty, No cells to draw boundaries were found' % tile['tile_id'])
    _df = pd.concat(res_list).astype({"label": int})

    # make a Dataframe to keep boundaries of the cells which are not clipped by the tile
    df_1 = _df.iloc[np.isin(_df.label, cell_props[~cell_props.is_clipped].label)]

    # get the labels of the clipped cells
    in_multiple_tiles = sorted(cell_props[cell_props.is_clipped].label.values)
    logger.info('There are %d cells whose boundaries span across multiple tiles' % len(in_multiple_tiles))

    # find the boundaries of the clipped cells
    _list = collate_borders_par(stage, in_multiple_tiles)
    df_2 = pd.DataFrame(_list.items(), columns=['label', 'coords'])\
        .astype({"label": int})
    # df_2 = pd.DataFrame(_list).astype({"label": int})

    # Both clipped and unclipped in a dataframe
    res = pd.concat([df_1, df_2])

    set_diff = set(cell_props.label) - set(res.label.values)
    if set_diff:
        unresolved_labels = pd.DataFrame({'label': list(set_diff), 'coords': np.nan * np.ones(len(set_diff))})
        res = pd.concat([res, unresolved_labels])

    # assert set(_df.label.values) == set(res.label.values)
    assert res.shape[0] == cell_props.shape[0]
    assert np.all(sorted(res.label) == sorted(cell_props.label))
    assert np.unique(res.label).size == res.shape[0], 'Array cannot have duplicates'
    return res.sort_values(['label'], ascending=[True])



def collate_borders_par(stage, labels):
    n = max(1, cpu_count() - 1)
    pool = ThreadPool(16)
    results = {}
    pool.map(collate_borders_helper(stage, results), labels)
    pool.close()
    pool.join()
    return results

def collate_borders_helper(stage, results):
    def inner_fun(label):
        logger.info('label: %d. Finding the cell boundaries' % label)
        label_image = stage.collate_arrays(stage.merge_register[label])
        offset_x, offset_y = stage.find_offset(stage.merge_register[label])
        results[label] = np.array(get_label_contours(label_image, label, offset_x, offset_y))
    return inner_fun

def obj_outline(tile, cell_props):
    logger.info('Getting cell boundaries for cells in tile: %d' % tile['tile_id'])
    label_image = tile['label_image'].toarray()
    offset_x = tile['tile_offset_x']
    offset_y = tile['tile_offset_y']
    clipped_cells = cell_props[cell_props.is_clipped].label.values

    df = extract_borders_dip(label_image.astype(np.uint64), offset_x, offset_y, clipped_cells)
    return df


def extract_borders(label_image, offset_x, offset_y, clipped_labels):
    """
    same as ''extract_borders_par'' but without parallelism
    :param label_image:
    :return:
    """
    labels = sorted(set(label_image.flatten()) - {0} - set(clipped_labels))
    out = {}
    for label in labels:
        y = label_image == label
        y = y * 255
        y = y.astype('uint8')
        contours, hierarchy = cv2.findContours(y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(offset_x, offset_y))
        contours = np.squeeze(contours)
        out[label] = contours.tolist()
    out = pd.DataFrame([out]).T
    out = out.reset_index()
    out.columns = ['label', 'coords']
    return out


def extract_borders_par(label_image, offset_x, offset_y, clipped_labels):
    """ Extracts the borders of the objects from the label_image

    :param label_image: a 2-d array, the same size as the image, where the value at position (i,j) denotes the label of
                        the object that corresponds at pixel position (i, j) of the image. If a pixel is on the background
                        then the label is zero
    :param offset_x: The x-coordinate of the top left corner of the image
    :param offset_y: The y-coordinate of the top left corner of the image
    :param clipped_labels: a list containing the labels of the cells that will be excluded from border extraction
    :return: A dataframe with columns '''label''' and '''coords'''. Column '''label''' contains the label of the cell
            and column '''coords''' contains a list of pairs describing the coordinates of the cell boundaries/contours.
            Each such pair is a list.
    """

    labels = sorted(set(label_image.flatten()) - {0} - set(clipped_labels))
    out = extract_borders_helper([label_image, offset_x, offset_y], labels)
    out = pd.DataFrame([out]).T
    out = out.reset_index()
    out.columns = ['label', 'coords']
    return out.sort_values(by=['label'])


def extract_borders_helper(args, labels):
    n = max(1, cpu_count() - 1)
    pool = ThreadPool(n)
    results = {}
    pool.map(wrapper_helper(args, results), labels)
    pool.close()
    pool.join()
    return results


def wrapper_helper(argsin, results):
    '''
    closure function to pass parameters to the function called by pool.map
    :param argsin:
    :param results:
    :return:
    '''
    def inner_fun(label):
        label_image = argsin[0]
        offset_x = argsin[1]
        offset_y = argsin[2]
        contours = get_label_contours(label_image, label, offset_x, offset_y)
        results[label] = contours
    return inner_fun


def get_label_contours(label_image, label, offset_x, offset_y):
    '''
    reads a label_image and gets the boundaries of the cell labeled by ''label''
    :param label_image:
    :param label:
    :param offset_x:
    :param offset_y:
    :return:
    '''
    img = label_image == label
    img = img * 255
    img = img.astype('uint8')
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(offset_x, offset_y))
    contours = np.squeeze(contours)
    return contours.tolist()


def extract_borders_dip(label_image, offset_x, offset_y, clipped_labels):
    """
    NOTES: # using Simplify drastically reduces the array that describes the polygon boundaries but you
             might end up with a slightly different polygon. The difference is only on a very few pixels.
             I do not know which one reflect the actual boundaries more closely, but using Simplify and
             reduce the size of the boudaries array is convenient. Also having few different pixels is not
             crucial, hence I think keeping Simplify makes sense
    Parameters
    ----------
    label_image
    offset_x
    offset_y
    clipped_labels

    Returns
    -------

    """
    labels = sorted(set(label_image.flatten()) - {0} - set(clipped_labels))
    cc = dip.GetImageChainCodes(label_image)  # input must be an unsigned integer type
    d = {}
    for c in cc:
        if c.objectID in labels:
            # p = np.array(c.Polygon())
            p = c.Polygon().Simplify()
            p = p + np.array([offset_x, offset_y])
            p = np.uint64(p).tolist()
            p.append(p[0])  # append the first pair at the end to close the polygon
            d[np.uint64(c.objectID)] = p
        else:
            pass
    df = pd.DataFrame([d]).T
    df = df.reset_index()
    df.columns = ['label', 'coords']
    return df


def outline_fix(label_image):
    res_list = []
    coo = coo_matrix(label_image)
    labels = np.unique(coo.data)
    for label in sorted(set(labels)):
        # print('label: %d' % label)
        c = coo.copy()
        c.data[c.data != label] = 0
        c = c.toarray()
        mask = binary_erosion(c)
        c[mask] = 0
        c = coo_matrix(c)
        if c.data.size > 0:
            df = pd.DataFrame({'coords': list(zip(c.col, c.row)), 'label': c.data})
            df = df.groupby('label')['coords'].apply(lambda group_series: group_series.tolist()).reset_index()
            df = df.astype({"label": int})
        else:
            df = pd.DataFrame()
        res_list.append(df)

    if res_list:
        out = pd.concat(res_list).astype({"label": int})
    else:
        out = pd.DataFrame()

    return out



if __name__ == "__main__":
    dummy_img = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 0, 0, 0],
        [0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 0, 0, 0],
        [0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 0, 0, 0],
        [0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 0, 0, 0],
        [0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    fov51 = pd.read_csv('../../fov51.csv', header=None)
    fov51 = fov51.values

    fov56 = pd.read_csv('../../fov56.csv', header=None)
    fov56 = fov56.values

    # x = dummy_img
    x = fov56

    start = time.time()
    out = extract_borders(x, 0, 0, [0])
    print(time.time() - start)

    # start = time.time()
    # out2 = outline_fix(x)
    # print(time.time() - start)

    start = time.time()
    # out3 = extract_borders(x, 0, 0, [0])
    out3 = extract_borders_par(x, 0, 0, [0])
    print(time.time() - start)

    print(out)
    # print(out2)
    print(out3)

    print('Done')