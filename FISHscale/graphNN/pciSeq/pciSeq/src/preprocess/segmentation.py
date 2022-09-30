import pandas as pd
import numpy as np
#import cv2
from scipy import ndimage
from skimage import morphology
from skimage import exposure, feature
from skimage.measure import regionprops
from skimage import segmentation
from scipy.ndimage import label, generate_binary_structure
from scipy.sparse import coo_matrix
from scipy import sparse
from src.preprocess.imimposemin import imimposemin
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
# from skimage.morphology import erosion
from skimage.morphology import disk
import math
from matplotlib import pyplot as plt
import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s:%(levelname)s:%(message)s"
# )
# logger = logging.getLogger()



# https://stackoverflow.com/questions/52725553/difference-between-the-functions-im2uint8-in-matlab-and-bytescale-in-pyth

def _imadjust(img):
    upper = np.percentile(img, 99)
    lower = np.percentile(img, 1)
    out = (img - lower) * (255 / (upper - lower))
    np.clip(out, 0, 255, out) # in-place clipping
    return np.around(out)


def imadjust2(img, lims):
    lims = lims.flatten()
    img2 = np.copy(img)
    lowIn = lims[0]
    highIn = lims[1]
    lowOut = 0
    highOut = 1
    gamma = 1
    lut = adjustWithLUT(img2, lowIn, highIn, lowOut, highOut, gamma)
    return lut[img2].astype(np.uint8)


def adjustWithLUT(img,lowIn,highIn,lowOut,highOut,gamma):
    lutLength = 256 # assumes uint8
    lut = np.linspace(0, 1, lutLength)
    lut = adjustArray(lut, lowIn, highIn, lowOut, highOut, gamma)
    lut = _img_as_ubyte(lut)
    return lut


def _img_as_ubyte(x):
    out = np.zeros(x.shape)
    out[x==0.3] = 77
    out[x!=0.3] = img_as_ubyte(x)[x!=0.3]
    return out


def adjustArray(img, lIn, hIn, lOut, hOut, g):
    # %make sure img is in the range [lIn;hIn]
    img = np.maximum(lIn, np.minimum(hIn, img));

    out = ((img - lIn) / (hIn - lIn)) ** g
    out = out ** (hOut - lOut) + lOut
    return out


def stretchlim(img):
    nbins = 255
    tol_low = 0.01
    tol_high = 0.99
    sz = np.shape(img)
    if len(sz) == 2:
        img = img[:, :, None]
        sz = np.shape(img)

    p = sz[2]
    ilowhigh = np.zeros([2, p])
    for i in range(0,p):
        hist,bins = np.histogram(img[:, :, i].ravel(), nbins+1, [0, nbins])
        cdf = np.cumsum(hist) / sum(hist)
        ilow = np.argmax(cdf > tol_low)
        ihigh = np.argmax(cdf >= tol_high)
        if ilow == ihigh:
            ilowhigh[:, i] = np.array([1, nbins])
        else:
            ilowhigh[:, i] = np.array([ilow, ihigh])

    lims = ilowhigh / nbins
    return lims


def disk(n):
    struct = np.zeros((2 * n + 1, 2 * n + 1))
    x, y = np.indices((2 * n + 1, 2 * n + 1))
    mask = (x - n)**2 + (y - n)**2 <= n**2
    struct[mask] = 1
    return struct.astype(np.uint8)


def imregionalmax(image, ksize=3):
  """Similar to matlab's imregionalmax"""
  filterkernel = np.ones((ksize, ksize)) # 8-connectivity
  reg_max_loc = feature.peak_local_max(image,
                               footprint=filterkernel, indices=False,
                               exclude_border=0)
  return reg_max_loc.astype(np.uint8)

# selem = morphology.disk(20)
# morphology.erosion(image, selem)

def watershed(img_url):
    # set the parameters
    DapiThresh = 90
    DapiMinSize = 5
    DapiMinSep = 7
    DapiMargin = 10
    MinCellArea = 200

    Dapi = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)
    lims = stretchlim(Dapi)
    # Dapi = _imadjust(Dapi)


    logger.info('Step 1')
    # STEP 1: Erode the image
    Dapi = imadjust2(Dapi, lims)
    ThresVal = np.percentile(Dapi[Dapi>0], DapiThresh)
    # kernel = disk(2)
    # image = cv2.erode(Dapi>ThresVal), kernel)
    # bwDapi = ndimage.binary_erosion(Dapi>ThresVal, structure=disk(2)).astype(int)
    bwDapi = morphology.erosion(Dapi>ThresVal, morphology.disk(2)).astype(int)

    logger.info('Step 2')
    # STEP 2: Compute the distance map
    dist = ndimage.distance_transform_edt(bwDapi)
    dist0 = dist.copy()
    dist0[dist < DapiMinSize] = 0

    logger.info('Step 3')
    # STEP 3: Dilate to remove small patterns
    selem = morphology.disk(DapiMinSep)
    ddist = morphology.dilation(dist0, selem)

    logger.info('Step 4')
    # STEP 4: modify the image so that cells are -inf(basins for the watershed method below)
    markers = imregionalmax(ddist) # marks with 1s the cells, 0 for background
    impim = imimposemin(-dist0, markers) # returns an array with negative values apart from the locations of the markers

    logger.info('Watershed')
    L = segmentation.watershed(impim, watershed_line=True)
    bwDapi0 = bwDapi.copy()
    bwDapi0[L == 0] = 0

    # % assign all pixels a label
    s = generate_binary_structure(2,2)
    labels, num_Features = label(bwDapi0, structure=s)
    d, _idx = ndimage.distance_transform_edt(bwDapi0 == 0, return_indices=True)
    idx = np.ravel_multi_index(_idx, bwDapi.shape)


    # % now expand the regions by a margin
    # CellMap0 = np.zeros(Dapi.shape).astype(np.uint32)
    Expansions = d < DapiMargin
    # CellMap0[Expansions] = labels[idx[Expansions]];
    CellMap0 = np.take(labels.flatten(), idx) * Expansions

    rProps0 = regionprops(CellMap0)

    # BigEnough = np.array([x.area > 200 for x in rProps0 ])
    # NewNumber = np.zeros(len(rProps0))
    # NewNumber[~BigEnough] = 0
    # NewNumber[BigEnough] = np.arange(1, 1+BigEnough.sum())
    # CellMap = CellMap0
    # CellMap[CellMap0 > 0] = NewNumber[CellMap0[CellMap0 > 0]]
    coo = coo_matrix(CellMap0)
    coo_data = coo.data.copy()
    to_remove = np.array([x.label for x in rProps0 if x.area <= MinCellArea])
    coo_data[np.in1d(coo_data, to_remove)] = 0
    _, res = np.unique(coo_data, return_inverse=True)
    coo.data = res

    return coo, CellMap0



if __name__ == "__main__":

    img_url = r"D:\Home\Dimitris\OneDrive - University College London\Data\David\Midbrain Sequencing Data\data\background_image\background_image.tif"
    coo, CellMap_0 = watershed(img_url)

    sparse.save_npz('coo_matrix.npz', coo)

    my_image = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)
    overlay = label2rgb(coo.toarray(), image=my_image, bg_label=0)
    # overlay = label2rgb(coo.toarray(), bg_label=0, bg_color=(1,1,1)) # white background
    # plt.imshow(overlay[:, :, 1], cmap='gray', interpolation='none')
    my_dpi = 72
    fig, ax = plt.subplots(figsize=(6000/my_dpi, 6000/my_dpi), dpi=my_dpi)
    plt.imshow(overlay)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    fig.savefig('david_segmented.tif', dpi=200)

    print(coo.data.max())
    print('done')

    # https://stackoverflow.com/questions/5260232/matlab-octave-bwdist-in-python-or-c

    # v = np.ones_like(i)
    # mat = coo_matrix((v, (i, j)), shape=(n, n))