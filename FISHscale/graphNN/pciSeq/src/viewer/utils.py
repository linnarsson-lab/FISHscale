""" Functions to manipulate flat files (split or minify them) """
from typing import Union
import pandas as pd
import numpy as np
import json
import os
import glob
import csv
import logging

logger = logging.getLogger(__name__)

'''from pciSeq import check_libvips
if check_libvips():
    import pyvips'''


def splitter_mb(df, dir_path, mb_size):
    """ Splits a text file in (almost) equally sized parts on the disk. Assumes that there is a header in the first line
    :param filepath: The path of the text file to be broken up into smaller files
    :param mb_size: size in MB of each chunk
    :return:
    """
    # OUT_DIR = os.path.join(os.path.splitext(filepath)[0] + '_split')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        files = glob.glob(dir_path + '/*.*')
        for f in files:
            os.remove(f)

    n = 0
    header_line = df.columns.tolist()
    # header_line = next(handle)[1].tolist()
    file_out, handle_out = _get_file(dir_path, n, header_line)
    # data_row = next(handle)[1].tolist()
    for index, row in df.iterrows():
        row = row.tolist()
        size = os.stat(file_out).st_size
        if size > mb_size*1024*1024:
            logger.info('saved %s with file size %4.3f MB' % (file_out, size/(1024*1024)))
            n += 1
            handle_out.close()
            file_out, handle_out = _get_file(dir_path, n, header_line)
        write = csv.writer(handle_out, delimiter='\t')
        write.writerow(row)

    # print(str(file_out) + " file size = \t" + str(size))
    logger.info('saved %s with file size %4.3f MB' % (file_out, size / (1024 * 1024)))
    handle_out.close()


def splitter_mb(filepath, mb_size):
    """ Splits a text file in (almost) equally sized parts on the disk. Assumes that there is a header in the first line
    :param filepath: The path of the text file to be broken up into smaller files
    :param mb_size: size in MB of each chunk
    :return:
    """
    handle = open(filepath, 'r')
    OUT_DIR = os.path.join(os.path.splitext(filepath)[0] + '_split')

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    else:
        files = glob.glob(OUT_DIR + '/*.*')
        for f in files:
            os.remove(f)

    n = 0
    header_line = next(handle)
    file_out, handle_out = _get_file(OUT_DIR, filepath, n, header_line)
    for line in handle:
        size = os.stat(file_out).st_size
        if size > mb_size*1024*1024:
            print('saved %s with file size %4.3f MB' % (file_out, size/(1024*1024)))
            n += 1
            handle_out.close()
            file_out, handle_out = _get_file(OUT_DIR, filepath, n, header_line)
        handle_out.write(str(line))

    # print(str(file_out) + " file size = \t" + str(size))
    print('saved %s with file size %4.3f MB' % (file_out, size / (1024 * 1024)))
    handle_out.close()


def splitter_n(filepath, n):
    """ Splits a text file in n smaller files
    :param filepath: The path of the text file to be broken up into smaller files
    :param n: determines how many smaller files will be created
    :return:
    """
    filename_ext = os.path.basename(filepath)
    [filename, ext] = filename_ext.split('.')

    OUT_DIR = os.path.join(os.path.splitext(filepath)[0] + '_split')

    if ext == 'json':
        df = pd.read_json(filepath)
    elif ext == 'tsv':
        df = pd.read_csv(filepath, sep='\t')
    else:
        df = None

    df_list = np.array_split(df, n)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    else:
        files = glob.glob(OUT_DIR + '/*.'+ext)
        for f in files:
            os.remove(f)

    for i, d in enumerate(df_list):
        fname = os.path.join(OUT_DIR, filename + '_%d.%s' % (i, ext))
        if ext == 'json':
            d.to_json(fname,  orient='records')
        elif ext == 'tsv':
            d.to_csv(fname, sep='\t', index=False)


def crush_data(d):
    """ Minifies the flatfiles that will be fed to the viewer.
    I keep only 3 decimal points and no more than the top 10 cell classes as
    possible assignments for any given cell
    """

    cell_min = None
    gene_min = None
    n = 10 # for each cell keep only the top10 possible cell types. Discard the rest. USE THIS WITH CAUTION. MAYBE IT IS NOT A GOOD IDEA AND I SHOULD KEEP ALL POSSIBLE CELL TYPES
    try:
        cellData_path = d['cellData']
        cell_min = _crush_cellData(cellData_path, n)
    except KeyError as e:
        print('key doesnt exist...')

    try:
        geneData_path = d['geneData']
        gene_min = _crush_geneData(geneData_path)
    except KeyError as e:
        print('key doesnt exist...')

    return [cell_min, gene_min]


def _crush_cellData(filepath, n):
    filename_ext = os.path.basename(filepath)
    [filename, ext] = filename_ext.split('.')

    cellData = pd.read_csv(filepath, sep='\t')
    temp = _order_prob(cellData, n)
    cellData['ClassName'] = temp[0]
    cellData['Prob'] = temp[1]

    cellData['Prob'] = _round_data2(cellData, 'Prob')
    cellData['CellGeneCount'] = _round_data(cellData, 'CellGeneCount')

    out_dir: Union[bytes, str] = os.path.join(os.path.dirname(filepath), filename + '_min')
    target_path = os.path.join(out_dir, filename_ext)
    _clean_dir(out_dir, ext)

    cellData.to_csv(target_path, sep='\t', index=False)
    print('Minimised file saved at: %s' % target_path)

    return target_path


def _crush_geneData(filepath):
    filename_ext = os.path.basename(filepath)
    [filename, ext] = filename_ext.split('.')

    geneData = pd.read_csv(filepath, sep='\t')

    # 1. First, do geneData
    # use int for the coordinates, not floats
    geneData = geneData.astype({'x': 'int32', 'y': 'int32'})
    geneData['neighbour_prob'] = _round_data(geneData, 'neighbour_prob')

    out_dir = os.path.join(os.path.dirname(filepath), filename + '_min')
    target_path = os.path.join(out_dir, filename_ext)
    _clean_dir(out_dir, ext)

    # save geneData
    geneData.to_csv(target_path, sep='\t', index=False)
    print('Minimised file saved at: %s' % target_path)

    return target_path


def _clean_dir(out_dir, ext):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        files = glob.glob(out_dir + '/*.'+ext)
        for f in files:
            print('removed file %s' % f)
            os.remove(f)


def _get_file(OUT_DIR, n, header_line):
    filename = os.path.basename(OUT_DIR).split('.')[0]
    file = os.path.join(OUT_DIR, filename + '_%d.%s' % (n, 'tsv'))
    handle = open(file, "a", newline='', encoding='utf-8')
    write = csv.writer(handle, delimiter='\t')
    write.writerow(header_line)
    return file, handle


def _order_prob(df, n, class_name=[], prob=[]):
    '''
    orders the list that keeps the cell classes probs from highest to lowest.
    Rearranges then the cell class names appropriately
    :param df:
    :param n:
    :param class_name:
    :param prob:
    :return:
    '''
    for index, row in df.iterrows():
        cn = np.array(json.loads(row['ClassName'].replace("'", '"')))
        p = np.array(json.loads(row['Prob']))

        idx = np.argsort(p)[::-1]
        cn = cn[idx]
        cn = cn.tolist()
        p = p[idx]
        class_name.append(cn[:n]) # keep the top-n only and append
        prob.append(p[:n])
    return [class_name, prob]


def rotate_image(img_in, img_out, deg):
    """
    rotates an image.
    img_in: path to the image to be rotated
    img_out: path to save the rotated image to
    deg: degrees to rotate the image by (clockwise)
    """
    x = pyvips.Image.new_from_file(img_in)
    x = x.rotate(deg, interpolate=pyvips.Interpolate.new("nearest"))
    x.write_to_file(img_out, compression="jpeg", tile=True)


_format = lambda x: round(x, 3) # keep only 3 decimal points


def _round_data(df, name):
    neighbour_prob = [json.loads(x) for x in df[name]]
    return [list(map(_format, x)) for x in neighbour_prob]


def _round_data2(df, name):
    return [list(map(_format, x)) for x in df[name]]



if __name__ == "__main__":
    mb_size = 99
    n = 4
    filepaths = [r"geneData.tsv"]
    for filepath in filepaths:
        splitter_mb(filepath, mb_size)
        splitter_n(filepath, n)