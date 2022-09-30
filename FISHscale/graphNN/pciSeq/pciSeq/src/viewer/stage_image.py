''''
splits the big background image  into smaller images size 2000x2000px.
Look also at https://stackoverflow.com/questions/10853119/chop-image-into-tiles-using-vips-command-line/15293104
for an alternative (and probably better way)
It also creates the pyramid tiles for the viewer
'''
import shutil
import os
import pyvips
import logging

# logger = logging.getLogger(__name__)


def split_image(im):
    '''
    you can just do:
        im.dzsave('./out', suffix='.tif', skip_blanks=-1, background=0, depth='one', overlap=0, tile_size=2000, layout='google')
    to split the image to smaller squares. However you need to write a couple of line to rename and move the file to the correct
    folders
    :param im:
    :return:
    '''
    im = pyvips.Image.new_from_file(im, access='random')
    tile_size = 2000;

    if im.width % tile_size == 0:
        tiles_across = int(im.width / tile_size)
    else:
        tiles_across = im.width // tile_size + 1


    if im.width % tile_size == 0:
        tiles_down = int(im.height/tile_size)
    else:
        tiles_down = im.height // tile_size + 1

    image = im.gravity('north-west', tiles_across * tile_size, tiles_down * tile_size)

    for j in range(tiles_down):
        logger.info('Moving to the next row: %d/%d '% (j, tiles_down-1) )
        y_top_left = j * tile_size
        for i in range(tiles_across):
            x_top_left = i * tile_size
            tile = image.crop(x_top_left, y_top_left, tile_size, tile_size)
            tile_num = j * tiles_across + i
            fov_id = 'fov_' + str(tile_num)

            out_dir = os.path.join(config.ROOT_DIR, 'fov', fov_id, 'img')
            full_path = os.path.join(out_dir, fov_id +'.tif')
            if not os.path.exists(os.path.dirname(full_path)):
                os.makedirs(os.path.dirname(full_path))
            tile.write_to_file(full_path)
            logger.info('tile: %s saved at %s' % (fov_id, full_path) )


def map_image_size(z):
    '''
    return the image size for each zoom level. Assumes that each map tile is 256x255
    :param z: 
    :return: 
    '''

    return 256 * 2 ** z


def tile_maker(z_depth, out_dir, img_path):
    """
    Makes a pyramid of tiles.
    z_depth: (int) Specifies how many zoom levels will be produced
    out_dir: (str) The path to the folder where the output (the pyramid of map tiles) will be saved to. If the folder
                   does not exist, it will be created automatically
    img_path: (str) The path to the image
    """
    # img_path = os.path.join(dir_path, 'demo_data', 'background_boundaries.tif')

    dim = map_image_size(z_depth)
    # remove the dir if it exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # now make a fresh one
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    im = pyvips.Image.new_from_file(img_path, access='sequential')

    # The following two lines add an alpha component to rgb which allows for transparency.
    # Is this worth it? It adds quite a bit on the execution time, about x2 increase
    # im = im.colourspace('srgb')
    # im = im.addalpha()

    logger.info('Resizing image: %s' % img_path)
    factor = dim / max(im.width, im.height)
    im = im.resize(factor)
    logger.info('Done! Image is now %d by %d' % (im.width, im.height))
    pixel_dims = [im.width, im.height]

    # sanity check
    assert max(im.width, im.height) == dim, 'Something went wrong. Image isnt scaled up properly. ' \
                                            'It should be %d pixels in its longest side' % dim

    # im = im.gravity('south-west', dim, dim) # <---- Uncomment this if the origin is the bottomleft corner

    # now you can create a fresh one and populate it with tiles
    logger.info('Started doing the image tiles ')
    im.dzsave(out_dir, layout='google', suffix='.jpg', background=0, skip_blanks=0)
    logger.info('Done. Pyramid of tiles saved at: %s' % out_dir)

    return pixel_dims



if __name__ == "__main__":
    imPath = r'data/background_image/background_image_landscape.tif'
    # split_image(im)

    # # to rotate the image do:
    # im = pyvips.Image.new_from_file(imPath)
    # im = im.rotate(90, interpolate=pyvips.Interpolate.new("nearest"))
    # im.write_to_file(r'data/background_image/background_image_adj_rot.tif')

    tile_maker(10, 'dashboard/img/262144px_landscape_jpg', imPath)




