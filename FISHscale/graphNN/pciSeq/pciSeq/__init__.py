import subprocess
import sys
import os
import logging

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s:%(levelname)s:%(message)s"
# )


def confirm_prompt(question):
    reply = None
    while reply not in ("", "y", "n"):
        reply = input(f"{question} (y/n): ").lower()
    return (reply in ("", "y"))


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_libvips():
    subprocess.check_call("apt-get update", shell=True)
    subprocess.check_call("apt-get install", shell=True)
    subprocess.check_call(['apt-get', 'install', '-y', 'libvips'],
               stdout=open(os.devnull, 'wb'), stderr=subprocess.STDOUT)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvips"])

#
# def check_libvips(logger):
#     confirm = confirm_prompt('Install libvips?')
#     if confirm:
#       install_libvips()
#     else:
#       print('>>>> libvips not installed')
#     return confirm


def check_libvips():
    try:
        import pyvips
        status = True
    except OSError:
        status = False
    except Exception as err:
        raise
    return status


from pciSeq.src._version import __version__
from pciSeq.app import fit
from pciSeq.app import cell_type
from pciSeq.src.preprocess.spot_labels import stage_data
import pciSeq.src.cell_call.utils as utils

#if check_libvips():
#    from pciSeq.src.viewer.stage_image import tile_maker
# else:
#     logger.warning('>>>> libvips is not installed. Please see https://www.libvips.org/install.html <<<<')
#     logger.warning('>>>> This is required if you want to do your own viewer. <<<<')
#     logger.warning('>>>> and visualise your results after cell typing. <<<<')
#     logger.warning('>>>> To do cell typing, libvips can be ignored, it is *not* necessary.  <<<<')
#     logger.warning('>>>> LIBVIPS_ENABLED is %s.  <<<<' % check_libvips())



