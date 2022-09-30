import numpy as np
import logging

# create a default logger
logger = logging.getLogger('pciSeq')


def attach_to_log(level=logging.DEBUG,
                  handler=None,
                  loggers=None,
                  colors=True,
                  capture_warnings=True,
                  blacklist=None):
    """
    Attach a stream handler to all loggers.
    Parameters
    ------------
    level : enum
      Logging level, like logging.INFO
    handler : None or logging.Handler
      Handler to attach
    loggers : None or (n,) logging.Logger
      If None, will try to attach to all available
    colors : bool
      If True try to use colorlog formatter
    blacklist : (n,) str
      Names of loggers NOT to attach to
    """

    # default blacklist includes ipython debugging stuff
    if blacklist is None:
        blacklist = ['TerminalIPythonApp',
                     'PYREADLINE',
                     'pyembree',
                     'shapely',
                     'matplotlib',
                     'numba',
                     'parso']

    # make sure we log warnings from the warnings module
    logging.captureWarnings(capture_warnings)

    # create a basic formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
        "%Y-%m-%d %H:%M:%S")
    if colors:
        try:
            from colorlog import ColoredFormatter
            formatter = ColoredFormatter(
                ("%(log_color)s%(levelname)-8s%(reset)s " +
                 "%(filename)17s:%(lineno)-4s  %(blue)4s%(message)s"),
                datefmt=None,
                reset=True,
                log_colors={'DEBUG': 'cyan',
                            'INFO': 'green',
                            'WARNING': 'yellow',
                            'ERROR': 'red',
                            'CRITICAL': 'red'})
        except ImportError:
            pass

    # if no handler was passed use a StreamHandler
    if handler is None:
        handler = logging.StreamHandler()

    # add the formatters and set the level
    handler.setFormatter(formatter)
    handler.setLevel(level)

    # if nothing passed use all available loggers
    if loggers is None:
        # de-duplicate loggers using a set
        loggers = set(logging.Logger.manager.loggerDict.values())
    # add the warnings logging
    loggers.add(logging.getLogger('py.warnings'))

    # disable pyembree warnings
    logging.getLogger('pyembree').disabled = True

    # loop through all available loggers
    for logger in loggers:
        # skip loggers on the blacklist
        if (logger.__class__.__name__ != 'Logger' or
                any(logger.name.startswith(b) for b in blacklist)):
            continue
        logger.addHandler(handler)
        logger.setLevel(level)

    # set nicer numpy print options
    np.set_printoptions(precision=5, suppress=True)