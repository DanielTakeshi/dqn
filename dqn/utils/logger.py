import os
import logging


_logging_setup_table = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
}


def setup_logger(dir_path, filename="root", level="info"):
    # Config the logging
    logging.basicConfig(
        level=_logging_setup_table[level],
        format='%(asctime)s %(name)-7s %(levelname)-6s %(message)s',
        datefmt='%m-%d %H:%M:%S',
        filename=os.path.join(dir_path, '{0}.log'.format(filename)),
        filemode='w')
    # Define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(_logging_setup_table[level])
    # Set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s '
                                  '%(message)s', datefmt='%m-%d %H:%M:%S')
    # Tell the handler to use this format
    console.setFormatter(formatter)
    # Add the handler to the root logger
    logging.getLogger('').addHandler(console)
