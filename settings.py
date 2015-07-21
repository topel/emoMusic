__author__ = 'thomas'

import logging

def init(log_file_name, withFile):
    global logger
    logger = logging.getLogger(__name__)

    # logger = logging.getLogger(__name__)
    # logging.basicConfig(level=logging.INFO,filename='example.log')

    log_level=logging.INFO
    logger.setLevel(log_level)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.addHandler(ch)
    ch.setFormatter(formatter)

    if withFile:
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)

    # return logger
