# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import logging

def config_logging(log_level=logging.DEBUG, console=True, filename=None):
    if console:
        logging.basicConfig(format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p',
                    level=log_level)
    elif filename is not None:
        logging.basicConfig(format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p',
                    filename = filename,
                    level=log_level)

