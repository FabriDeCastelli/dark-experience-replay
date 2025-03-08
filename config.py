""" This script contains the project configuration. """

import os

# PATHS
PROJECT_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_FOLDER_PATH, 'data')

HPARAMS_ROOT = PROJECT_FOLDER_PATH + '/hyperparameters/{}.yaml'