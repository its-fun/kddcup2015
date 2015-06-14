#! /usr/local/bin/python3
# -*- utf-8 -*-

"""
Configurations.

+ modeling:
  - feature extraction methods
  - training models
  - ensemble method
  - score function
+ datasets paths
"""


from os.path import abspath

from feature_extraction import source_event_counter


MODELING = {
    'features': [source_event_counter],
    'training_models': [],
    'ensemble_method': None,
    'score_func': 'auc'
}


# DATASETS PATHS

CACHE_PATH = abspath('data/cache/')

OBJECT_PATH = abspath('data/object.csv')

TEST_DATASET_PATHS = {
    'enrollment': abspath('data/test/enrollment_test.csv'),
    'log': abspath('data/test/log_test.csv')
}

TRAIN_DATASET_PATHS = {
    'enrollment': abspath('data/train/enrollment_train.csv'),
    'log': abspath('data/train/log_train.csv'),
    'truth': abspath('data/train/truth_train.csv')
}
