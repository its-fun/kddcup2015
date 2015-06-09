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


MODELING = {
    'features': [],
    'training_models': [],
    'ensemble_method': None,
    'score_func': 'auc'
}


# DATASETS PATHS

COMMON_PATHS = {
    'object.csv': abspath('data/object.csv')
}

TEST_DATASET_PATHS = {
    'enrollment_test.csv': abspath('data/test/enrollment_test.csv'),
    'log_test.csv': abspath('data/test/log_test.csv')
}

TRAIN_DATASET_PATHS = {
    'enrollment_test.csv': abspath('data/train/enrollment_train.csv'),
    'log_test.csv': abspath('data/train/log_train.csv'),
    'truth_train.csv': abspath('data/train/truth_train.csv')
}
