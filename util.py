#!/usr/local/bin/python3
# -*- coding:utf-8 -*-


import os
import pandas as pd
import numpy as np
import pickle as pkl

import config


def cache_path(filename):
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    return os.path.join(config.CACHE_PATH, filename)


def dump(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)


def fetch(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


def __cache__(func):
    def cached_func(path):
        pkl_path = path + '.pkl'
        if os.path.exists(pkl_path):
            data = fetch(pkl_path)
        else:
            data = func(path)
            dump(data, pkl_path)
        return data
    return cached_func


@__cache__
def load_log(path):
    """Load log set as pandas DataFrame"""
    log_set = pd.read_csv(path, parse_dates=['time'])
    log_set['event'] = log_set['event'].replace('nagivate', 'navigate')
    return log_set


def load_log_train():
    """Load training log set as pandas DataFrame"""
    return load_log(config.TRAIN_DATASET_PATHS['log'])


def load_log_test():
    """Load testing log set as pandas DataFrame"""
    return load_log(config.TEST_DATASET_PATHS['log'])


def load_logs():
    """Load all log sets as pandas DataFrame"""
    return load_log_train().append(load_log_test(), ignore_index=True)


@__cache__
def load_enrollment(path):
    """Load enrollment set as pandas DataFrame"""
    return pd.read_csv(path)


def load_enrollment_train():
    """Load training enrollment set as pandas DataFrame"""
    return load_enrollment(config.TRAIN_DATASET_PATHS['enrollment'])


def load_enrollment_test():
    """Load testing enrollment set as pandas DataFrame"""
    return load_enrollment(config.TEST_DATASET_PATHS['enrollment'])


def load_enrollments():
    """Load all enrollment sets as pandas DataFrame"""
    return load_enrollment_train().append(load_enrollment_test(),
                                          ignore_index=True)


@__cache__
def load_object(path=config.OBJECT_PATH):
    """Load object set as pandas DataFrame"""
    return pd.read_csv(path, parse_dates=['start'], na_values=['null'])


@__cache__
def load_val_y(path=config.TRAIN_DATASET_PATHS['truth']):
    """Load enrollment-labels pairs of validation set as numpy ndarray"""
    return np.loadtxt(path, dtype=np.int, delimiter=',')


if __name__ == '__main__':
    import sys
    import glob
    if sys.argv[1] == 'clean':
        cached_files = glob.glob('data/cache/*.pkl')
        for path in cached_files:
            os.remove(path)
