#!/usr/local/bin/python3
# -*- coding:utf-8 -*-


import os
import pandas as pd
import numpy as np
import pickle as pkl
import gzip

from path_config import (CACHE_PATH,
                         OBJECT_PATH,
                         TEST_DATASET_PATHS,
                         TRAIN_DATASET_PATHS)


def cache_path(filename):
    if not filename.endswith('.pkl') and not filename.endswith('.pklz'):
        filename += '.pklz'
    return os.path.join(CACHE_PATH, filename)


def dump(obj, path):
    if path.endswith('.pklz') or path.endswith('.pkl.gz'):
        with gzip.open(path, 'wb') as f:
            pkl.dump(obj, f)
    else:
        with open(path, 'wb') as f:
            pkl.dump(obj, f)


def fetch(path):
    if path.endswith('.pklz') or path.endswith('.pkl.gz'):
        with gzip.open(path, 'rb') as f:
            data = pkl.load(f)
    else:
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
    return load_log(TRAIN_DATASET_PATHS['log'])


def load_log_test():
    """Load testing log set as pandas DataFrame"""
    return load_log(TEST_DATASET_PATHS['log'])


def load_logs():
    """Load all log sets as pandas DataFrame"""
    return load_log_train().append(load_log_test(), ignore_index=True)


@__cache__
def load_enrollment(path):
    """Load enrollment set as pandas DataFrame"""
    return pd.read_csv(path)


def load_enrollment_train():
    """Load training enrollment set as pandas DataFrame"""
    return load_enrollment(TRAIN_DATASET_PATHS['enrollment'])


def load_enrollment_test():
    """Load testing enrollment set as pandas DataFrame"""
    return load_enrollment(TEST_DATASET_PATHS['enrollment'])


def load_enrollments():
    """Load all enrollment sets as pandas DataFrame"""
    return load_enrollment_train().append(load_enrollment_test(),
                                          ignore_index=True)


def load_object(path=OBJECT_PATH):
    """Load object set as pandas DataFrame"""
    return pd.read_csv(path, parse_dates=['start'], na_values=['null'])


def load_val_y(path=TRAIN_DATASET_PATHS['truth']):
    """Load enrollment-labels pairs of validation set as numpy ndarray"""
    return np.loadtxt(path, dtype=np.int, delimiter=',')


if __name__ == '__main__':
    import sys
    import glob
    if sys.argv[1] == 'clean':
        cached_files = glob.glob(cache_path('*.pkl'))
        cached_files += glob.glob(cache_path('*.pklz'))
        cached_files += glob.glob(cache_path('*.pkl.gz'))
        for path in cached_files:
            os.remove(path)

    elif sys.argv[1] == 'gzip':
        cached_files = glob.glob(cache_path('*.pkl'))
        for path in cached_files:
            new_path = path + 'z'
            dump(fetch(path), new_path)
            os.remove(path)
