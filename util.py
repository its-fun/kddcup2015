#!/usr/local/bin/python3
# -*- coding:utf-8 -*-


import os
import pandas as pd
import numpy as np
import pickle as pkl


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


@__cache__
def load_enrollment(path):
    """Load enrollment set as pandas DataFrame"""
    return pd.read_csv(path)


@__cache__
def load_object(path):
    """Load object set as pandas DataFrame"""
    return pd.read_csv(path, parse_dates=['start'], na_values=['null'])


@__cache__
def load_val_y(path):
    """Load labels of validation set as numpy ndarray"""
    return np.loadtxt(path, dtype=np.int, delimiter=',')


if __name__ == '__main__':
    import sys
    import glob
    if sys.argv[1] == 'clean':
        cached_files = glob.glob('data/cache/*.pkl')
        cached_files.append('data/object.csv.pkl')
        for path in cached_files:
            os.remove(path)
