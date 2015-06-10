#!/usr/local/bin/python3
# -*- coding:utf-8 -*-


import pandas as pd
import numpy as np


def load_log(path):
    """Load log set as pandas DataFrame"""
    log_set = pd.read_csv(path, parse_dates=['time'])
    log_set['event'] = log_set['event'].replace('nagivate', 'navigate')
    return log_set


def load_enrollment(path):
    """Load enrollment set as pandas DataFrame"""
    return pd.read_csv(path)


def load_object(path):
    """Load object set as pandas DataFrame"""
    return pd.read_csv(path, parse_dates=['start'])


def load_val_y(path):
    """Load labels of validation set as numpy ndarray"""
    return np.loadtxt(path, dtype=np.int, delimiter=',')
