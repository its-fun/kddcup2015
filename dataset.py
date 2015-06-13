#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Generate datasets for training and validating, and load dataset of testing.
"""


import os
import numpy as np
from datetime import timedelta

import config
import util


def load_test():
    """
    Load dataset for testing.

    Returns
    -------
    X: numpy ndarray, shape: (num_of_enrollments, num_of_features)
    Rows of features.
    """
    pkl_path = util.cache_path('test_X')
    if os.path.exists(pkl_path):
        X = util.fetch(pkl_path)
    else:
        Object = util.load_object(config.COMMON_PATHS['object.csv'])
        Enroll_test = util.load_enrollment(
            config.TEST_DATASET_PATHS['enrollment_test.csv'])
        Log_test = util.load_log(config.TEST_DATASET_PATHS['log_test.csv'])
        Log_train = util.load_log(config.TRAIN_DATASET_PATHS['log_train.csv'])
        Log = Log_train.append(Log_test, ignore_index=True)
        base_date = Log['time'].max().to_datetime()
        X = None
        for f in config.MODELING['features']:
            X_ = f(Object, Enroll_test, Log, base_date)
            if X is None:
                X = X_
            else:
                X = np.c_[X, X_]
        util.dump(X, pkl_path)
    return X


def load_train(until=None):
    """
    Load dataset for training and validating.

    *NOTE*  If you need a validating set, you SHOULD split from training set
    by yourself.

    Parameters
    ----------
    until: datetime object, or None (by default)
    Logs no later than `until' will be transformed to features, and others will
    be transformed to labels (for each enrollment, if there is no log after
    `until', the label be 1 - dropout, 0 otherwise).

    Returns
    -------
    X: numpy ndarray, shape: (num_of_enrollments, num_of_features)
    Rows of features.

    y: numpy ndarray, shape: (num_of_enrollments,)
    Vector of labels.
    """
    Object = util.load_object(config.COMMON_PATHS['object.csv'])
    Enroll_train = util.load_enrollment(
        config.TRAIN_DATASET_PATHS['enrollment_train.csv'])
    Log_test = util.load_log(config.TEST_DATASET_PATHS['log_test.csv'])
    Log_train = util.load_log(config.TRAIN_DATASET_PATHS['log_train.csv'])
    Log = Log_train.append(Log_test, ignore_index=True)
    base_date = Log['time'].max().to_datetime()
    X = None
    Dw = timedelta(days=7)
    while not Log.empty:
        X_temp = None
        for f in config.MODELING['features']:
            X_ = f(Object, Enroll_train, Log, base_date)
            if X_temp is None:
                X_temp = X_
            else:
                X_temp = np.c_[X_temp, X_]
        # TODO: check X_temp and update to X; update y
        base_date -= Dw
        Log = Log[Log['time'] <= base_date]
    return X, y
