#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Generate datasets for training and validating, and load dataset of testing.
"""


import config


def load_test():
    """
    Load dataset for testing.

    Returns
    -------
    X: numpy ndarray, shape: (num_of_enrollments, num_of_features)
    Rows of features.
    """
    pass


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
    pass
