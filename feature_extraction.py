#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Extracting feature(s).

Every function should fit the signature described as below:

Parameters
----------
obj_set: pandas DataFrame
Data from object.csv.

enrollment_set: pandas DataFrame

log_set: pandas DataFrame

Returns
-------
X: numpy ndarray, shape: (num_of_enrollments, num_of_features) or
(num_of_enrollments,)
Rows of feature(s). A function can extract one or more features, which will
be concatenated together.
"""


import numpy as np


def source_event_counter(obj_set, enrollment_set, log_set):
    """
    Counts the source-event pairs.
    """
    pass
