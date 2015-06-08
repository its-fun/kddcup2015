#! /usr/local/bin/python3
# -*- utf-8 -*-

import pandas as pd
import numpy as np


def load_train(path):
    """Load train data from CSV file."""
    train = pd.read_csv(path, parse_dates=['time'],
                        index_col=['enrollment_id', 'time'])
    train = train.replace('nagivate', 'navigate')
    return train
