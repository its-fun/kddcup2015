#!/usr/local/bin/python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

import pickle as pkl


class Extractor:
    """A util class to extract features from KDD CUP Data"""
    def __init__(self, train):
        self.train = train

    def fgen_events(self):
        index = self.train.index.droplevel(1)
        train = self.train.set_index(index)
        events = train['event'].groupby(level=0).value_counts()
        return events.unstack().fillna(0)

    def fgen_breaks(self, intervning=10):
        from datetime import timedelta

        def break_count(series, day):
            if len(series) < 2:
                return -1
            cnt = 0
            for i in range(len(series) - 2):
                if series.iat[i + 1] - series.iat[i] >= timedelta(days=day):
                    cnt += 1
            return cnt

        times = self.train.reset_index('time')
        grouped = times.groupby(level=0)
        breaks = grouped.apply(lambda x: break_count(x['time'], intervning))
        breaks.name = 'break_times'
        return pd.DataFrame(breaks)

    def fgen_time_distribution(self):
        grouped = self.train.groupby('object')

        def date_count(obj_group):
            return obj_group.groupby(lambda x: x.date,
                                     level=1)['object'].count()

        return grouped.apply(date_count)


def save(path, obj):
    f = open(path, 'wb')
    pkl.dump(obj, f)
    f.close()
    print("Success saved to %s." % path)


def load(path):
    f = open(path, 'rb')
    obj = pkl.load(f)
    f.close()
    return obj
