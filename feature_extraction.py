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

base_date: datetime object
Generated features are to predict user dropout bahaviour in the next 10 days
after `base_date'.

Returns
-------
X: numpy ndarray, shape: (num_of_enrollments, num_of_features) or
(num_of_enrollments,)
Rows of feature(s). A function can extract one or more features, which will
be concatenated together.
"""


import numpy as np
from datetime import datetime


def source_event_counter(obj_set, enrollment_set, log_set, base_date):
    """
    Counts the source-event pairs.

    Features
    --------
    """
    def event_counter(df):
        week_span = [0, 1, 2, 3]
        count_by_week = []
        for wn in week_span:
            ecs = df[df['time_diff'] == wn]['event_count']
            if ecs.empty:
                count_by_week.append(0)
            elif ecs.size > 1:
                raise RuntimeError('ecs.size = %s' % ecs.size)
            else:
                count_by_week.append(ecs.values[0])
        ecs = df[df['time_diff'] > week_span[-1]]['event_count']
        if ecs.empty:
            count_by_week.append(0)
        else:
            count_by_week.append(np.average(ecs))
        return count_by_week
    source_event_types = ['browser-access', 'browser-page_close',
                          'browser-problem', 'browser-video',
                          'server-access', 'server-discussion',
                          'server-navigate', 'server-problem', 'server-wiki']
    Enroll = enrollment_set.sort(columns='enrollment_id')
    Log = log_set.copy()
    Log['source_event'] = Log['source'] + '-' + Log['event']
    Log['time_diff'] = (base_date - Log['time']).dt.days // 7
    Log['event_count'] = 1
    Log = Log.groupby(['enrollment_id', 'source_event', 'time_diff'])\
        .agg({'event_count': np.sum}).reset_index()
    X = []
    for eid in Enroll['enrollment_id']:
        features = []
        eq_eid = Log['enrollment_id'] == eid
        for se in source_event_types:
            eq_se = Log['source_event'] == se
            features += event_counter(Log[eq_eid & eq_se])
        X.append(features)
    return np.array(X, dtype=np.float)
