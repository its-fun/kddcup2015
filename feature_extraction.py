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
import logging
import sys
import os


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')


def source_event_counter(obj_set, enrollment_set, log_set, base_date):
    """
    Counts the source-event pairs.

    Features
    --------
    """
    log = logging.getLogger('source_event_counter')
    log.debug('preparing dataset')

    week_span = [0, 1, 2, 3]
    source_event_types = ['browser-access', 'browser-page_close',
                          'browser-problem', 'browser-video',
                          'server-access', 'server-discussion',
                          'server-navigate', 'server-problem', 'server-wiki']

    def count_event(df):
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

    def get_counting_feature(df):
        x = []
        for se in source_event_types:
            x += count_event(df[df['source_event'] == se])
        return np.array(x)

    Enroll = enrollment_set.set_index('enrollment_id')

    log.debug('Enroll prepared')

    pkl_path = 'data/cache/Log.pkl'
    if os.path.exists(pkl_path):
        Log = util.fetch(pkl_path)
    else:
        Log = log_set.copy()
        Log['source_event'] = Log['source'] + '-' + Log['event']
        Log['time_diff'] = (base_date - Log['time']).dt.days // 7
        Log['event_count'] = 1
        Log = Log.groupby(['enrollment_id', 'source_event', 'time_diff'])\
            .agg({'event_count': np.sum}).reset_index()
        util.dump(Log, pkl_path)

    log.debug('dataset prepared')

    log.debug('counting source-event pairs')
    D = Enroll.join(Log.set_index('enrollment_id')).reset_index()
    log.debug('datasets joined')

    X = []
    for eid, df in D.groupby(['enrollment_id']):
        X.append(get_counting_feature(df))
        if len(X) % 100 == 0:
            log.debug('len(X): %d', len(X))
    return np.array(X, dtype=np.float)


import util
print(source_event_counter(util.load_object('data/object.csv'),
                     util.load_enrollment('data/train/enrollment_train.csv'),
                     util.load_log('data/train/log_train.csv'),
                     datetime(2014, 8, 2)).shape)
