#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Extracting feature(s).

Every function should fit the signature described as below:

Parameters
----------
enrollment_set: sorted numpy ndarray
Enrollment ids to generate features.

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
import multiprocessing as par
import logging
import sys
import os

import util


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')


__week_span__ = [0, 1, 2, 3]
__source_event_types__ = ['browser-access', 'browser-page_close',
                          'browser-problem', 'browser-video',
                          'server-access', 'server-discussion',
                          'server-navigate', 'server-problem', 'server-wiki']


def __count_event__(df):
    """get weekly spanned counts of an enrollment_id and source_event"""
    count_by_week = []
    for wn in __week_span__:
        ecs = df[df['time_diff'] == wn]['event_count']
        if ecs.empty:
            count_by_week.append(0)
        elif ecs.size > 1:
            raise RuntimeError('ecs.size = %s' % ecs.size)
        else:
            count_by_week.append(ecs.values[0])
    ecs = df[df['time_diff'] > __week_span__[-1]]['event_count']
    if ecs.empty:
        count_by_week.append(0)
    else:
        count_by_week.append(np.average(ecs))
    return count_by_week


def __get_counting_feature__(df):
    """get source-event counts of an enrollment_id"""
    x = []
    for se in __source_event_types__:
        x += __count_event__(df[df['source_event'] == se])
    return np.array(x)


def source_event_counter(enrollment_set, base_date):
    """
    Counts the source-event pairs.

    Features
    --------
    """
    logger = logging.getLogger('source_event_counter')
    logger.debug('preparing datasets')

    Enroll = util.load_enrollments()

    pkl_path = util.cache_path('Log_all_before_%s' % base_date.isoformat())
    if os.path.exists(pkl_path):
        Log = util.fetch(pkl_path)
    else:
        Log = util.load_logs()
        Log = Log[Log['time'] <= base_date]
        Log['source_event'] = Log['source'] + '-' + Log['event']
        Log['time_diff'] = (base_date - Log['time']).dt.days // 7
        Log['event_count'] = 1
        Log = Log.groupby(['enrollment_id', 'source_event', 'time_diff'])\
            .agg({'event_count': np.sum}).reset_index()

        util.dump(Log, pkl_path)

    logger.debug('datasets prepared')

    D = Enroll.set_index('enrollment_id').ix[enrollment_set]\
        .join(Log.set_index('enrollment_id')).reset_index()
    params = [df for _, df in D.groupby(['enrollment_id'])]

    n_proc = par.cpu_count()
    pool = par.Pool(processes=min(n_proc, len(params)))
    X = np.array(pool.map(__get_counting_feature__, params),
                 dtype=np.float)
    pool.close()
    pool.join()

    logger.debug('source-event pairs counted')

    user_wn_courses = {}
    for u, df in D.groupby(['username']):
        x = []
        for wn in __week_span__:
            x.append(len(df[df['time_diff'] == wn]['course_id'].unique()))
        user_wn_courses[u] = x

    X1 = np.array([user_wn_courses[u] for u in Enroll['username']])

    logger.debug('courses by user counted')

    course_population = {}
    for c, df in D.groupby(['course_id']):
        course_population[c] = len(df['username'].unique())

    X2 = np.array([course_population[c] for c in Enroll['course_id']])

    logger.debug('course population counted')

    return np.c_[X, X1, X2]
