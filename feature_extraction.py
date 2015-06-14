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
        ecs = df[df['week_diff'] == wn]['event_count']
        if ecs.empty:
            count_by_week.append(0)
        elif ecs.size > 1:
            raise RuntimeError('ecs.size = %s' % ecs.size)
        else:
            count_by_week.append(ecs.values[0])
    ecs = df[df['week_diff'] > __week_span__[-1]]['event_count']
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
        Log['day_diff'] = (base_date - Log['time']).dt.days
        Log['week_diff'] = Log['day_diff'] // 7
        Log['event_count'] = 1

        util.dump(Log, pkl_path)

    Log_counted = Log.groupby(['enrollment_id', 'source_event', 'week_diff'])\
        .agg({'event_count': np.sum}).reset_index()

    logger.debug('datasets prepared')

    D_counted = Enroll.set_index('enrollment_id').ix[enrollment_set]\
        .join(Log_counted.set_index('enrollment_id')).reset_index()
    params = [df for _, df in D_counted.groupby(['enrollment_id'])]

    n_proc = par.cpu_count()
    pool = par.Pool(processes=min(n_proc, len(params)))
    X = np.array(pool.map(__get_counting_feature__, params),
                 dtype=np.float)
    pool.close()
    pool.join()

    logger.debug('source-event pairs counted')

    pkl_path = util.cache_path('D_full_before_%s' % base_date.isoformat())
    if os.path.exists(pkl_path):
        D_full = util.fetch(pkl_path)
    else:
        D_full = Enroll.set_index('enrollment_id')\
            .join(Log.set_index('enrollment_id')).reset_index()

        util.dump(D_full, pkl_path)

    pkl_path = util.cache_path('user_wn_courses_before_%s' %
                               base_date.isoformat())
    if os.path.exists(pkl_path):
        user_wn_courses = util.fetch(pkl_path)
    else:
        user_wn_courses = {}
        for u, df in D_full.groupby(['username']):
            x = []
            for wn in __week_span__:
                x.append(len(df[df['week_diff'] == wn]['course_id'].unique()))
            user_wn_courses[u] = x

        util.dump(user_wn_courses, pkl_path)

    X1 = np.array([user_wn_courses[u] for u in Enroll['username']])

    logger.debug('courses by user counted')

    pkl_path = util.cache_path('course_population_before_%s' %
                               base_date.isoformat())
    if os.path.exists(pkl_path):
        course_population = util.fetch(pkl_path)
    else:
        course_population = {}
        for c, df in D_full.groupby(['course_id']):
            course_population[c] = len(df['username'].unique())

        util.dump(course_population, pkl_path)

    X2 = np.array([course_population[c] for c in Enroll['course_id']])

    logger.debug('course population counted')

    pkl_path = util.cache_path('course_dropout_count_before_%s' %
                               base_date.isoformat())
    if os.path.exists(pkl_path):
        course_dropout_count = util.fetch(pkl_path)
    else:
        course_dropout_count = course_population.copy()
        for c, df in D_full[D_full['day_diff'] < 10].groupby(['course_id']):
            course_dropout_count[c] -= len(df['username'].unique())

        util.dump(course_dropout_count, pkl_path)

    X3 = np.array([course_dropout_count[c] for c in Enroll['course_id']])

    logger.debug('course dropout counted')

    return np.c_[X, X1, X2, X3]
