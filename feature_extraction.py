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
import pandas as pd
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
    X_pkl_path = util.cache_path('source_event_counter_before_%s' %
                                 base_date.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(X_pkl_path):
        return util.fetch(X_pkl_path)

    logger = logging.getLogger('source_event_counter')
    logger.debug('preparing datasets')

    Enroll_all = util.load_enrollments()

    pkl_path = util.cache_path('Log_all_before_%s' %
                               base_date.strftime('%Y-%m-%d_%H-%M-%S'))
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

    Enroll = Enroll_all.set_index('enrollment_id').ix[enrollment_set]\
        .reset_index()

    n_proc = par.cpu_count()

    pkl_path = util.cache_path('event_count_by_eid_before_%s' %
                               base_date.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(pkl_path):
        event_count_by_eid = util.fetch(pkl_path)
    else:
        params = []
        eids = []
        for eid, df in pd.merge(Enroll_all, Log_counted, on=['enrollment_id'])\
                .groupby(['enrollment_id']):
            params.append(df)
            eids.append(eid)
        pool = par.Pool(processes=min(n_proc, len(params)))
        event_count_by_eid = dict(zip(eids,
                                      pool.map(__get_counting_feature__,
                                               params)))
        pool.close()
        pool.join()

        util.dump(event_count_by_eid, pkl_path)

    X0 = np.array([event_count_by_eid[i] for i in Enroll['enrollment_id']])

    logger.debug('source-event pairs counted, has nan: %s, shape: %s',
                 np.any(np.isnan(X0)), repr(X0.shape))

    pkl_path = util.cache_path('D_full_before_%s' %
                               base_date.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(pkl_path):
        D_full = util.fetch(pkl_path)
    else:
        D_full = pd.merge(Enroll_all, Log, on=['enrollment_id'])

        util.dump(D_full, pkl_path)

    pkl_path = util.cache_path('user_wn_courses_before_%s' %
                               base_date.strftime('%Y-%m-%d_%H-%M-%S'))
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

    logger.debug('courses by user counted, has nan: %s, shape: %s',
                 np.any(np.isnan(X1)), repr(X1.shape))

    pkl_path = util.cache_path('course_population_before_%s' %
                               base_date.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(pkl_path):
        course_population = util.fetch(pkl_path)
    else:
        course_population = {}
        for c, df in D_full.groupby(['course_id']):
            course_population[c] = len(df['username'].unique())

        util.dump(course_population, pkl_path)

    X2 = np.array([course_population.get(c, 0) for c in Enroll['course_id']])

    logger.debug('course population counted, has nan: %s, shape: %s',
                 np.any(np.isnan(X2)), repr(X2.shape))

    pkl_path = util.cache_path('course_dropout_count_before_%s' %
                               base_date.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(pkl_path):
        course_dropout_count = util.fetch(pkl_path)
    else:
        course_dropout_count = course_population.copy()
        for c, df in D_full[D_full['day_diff'] < 10].groupby(['course_id']):
            course_dropout_count[c] -= len(df['username'].unique())

        util.dump(course_dropout_count, pkl_path)

    X3 = np.array([course_dropout_count.get(c, 0)
                   for c in Enroll['course_id']])

    logger.debug('course dropout counted, has nan: %s, shape: %s',
                 np.any(np.isnan(X3)), repr(X3.shape))

    pkl_path = util.cache_path('user_ops_count_before_%s' %
                               base_date.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(pkl_path):
        user_ops_count = util.fetch(pkl_path)
    else:
        user_ops_on_all_courses = D_full.groupby(
            ['username', 'source_event', 'week_diff'])\
            .agg({'event_count': np.sum}).reset_index()
        params = []
        users = []
        for u, df in user_ops_on_all_courses.groupby(['username']):
            params.append(df)
            users.append(u)
        pool = par.Pool(processes=min(n_proc, len(params)))
        user_ops_count = dict(zip(users,
                                  pool.map(__get_counting_feature__, params)))
        pool.close()
        pool.join()

        util.dump(user_ops_count, pkl_path)

    X4 = X0 / [user_ops_count[u] for u in Enroll['username']]
    X4[np.isnan(X4)] = 0

    logger.debug('ratio of user ops on all courses, has nan: %s, shape: %s',
                 np.any(np.isnan(X4)), repr(X4.shape))

    pkl_path = util.cache_path('course_ops_count_before_%s' %
                               base_date.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(pkl_path):
        course_ops_count = util.fetch(pkl_path)
    else:
        course_ops_of_all_users = D_full.groupby(
            ['course_id', 'source_event', 'week_diff'])\
            .agg({'event_count': np.sum}).reset_index()
        params = []
        courses = []
        for c, df in course_ops_of_all_users.groupby(['course_id']):
            params.append(df)
            courses.append(c)
        pool = par.Pool(processes=min(n_proc, len(params)))
        course_ops_count = dict(zip(courses,
                                    pool.map(__get_counting_feature__,
                                             params)))
        pool.close()
        pool.join()

        util.dump(course_ops_count, pkl_path)

    X5 = X0 / [course_ops_count[c] for c in Enroll['course_id']]
    X5[np.isnan(X5)] = 0

    logger.debug('ratio of courses ops of all users, has nan: %s, shape: %s',
                 np.any(np.isnan(X5)), repr(X5.shape))

    X6 = np.array([course_dropout_count.get(c, 0) / course_population.get(c, 1)
                   for c in Enroll['course_id']])

    logger.debug('dropout ratio of courses, has nan: %s, shape: %s',
                 np.any(np.isnan(X6)), repr(X6.shape))

    Obj = util.load_object()
    Obj = Obj[Obj['start'] <= base_date]
    course_time = {}
    for c, df in Obj.groupby(['course_id']):
        start_time = np.min(df['start'])
        update_time = np.max(df['start'])
        course_time[c] = [
            (base_date - start_time).days,
            (base_date - update_time).days]

    avg_start_days = np.average([t[0] for _, t in course_time.items()])
    avg_update_days = np.average([t[1] for _, t in course_time.items()])
    default_case = [avg_start_days, avg_update_days]

    X7 = np.array([course_time.get(c, default_case)[0]
                   for c in Enroll['course_id']])

    logger.debug('days from course first update, has nan: %s, shape: %s',
                 np.any(np.isnan(X7)), repr(X7.shape))

    X8 = np.array([course_time.get(c, default_case)[1]
                   for c in Enroll['course_id']])

    logger.debug('days from course last update, has nan: %s, shape: %s',
                 np.any(np.isnan(X8)), repr(X8.shape))

    user_ops_time = pd.merge(Enroll, Log, how='left', on=['enrollment_id'])\
        .groupby(['enrollment_id']).agg({'day_diff': [np.min, np.max]})\
        .fillna(0)
    X9 = np.array(user_ops_time['day_diff']['amin'])

    logger.debug('days from user last op, has nan: %s, shape: %s',
                 np.any(np.isnan(X9)), repr(X9.shape))

    X10 = np.array(user_ops_time['day_diff']['amax'])

    logger.debug('days from user first op, has nan: %s, shape: %s',
                 np.any(np.isnan(X10)), repr(X10.shape))

    X11 = X7 - X10

    logger.debug('days from course first update to user first op, has nan: %s'
                 ', shape: %s', np.any(np.isnan(X11)), repr(X11.shape))

    X = np.c_[X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11]
    util.dump(X, X_pkl_path)

    return X


def __get_dropout_feature__(df):
    """get dropout count of an enrollment_id"""
    df = df.sort('time')['time']
    dd = np.array((df[1:].reset_index() - df[:-1].reset_index())['time']
                  .dt.days)
    return np.sum(dd >= 10)


def dropout_history(enrollment_set, base_date):
    X_pkl_path = util.cache_path('dropout_history_before_%s' %
                                 base_date.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(X_pkl_path):
        return util.fetch(X_pkl_path)

    logger = logging.getLogger('dropout_history')

    n_proc = par.cpu_count()

    pkl_path = util.cache_path('Dropout_count_before_%s' %
                               base_date.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(pkl_path):
        logger.debug('load from cache')

        Dropout_count = util.fetch(pkl_path)
    else:
        logger.debug('preparing datasets')

        Enroll_all = util.load_enrollments()

        Log = util.load_logs()
        Log = Log[Log['time'] <= base_date]
        Log_enroll_ids = pd.DataFrame(np.unique(Log['enrollment_id']),
                                      columns=['enrollment_id'])

        logger.debug('datasets prepared')

        params = []
        enroll_ids = []
        for i, df in Log.groupby(['enrollment_id']):
            params.append(df)
            enroll_ids.append(i)
        pool = par.Pool(processes=min(n_proc, len(params)))
        enroll_dropout_count = dict(zip(enroll_ids,
                                        pool.map(__get_dropout_feature__,
                                                 params)))
        pool.close()
        pool.join()

        enroll_dropout_count = pd.Series(enroll_dropout_count,
                                         name='dropout_count')
        enroll_dropout_count.index.name = 'enrollment_id'
        enroll_dropout_count = enroll_dropout_count.reset_index()

        Enroll_counted = pd.merge(Enroll_all, enroll_dropout_count, how='left',
                                  on=['enrollment_id'])
        Dropout_count = pd.merge(Log_enroll_ids, Enroll_counted, how='left',
                                 on=['enrollment_id'])

        util.dump(Dropout_count, pkl_path)

    Dgb = Dropout_count.groupby('username')
    total_dropout = Dgb.agg({'dropout_count': np.sum}).reset_index().rename(
        columns={'dropout_count': 'total_dropout'})
    avg_dropout = Dgb.agg({'dropout_count': np.average}).reset_index().rename(
        columns={'dropout_count': 'avg_dropout'})
    drop_courses = Dgb.agg(
        {'dropout_count': lambda x: len([i for i in x if i > 0])})\
        .reset_index().rename(columns={'dropout_count': 'drop_courses'})
    course_count = Dgb.agg({'dropout_count': len}).reset_index().rename(
        columns={'dropout_count': 'course_count'})

    Dropout_count = pd.merge(Dropout_count, total_dropout, how='left',
                             on=['username'])
    Dropout_count = pd.merge(Dropout_count, avg_dropout, how='left',
                             on=['username'])
    Dropout_count = pd.merge(Dropout_count, drop_courses, how='left',
                             on=['username'])
    Dropout_count = pd.merge(Dropout_count, course_count, how='left',
                             on=['username'])

    Dropout_count['drop_ratio'] = (Dropout_count['drop_courses'] /
                                   Dropout_count['course_count'])

    Enroll = Enroll_all.set_index('enrollment_id').ix[enrollment_set]\
        .reset_index()

    X = pd.merge(Enroll, Dropout_count, how='left', on=['enrollment_id'])\
        .as_matrix(columns=['dropout_count', 'total_dropout', 'avg_dropout',
                            'drop_courses', 'course_count', 'drop_ratio'])

    logger.debug('dropout history, has nan: %s, shape: %s',
                 np.any(np.isnan(X)), repr(X.shape))

    util.dump(X, X_pkl_path)
    return X
