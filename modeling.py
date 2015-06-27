#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Generate model with respect to dataset.
"""

import logging
import sys

import util
import dataset

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
logger = logging.getLogger('modeling')


def auc_score(clf, X, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, clf.predict_proba(X)[:, 1])


def to_submission(clf, filename):
    path = filename
    if not path.startswith('submission/'):
        path = 'submission/' + path
    if not path.endswith('.csv'):
        path += '.not-submitted.csv'
    Enroll_test = util.load_enrollment_test()['enrollment_id']
    X_test = dataset.load_test()
    y_test = clf.predict_proba(X_test)[:, 1]
    lines = ['%d,%f\n' % l for l in zip(Enroll_test, y_test)]
    with open(path, 'w') as f:
        f.writelines(lines)


def lr():
    """
    Submission: lr_0618.csv
    E_val: <missing>
    E_in: <missing>
    E_out: 0.8119110960575004
    """
    from sklearn.linear_model import LogisticRegressionCV
    X = util.fetch(util.cache_path('train_X_before_2014-08-01_22-00-47'))
    y = util.fetch(util.cache_path('train_y_before_2014-08-01_22-00-47'))
    clf = LogisticRegressionCV(cv=10, scoring='roc_auc', n_jobs=-1)
    clf.fit(X, y)
    print(auc_score(clf, X, y))
    to_submission(clf, 'lr_0618_xxx')


def lr_with_scale():
    """
    Submission: lr_with_scale_0620_04.csv
    E_val: <missing>
    E_in: 0.857351105162
    E_out: 0.854097855439904
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = util.fetch(util.cache_path('train_X_before_2014-08-01_22-00-47'))
    y = util.fetch(util.cache_path('train_y_before_2014-08-01_22-00-47'))

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    clf = LogisticRegressionCV(cv=10, scoring='roc_auc', n_jobs=-1)
    clf.fit(X_scaled, y)
    print(auc_score(clf, X_scaled, y))
    to_submission(Pipeline([('scale_raw', raw_scaler),
                            ('lr', clf)]), 'lr_with_scale_0620_04')


def lr_with_fs():
    """
    Submission: lr_with_fs_0620_02.csv
    E_val: <missing>
    E_in: 0.856252488379
    E_out: 0.8552577388980213
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = util.fetch(util.cache_path('train_X_before_2014-08-01_22-00-47'))
    y = util.fetch(util.cache_path('train_y_before_2014-08-01_22-00-47'))

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    rfe = util.fetch(util.cache_path('feature_selection.RFE.21'))

    X_pruned = rfe.transform(X_scaled)

    new_scaler = StandardScaler()
    new_scaler.fit(X_pruned)
    X_new = new_scaler.transform(X_pruned)

    clf = LogisticRegressionCV(cv=10, scoring='roc_auc', n_jobs=-1)
    clf.fit(X_new, y)
    print(auc_score(clf, X_new, y))
    to_submission(Pipeline([('scale_raw', raw_scaler),
                            ('rfe', rfe),
                            ('scale_new', new_scaler),
                            ('lr', clf)]), 'lr_with_fs_0620_02')


def svc_1():
    """
    Submission: svc_1_0620_01.csv
    E_val: 0.866856950449
    E_in: 0.855948
    E_out: 0.8546898189645258
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFE
    from sklearn.grid_search import RandomizedSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import expon

    logger.debug('svc_1')

    X = util.fetch(util.cache_path('train_X_before_2014-08-01_22-00-47'))
    y = util.fetch(util.cache_path('train_y_before_2014-08-01_22-00-47'))

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    rfe = RFE(estimator=LogisticRegression(class_weight='auto'), step=1,
              n_features_to_select=21)
    rfe.fit(X_scaled, y)
    util.dump(rfe, util.cache_path('feature_selection.RFE.21'))

    X_pruned = rfe.transform(X_scaled)

    logger.debug('Features selected.')

    new_scaler = StandardScaler()
    new_scaler.fit(X_pruned)
    X_new = new_scaler.transform(X_pruned)

    svc = LinearSVC(dual=False, class_weight='auto')
    rs = RandomizedSearchCV(svc, n_iter=50, scoring='roc_auc', n_jobs=-1,
                            cv=StratifiedKFold(y, 5),
                            param_distributions={'C': expon()})
    rs.fit(X_new, y)

    logger.debug('Got best SVC.')
    logger.debug('Grid scores: %s', rs.grid_scores_)
    logger.debug('Best score (E_val): %s', rs.best_score_)
    logger.debug('Best params: %s', rs.best_params_)

    svc = rs.best_estimator_
    util.dump(svc, util.cache_path('new_data.SVC'))

    isotonic = CalibratedClassifierCV(svc, cv=StratifiedKFold(y, 5),
                                      method='isotonic')
    isotonic.fit(X_new, y)
    util.dump(isotonic,
              util.cache_path('new_data.CalibratedClassifierCV.isotonic'))

    logger.debug('Got best isotonic CalibratedClassifier.')
    logger.debug('E_in (isotonic): %f', auc_score(isotonic, X_new, y))

    to_submission(Pipeline([('scale_raw', raw_scaler),
                            ('rfe', rfe),
                            ('scale_new', new_scaler),
                            ('svc', isotonic)]), 'svc_1_0620_01')


def sgd():
    """
    Submission: sgd_0620_03.csv
    E_val: 0.863628
    E_in: 0.854373
    E_out:
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import StratifiedKFold

    X = util.fetch(util.cache_path('train_X_before_2014-08-01_22-00-47'))
    y = util.fetch(util.cache_path('train_y_before_2014-08-01_22-00-47'))

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    rfe = util.fetch(util.cache_path('feature_selection.RFE.21'))

    X_pruned = rfe.transform(X_scaled)

    new_scaler = StandardScaler()
    new_scaler.fit(X_pruned)
    X_new = new_scaler.transform(X_pruned)

    sgd = SGDClassifier(n_iter=50, n_jobs=-1)
    params = {
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge',
                 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive',
                 'squared_epsilon_insensitive']
    }
    grid = GridSearchCV(sgd, param_grid=params, cv=StratifiedKFold(y, 5),
                        scoring='roc_auc', n_jobs=-1)
    grid.fit(X_new, y)

    logger.debug('Best score (E_val): %f', grid.best_score_)

    sgd = grid.best_estimator_

    logger.debug('E_in: %f', auc_score(sgd, X_new, y))
    to_submission(Pipeline([('scale_raw', raw_scaler),
                            ('rfe', rfe),
                            ('scale_new', new_scaler),
                            ('sgd', sgd)]), 'sgd_0620_03')


def dt():
    """
    Submission: dt_0620_05.csv
    E_val: 0.820972
    E_in: 0.835177
    E_out:
    Comment: {'max_depth': 5}
    """
    from sklearn.tree import DecisionTreeClassifier, export_graphviz

    X = util.fetch(util.cache_path('train_X_before_2014-08-01_22-00-47'))
    y = util.fetch(util.cache_path('train_y_before_2014-08-01_22-00-47'))

    dt = DecisionTreeClassifier(max_depth=5, class_weight='auto')
    dt.fit(X, y)

    export_graphviz(dt, 'tree.dot')

    logger.debug('E_in: %f', auc_score(dt, X, y))
    to_submission(dt, 'dt_0620_05')


if __name__ == '__main__':
    from inspect import isfunction
    variables = locals()
    if len(sys.argv) > 1:
        for fn in sys.argv[1:]:
            if fn not in variables or not isfunction(variables[fn]):
                print('function %s not found' % repr(fn))
            variables[fn]()
