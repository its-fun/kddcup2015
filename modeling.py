#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Generate model with respect to dataset.
"""

import util
import dataset


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
    E_in: 0.8119110960575004
    """
    from sklearn.linear_model import LogisticRegressionCV
    X = util.fetch(util.cache_path('train_X_before_2014-08-01_22-00-47.pkl'))
    y = util.fetch(util.cache_path('train_y_before_2014-08-01_22-00-47.pkl'))
    clf = LogisticRegressionCV(cv=10, scoring='roc_auc', n_jobs=-1)
    clf.fit(X, y)
    print(auc_score(clf, X, y))
    to_submission(clf, 'lr_0618_xxx')


def svc_1():
    """
    Submission: svc_1_0619_01.csv
    E_val:
    E_in:
    """
    from sklearn import preprocessing
    from sklearn.svm import LinearSVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFE
    from sklearn.grid_search import RandomizedSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from scipy.stats import expon

    X = util.fetch(util.cache_path('train_X_before_2014-08-01_22-00-47.pkl'))
    y = util.fetch(util.cache_path('train_y_before_2014-08-01_22-00-47.pkl'))

    X_scaled = preprocessing.scale(X)

    cv = StratifiedKFold(y, 10)
    params = {'C': expon()}

    svc = LinearSVC(dual=False)
    rs = RandomizedSearchCV(svc, n_iter=50, scoring='roc_auc', n_jobs=-1,
                            cv=cv, param_distributions=params)
    rs.fit(X_scaled, y)
    print('Grid scores: %s' % rs.grid_scores_)
    print('Best score: %s' % rs.best_score_)
    print('Best params: %s' % rs.best_params_)

    rfe = RFE(estimator=rs.best_estimator_, step=1,
              n_features_to_select=21, scoring='roc_auc')
    rfe.fit(X_scaled, y)

    X_new = preprocessing.scale(rfe.transform(X_scaled))
    svc = LinearSVC(dual=False)
    rs = RandomizedSearchCV(svc, n_iter=50, scoring='roc_auc', n_jobs=-1,
                            cv=cv, param_distributions=params)
    rs.fit(X_new, y)
    print('Grid scores: %s' % rs.grid_scores_)
    print('Best score (E_val): %s' % rs.best_score_)
    print('Best params: %s' % rs.best_params_)

    svc = rs.best_estimator_
    isotonic = CalibratedClassifierCV(svc, cv=cv, method='isotonic')
    sigmoid = CalibratedClassifierCV(svc, cv=cv, method='sigmoid')
    isotonic.fit(X_new, y)
    sigmoid.fit(X_new, y)
    print('E_in (isotonic): %f' % auc_score(isotonic, X_new, y))
    print('E_in (sigmoid): %f' % auc_score(sigmoid, X_new, y))

    to_submission(isotonic, 'svc_1_0619_01')


if __name__ == '__main__':
    import sys
    from inspect import isfunction
    variables = locals()
    if len(sys.argv) > 1:
        for fn in sys.argv[1:]:
            if fn not in variables or not isfunction(variables[fn]):
                print('function %s not found' % repr(fn))
            variables[fn]()
