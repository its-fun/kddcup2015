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
        path += '.csv'
    Enroll_test = util.load_enrollment_test()['enrollment_id']
    X_test = dataset.load_test()
    y_test = clf.predict_proba(X_test)[:, 1]
    lines = ['%d, %f\n' % l for l in zip(Enroll_test, y_test)]
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
    import matplotlib.pyplot as plt
    from sklearn.svm import LinearSVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.grid_search import RandomizedSearchCV
    from scipy.stats import expon

    X = util.fetch(util.cache_path('train_X_before_2014-08-01_22-00-47.pkl'))
    y = util.fetch(util.cache_path('train_y_before_2014-08-01_22-00-47.pkl'))

    svc = LinearSVC(dual=False)
    rs = RandomizedSearchCV(svc, n_iter=50, scoring='roc_auc', n_jobs=-1,
                            param_distributions={'C': expon()})
    rs.fit(X, y)
    print('Grid scores: %s' % rs.grid_scores_)
    print('Best score: %s' % rs.best_score_)
    print('Best params: %s' % rs.best_params_)

    rfecv = RFECV(estimator=rs.best_estimator_, step=1,
                  cv=StratifiedKFold(y, 5), scoring='roc_auc')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    X_new = rfecv.transform(X)
    svc = LinearSVC(dual=False)
    rs = RandomizedSearchCV(svc, n_iter=50, scoring='roc_auc', n_jobs=-1,
                            param_distributions={'C': expon()})
    rs.fit(X_new, y)
    print('Grid scores: %s' % rs.grid_scores_)
    print('Best score (E_val): %s' % rs.best_score_)
    print('Best params: %s' % rs.best_params_)

    svc = rs.best_estimator_
    svc.fit(X_new, y)
    print('E_in: %f' % auc_score(svc, X_new, y))
    to_submission(svc, 'svc_1_0619_01')


if __name__ == '__main__':
    svc_1()
