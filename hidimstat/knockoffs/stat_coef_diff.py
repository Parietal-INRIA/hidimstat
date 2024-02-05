# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from sklearn.linear_model import (ElasticNetCV, LassoCV, LogisticRegressionCV,
                                  RidgeCV, Lasso, LassoLarsCV, LassoLars)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
# from sklearn.linear_model._coordinate_descent import _alpha_grid
# from sklearn.model_selection import GridSearchCV

@ignore_warnings(category=ConvergenceWarning)
def stat_coef_diff(X, X_tilde, y, alpha_chosen=None, active_set=None, method='lasso_cv', n_splits=5, n_jobs=1,
                   n_lambdas=10, n_iter=1000, group_reg=1e-3, l1_reg=1e-3,
                   joblib_verbose=0, return_coef=False, return_alpha=False, 
                   solver='liblinear', seed=0):
    """Calculate test statistic by doing estimation with Cross-validation on
    concatenated design matrix [X X_tilde] to find coefficients [beta
    beta_tilda]. The test statistic is then:

                        W_j =  abs(beta_j) - abs(beta_tilda_j)

    with j = 1, ..., n_features

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        Original design matrix

    X_tilde : 2D ndarray (n_samples, n_features)
        Knockoff design matrix

    y : 1D ndarray (n_samples, )
        Response vector

    loss : str, optional
        if the response vector is continuous, the loss used should be
        'least_square', otherwise
        if the response vector is binary, it should be 'logistic'

    n_splits : int, optional
        number of cross-validation folds

    solver : str, optional
        solver used by sklearn function LogisticRegressionCV

    n_regu : int, optional
        number of regulation used in the regression problem

    return_coef : bool, optional
        return regression coefficient if set to True

    Returns
    -------
    test_score : 1D ndarray (n_features, )
        vector of test statistic

    coef: 1D ndarray (n_features * 2, )
        coefficients of the estimation problem
    """

    n_features = X.shape[1]
    X_ko = np.column_stack([X, X_tilde])
    lambda_max = np.max(np.dot(X_ko.T, y)) / (2 * n_features)
    lambdas = np.linspace(
        lambda_max*np.exp(-n_lambdas), lambda_max, n_lambdas)

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
#   
    estimator = {
        'lasso_cv': LassoLarsCV(n_jobs=n_jobs, cv=cv),
        'logistic_l1': LogisticRegressionCV(
            penalty='l1', max_iter=int(1e4),
            solver=solver, cv=cv,
            n_jobs=n_jobs),
        'logistic_l2': LogisticRegressionCV(
            penalty='l2', max_iter=int(1e4), n_jobs=n_jobs,
            verbose=joblib_verbose, cv=cv, tol=1e-8),
        'logistic_l2_nocv': LogisticRegression(
            penalty='l2', max_iter=int(1e4), n_jobs=n_jobs),
        'enet': ElasticNetCV(cv=cv, max_iter=int(1e4), tol=1e-6,
                             n_jobs=n_jobs, verbose=joblib_verbose),
        'xgb' : XGBRegressor(n_jobs=n_jobs, booster='gblinear'),
        'linear_reg': LinearRegression(n_jobs=n_jobs),
        'ridge': RidgeCV([1, 5, 10, 20, 50, 100], cv=cv),
        'lasso_no_cv': Lasso(alpha=1e-5,max_iter=int(1e4)),


    }

    try:
        clf = estimator[method]
    except KeyError:
        print('{} is not a valid estimator'.format(method))

    if alpha_chosen is not None:
        if method == 'lasso_cv':
            clf = LassoLars(alpha=alpha_chosen, max_iter=active_set)
        if method == 'logistic_l2_nocv':
            clf = LogisticRegression(penalty='l2', C=alpha_chosen, max_iter=int(1e4), n_jobs=n_jobs)

    clf.fit(X_ko, y)
    

    try:
        coef = np.ravel(clf.coef_)
    except AttributeError:
        coef = np.ravel(clf.best_estimator_.coef_)  # for GridSearchCV object

    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])

    if return_coef:
        return test_score, coef

    if alpha_chosen is None:
        if method == 'lasso_cv' and return_alpha:
            alpha_ = clf.alpha_
            # active_set = len(clf.active_)
            active_set = 0
            return test_score, alpha_, active_set
    
    # try:
        # print(clf.alpha_)
        
    # except AttributeError:
        # print(alpha_chosen)
    
    return test_score


@ignore_warnings(category=ConvergenceWarning)
def stat_coef_diff_test(X, X_tilde, y, alpha_chosen=None, active_set=None, method='lasso_cv', n_splits=5, n_jobs=1,
                   n_lambdas=10, n_iter=1000, group_reg=1e-3, l1_reg=1e-3,
                   joblib_verbose=0, return_coef=False, return_alpha=False, 
                   solver='liblinear', seed=0):

    n_features = X.shape[1]
    X_ko = np.column_stack([X, X_tilde])

    if alpha_chosen is not None and method == 'lasso_cv':
        clf = LassoLars(alpha=alpha_chosen, precompute=False, max_iter=active_set)

    clf.fit(X_ko, y)
    
    try:
        coef = np.ravel(clf.coef_)
    except AttributeError:
        coef = np.ravel(clf.best_estimator_.coef_)  # for GridSearchCV object

    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])
    
    return test_score


def _coef_diff_threshold(test_score, fdr=0.1, offset=1):
    """Calculate the knockoff threshold based on the procedure stated in the
    article.

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        vector of test statistic

    fdr : float, optional
        desired controlled FDR level

    offset : int, 0 or 1, optional
        offset equals 1 is the knockoff+ procedure

    Returns
    -------
    thres : float or np.inf
        threshold level
    """
    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    t_mesh = np.sort(np.abs(test_score[test_score != 0]))
    for t in t_mesh:
        false_pos = np.sum(test_score <= -t)
        selected = np.sum(test_score >= t)
        if (offset + false_pos) / np.maximum(selected, 1) <= fdr:
            return t

    return np.inf