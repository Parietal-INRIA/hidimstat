# -*- coding: utf-8 -*-
# Author: Binh Nguyen <tuan-binh.nguyen@inria.fr> & Jerome-Alexis Chevalier
import numpy as np


def fdr_threshold(pvals, fdr=0.1, method='bhq', reshaping_function=None):
    if method == 'bhq':
        return _bhq_threshold(pvals, fdr=fdr)
    elif method == 'bhy':
        return _bhy_threshold(
            pvals, fdr=fdr, reshaping_function=reshaping_function)
    elif method == 'adapt':
        return _adapt_threshold(pvals, fdr=fdr)
    else:
        raise ValueError(
            '{} is not support FDR control method'.format(method))


def _bhq_threshold(pvals, fdr=0.1):
    """Standard Benjamini-Hochberg for controlling False discovery rate
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if pvals_sorted[i] <= fdr * (i + 1) / n_features:
            selected_index = i
            break
    if selected_index <= n_features:
        return pvals_sorted[selected_index]
    else:
        return -1.0


def _bhy_threshold(pvals, reshaping_function=None, fdr=0.1):
    """Benjamini-Hochberg-Yekutieli procedure for controlling FDR, with input
    shape function. Reference: Ramdas et al (2017)
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    # Default value for reshaping function -- defined in
    # Benjamini & Yekutieli (2001)
    if reshaping_function is None:
        temp = np.arange(n_features)
        sum_inverse = np.sum(1 / (temp + 1))
        return _bhq_threshold(pvals, fdr / sum_inverse)
    else:
        for i in range(n_features - 1, -1, -1):
            if pvals_sorted[i] <= fdr * reshaping_function(i + 1) / n_features:
                selected_index = i
                break
        if selected_index <= n_features:
            return pvals_sorted[selected_index]
        else:
            return -1.0


def _adapt_threshold(pvals, fdr=0.1):
    """FDR controlling with AdaPT procedure (Lei & Fithian '18), in particular
    using the intercept only version, shown in Wang & Janson '20 section 3
    """
    pvals_sorted = pvals[np.argsort(-pvals)]

    for pv in pvals_sorted:
        false_pos = np.sum(pvals >= 1 - pv)
        selected = np.sum(pvals <= pv)
        if (1 + false_pos) / np.maximum(1, selected) <= fdr:
            return pv

    return -1.0


def _lambda_max(X, y, use_noise_estimate=True):
    """Calculation of lambda_max, the smallest value of regularization parameter in
    lasso program for non-zero coefficient
    """
    n_samples, _ = X.shape

    if not use_noise_estimate:
        return np.max(np.dot(X.T, y)) / n_samples

    norm_y = np.linalg.norm(y, ord=2)
    sigma_0 = (norm_y / np.sqrt(n_samples)) * 1e-3
    sig_star = max(sigma_0, norm_y / np.sqrt(n_samples))

    return np.max(np.abs(np.dot(X.T, y)) / (n_samples * sig_star))
