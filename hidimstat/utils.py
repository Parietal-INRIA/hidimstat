import numpy as np


def quantile_aggregation(pvals, gamma=0.5, gamma_min=0.05, adaptive=False):
    if adaptive:
        return _adaptive_quantile_aggregation(pvals, gamma_min)
    else:
        return _fixed_quantile_aggregation(pvals, gamma)


def fdr_threshold(pvals, fdr=0.1, method='bhq', reshaping_function=None):
    if method == 'bhq':
        return _bhq_threshold(pvals, fdr=fdr)
    elif method == 'bhy':
        return _bhy_threshold(
            pvals, fdr=fdr, reshaping_function=reshaping_function)
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


def _fixed_quantile_aggregation(pvals, gamma=0.5):
    """Quantile aggregation function based on Meinshausen et al (2008)
    Parameters
    ----------
    pvals : 2D ndarray (n_bootstrap, n_test)
        p-value (adjusted)
    gamma : float
        Percentile value used for aggregation.
    Returns
    -------
    1D ndarray (n_tests, )
        Vector of aggregated p-value
    """
    converted_score = (1 / gamma) * (
        np.percentile(pvals, q=100*gamma, axis=0))

    return np.minimum(1, converted_score)


def _adaptive_quantile_aggregation(pvals, gamma_min=0.05):
    """adaptive version of the quantile aggregation method, Meinshausen et al.
    (2008)"""
    gammas = np.arange(gamma_min, 1.05, 0.05)
    list_Q = np.array([
        _fixed_quantile_aggregation(pvals, gamma) for gamma in gammas])

    return np.minimum(1, (1 - np.log(gamma_min)) * list_Q.min(0))