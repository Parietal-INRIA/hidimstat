import numpy as np
from scipy.linalg import toeplitz
from scipy.special import expit
from joblib import Parallel, delayed
from sklearn.linear_model import (LassoCV, LinearRegression, Lasso,
                                  LogisticRegression, LogisticRegressionCV)
from sklearn.linear_model import (LassoLarsCV, LassoLars)
from sklearn.utils import check_random_state
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter

def simu_data(n, p, rho=0.25, snr=2.0, sparsity=0.06, effect=1.0, Sigma_real=None, binarize=False, seed=None):
    """Function to simulate data follow an autoregressive structure with Toeplitz
    covariance matrix

    Parameters
    ----------
    n : int
        number of observations
    p : int
        number of variables
    sparsity : float, optional
        ratio of number of variables with non-zero coefficients over total
        coefficients
    rho : float, optional
        correlation parameter
    effect : float, optional
        signal magnitude, value of non-null coefficients
    seed : None or Int, optional
        random seed for generator

    Returns
    -------
    X : ndarray, shape (n, p)
        Design matrix resulted from simulation
    y : ndarray, shape (n, )
        Response vector resulted from simulation
    beta_true : ndarray, shape (n, )
        Vector of true coefficient value
    non_zero : ndarray, shape (n, )
        Vector of non zero coefficients index

    """
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # Number of non-null
    k = int(sparsity * p)

    # Generate the variables from a multivariate normal distribution
    mu = np.zeros(p)
    if Sigma_real is None:
        Sigma = toeplitz(rho ** np.arange(0, p))  # covariance matrix of X
    else:
        Sigma = Sigma_real
    # X = np.dot(np.random.normal(size=(n, p)), cholesky(Sigma))
    X = rng.multivariate_normal(mu, Sigma, size=(n))
    # Generate the response from a linear model
    blob_indexes = np.linspace(0, p - 6, int(k/5), dtype=int)
    non_zero = np.array([np.arange(i, i+5) for i in blob_indexes], dtype=int)
    # non_zero = rng.choice(p, k, replace=False)
    beta_true = np.zeros(p)
    beta_true[non_zero] = effect
    eps = rng.standard_normal(size=n)
    prod_temp = np.dot(X, beta_true)
    noise_mag = np.linalg.norm(prod_temp) / (snr * np.linalg.norm(eps))
    if binarize:
        y = [(2 * np.random.binomial(1, ysig)) - 1 for ysig in expit(prod_temp)]
    
    else:
        y = prod_temp + noise_mag * eps

    return X, y, beta_true, non_zero, Sigma


def simu_data_conditional(X_real, beta_input=None, snr=2.0, sparsity=0.06, effect=1.0, Sigma_real=None, binarize=False, n_jobs=1, seed=None):
    """Function to simulate data follow an autoregressive structure with Toeplitz
    covariance matrix

    Parameters
    ----------
    n : int
        number of observations
    p : int
        number of variables
    sparsity : float, optional
        ratio of number of variables with non-zero coefficients over total
        coefficients
    rho : float, optional
        correlation parameter
    effect : float, optional
        signal magnitude, value of non-null coefficients
    seed : None or Int, optional
        random seed for generator

    Returns
    -------
    X : ndarray, shape (n, p)
        Design matrix resulted from simulation
    y : ndarray, shape (n, )
        Response vector resulted from simulation
    beta_true : ndarray, shape (n, )
        Vector of true coefficient value
    non_zero : ndarray, shape (n, )
        Vector of non zero coefficients index

    """
    # Setup seed generator
    rng = np.random.default_rng(seed)

    n, p = X_real.shape

    X = conditional_sequential_gen(X_real, n_jobs=n_jobs, seed=seed)

    # Number of non-null
    k = int(sparsity * p)

    # Generate the response from a linear model
    if beta_input is None:
        blob_indexes = np.linspace(0, p - 6, int(k/5), dtype=int)
        non_zero = np.array([np.arange(i, i+5) for i in blob_indexes], dtype=int)
        # non_zero = rng.choice(p, k, replace=False)
        beta_true = np.zeros(p)
        beta_true[non_zero] = effect
    else:
        beta_true = beta_input
        non_zero = np.where(beta_true != 0)[0]
        #if len(non_zero) > int(sparsity * p):
            #non_zero = np.argsort(- abs(beta_true))[:int(sparsity * p)]
    eps = rng.standard_normal(size=n)
    prod_temp = np.dot(X, beta_true)
    noise_mag = np.linalg.norm(prod_temp) / (snr * np.linalg.norm(eps))
    if binarize:
        y = [(2 * np.random.binomial(1, ysig)) - 1 for ysig in expit(prod_temp)]
    
    else:
        y = prod_temp + noise_mag * eps

    return X, y, beta_true, non_zero, None


def conditional_sequential_gen(X, n_jobs=1, seed=None):
    rng = check_random_state(seed)
    
    n, p = X.shape

    clfs = Parallel(n_jobs=n_jobs)(delayed(
        _get_single_clf)(X, j) for j in tqdm(range(1, p)))

    samples = _get_samples(X, clfs, seed=seed)
    
    return samples


def _get_single_clf(X, j):
    lambda_max = np.max(np.dot(X[:, :j].T, X[:, j])) / (2 * j)
    alpha = (lambda_max / 100)
    clf = Lasso(alpha)
    clf.fit(X[:, :j], X[:, j])
    return clf


def _get_samples(X, clfs, seed=None):
    np.random.seed(seed)
    n, p = X.shape
    X_to_shuffle = X.copy()
    samples = np.zeros((n, p))
    
    current_col = X_to_shuffle[:, 0]
    np.random.shuffle(current_col)
    samples[:, 0] = current_col

    for j in tqdm(range(len(clfs))):
        residuals = X[:, j + 1] - clfs[j].predict(X[:, :j + 1])
        indices_ = np.arange(residuals.shape[0])
        np.random.shuffle(indices_)

        samples[:, j + 1] = clfs[j].predict(samples[:, :j + 1]) + residuals[indices_]
        c = 1
        # samples[:, j + 1] = _adjust_marginal(samples[:, j + 1], X[:, j + 1])

    
    return samples


def simu_data_cov(beta_input, n, p, snr=2.0, sparsity=0.06, effect=1.0, Sigma_real=None, binarize=False, n_jobs=1, seed=None):
    """Function to simulate data follow an autoregressive structure with Toeplitz
    covariance matrix

    Parameters
    ----------
    n : int
        number of observations
    p : int
        number of variables
    sparsity : float, optional
        ratio of number of variables with non-zero coefficients over total
        coefficients
    rho : float, optional
        correlation parameter
    effect : float, optional
        signal magnitude, value of non-null coefficients
    seed : None or Int, optional
        random seed for generator

    Returns
    -------
    X : ndarray, shape (n, p)
        Design matrix resulted from simulation
    y : ndarray, shape (n, )
        Response vector resulted from simulation
    beta_true : ndarray, shape (n, )
        Vector of true coefficient value
    non_zero : ndarray, shape (n, )
        Vector of non zero coefficients index

    """
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # p = len(Sigma_real)

    mu = np.zeros(p)

    if Sigma_real is None:
        Sigma = toeplitz(rho ** np.arange(0, p))  # covariance matrix of X
    else:
        Sigma = Sigma_real
    # X = np.dot(np.random.normal(size=(n, p)), cholesky(Sigma))
    X = rng.multivariate_normal(mu, Sigma, size=(n))

    # Number of non-null
    k = int(sparsity * p)

    # Generate the response from a linear model
    if beta_input is None:
        blob_indexes = np.linspace(0, p - 6, int(k/5), dtype=int)
        non_zero = np.array([np.arange(i, i+5) for i in blob_indexes], dtype=int)
        # non_zero = rng.choice(p, k, replace=False)
        beta_true = np.zeros(p)
        beta_true[non_zero] = effect
    else:
        beta_true = beta_input
        non_zero = np.where(beta_true != 0)[0]
        if len(non_zero) > int(sparsity * p):
            non_zero = np.argsort(- abs(beta_true))[:int(sparsity * p)]
    eps = rng.standard_normal(size=n)
    prod_temp = np.dot(X, beta_true)
    noise_mag = np.linalg.norm(prod_temp) / (snr * np.linalg.norm(eps))
    if binarize:
        y = [(2 * np.random.binomial(1, ysig)) - 1 for ysig in expit(prod_temp)]
    
    else:
        y = prod_temp + noise_mag * eps

    return X, y, beta_true, non_zero, None


def _adjust_marginal(v, ref):
    """
    Make v follow the marginal of ref.
    """
    F = ECDF(ref)
    G = ECDF(v)
    G_inv = monotone_fn_inverter(G, v, vectorized=False)
    c = 1
    return F(G_inv(v))