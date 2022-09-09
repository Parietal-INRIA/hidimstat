"""
This example compares the performance of d0crt based on
the lasso (1) and random forest (2) implementations. The number of
repetitions is set to 100. The metrics used are the type-I error and
the power
""" 

import numpy as np
from scipy.linalg import toeplitz
from hidimstat.dcrt import dcrt_zero

dcrtLasso_type1 = []
dcrtLasso_power = []
dcrtForest_type1 = []
dcrtForest_power = []

for sim_ind in range(100):
    print(f"Processing: {sim_ind+1}")
    np.random.seed(sim_ind)
    DEBUG = False

    n = 1000 if not DEBUG else 10
    p = 10 if not DEBUG else 5
    n_signal = 2 if not DEBUG else 1
    snr = 4
    rho = 0.8

    # Create Correlation matrix with the toeplitz design
    cov_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j:
                cov_matrix[i, j] = rho
            else:
                cov_matrix[i, j] = 1

    # Generation of the predictors
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov_matrix, size=n)

    # Random choice of the relevant variables
    list_var = np.random.choice(p, n_signal, replace=False)
    reorder_var = np.array([i for i in range(p) if i not in list_var])

    # Reorder data matrix so that first n_signal predictors are the signal predictors
    X = X[:, np.concatenate([list_var, reorder_var], axis=0)]

    # Random choice of the coefficients for the signal
    effectset = np.array([-0.5, -1, -2, -3, 0.5, 1, 2, 3])
    betas = np.random.choice(effectset, n_signal)

    prod_signal = np.dot(X[:, :n_signal], betas)
    sigma_noise = np.linalg.norm(prod_signal) / (snr * np.sqrt(X.shape[0]))

    # Generation of the outcome
    y = prod_signal + sigma_noise * np.random.normal(0, 1, size=X.shape[0])
    y = np.array([max(0.0, i) for i in y])

    ## dcrt Lasso ##
    res_lasso = dcrt_zero(X, y, screening=False, verbose=True)
    dcrtLasso_type1.append(sum(res_lasso[1][n_signal:] < 5e-2) / (p-n_signal))
    dcrtLasso_power.append(sum(res_lasso[1][:n_signal] < 5e-2) / (n_signal))

    ## dcrt Random Forest ##
    res_forest = dcrt_zero(X, y, screening=False, statistic="randomforest", verbose=True)
    dcrtForest_type1.append(sum(res_forest[1][n_signal:] < 5e-2) / (p-n_signal))
    dcrtForest_power.append(sum(res_forest[1][:n_signal] < 5e-2) / (n_signal))

print(f"Lasso: Type-I error = {np.mean(np.array(dcrtLasso_type1))}")
print(f"Lasso: Power = {np.mean(np.array(dcrtLasso_power))}")

print(f"Forest: Type-I error = {np.mean(np.array(dcrtForest_type1))}")
print(f"Forest: Power = {np.mean(np.array(dcrtForest_power))}")