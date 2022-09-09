"""
This example compares the performance of d0crt based on
the lasso (1) and random forest (2) implementations. The number of
repetitions is set to 100. The metrics used are the type-I error and
the power
"""

import numpy as np
import matplotlib.pyplot as plt
from hidimstat.dcrt import dcrt_zero

typeI_error = {"Lasso": [], "Forest": []}
power = {"Lasso": [], "Forest": []}

for sim_ind in range(100):
    print(f"Processing: {sim_ind+1}")
    np.random.seed(sim_ind)

    # Number of observations
    n = 1000
    # Number of variables
    p = 10
    # Number of relevant variables
    n_signal = 2
    # Signal-to-noise ratio
    snr = 4
    # Correlation coefficient
    rho = 0.8
    # Nominal false positive rate
    alpha = 5e-2

    # Create Correlation matrix
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

    # Reorder data matrix so that first n_signal predictors
    # are the signal predictors
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
    results_lasso = dcrt_zero(X, y, screening=False, verbose=True)
    typeI_error["Lasso"].append(
        sum(results_lasso[1][n_signal:] < alpha) / (p - n_signal))
    power["Lasso"].append(
        sum(results_lasso[1][:n_signal] < alpha) / (n_signal))

    ## dcrt Random Forest ##
    results_forest = dcrt_zero(X, y, screening=False, statistic="randomforest",
                               verbose=True)
    typeI_error["Forest"].append(
        sum(results_forest[1][n_signal:] < alpha) / (p - n_signal))
    power["Forest"].append(
        sum(results_forest[1][:n_signal] < alpha) / (n_signal))

fig, ax = plt.subplots()
ax.set_title("Type-I Error")
ax.boxplot(typeI_error.values())
ax.set_xticklabels(typeI_error.keys())
ax.axhline(linewidth=2, color='r')

fig, ax = plt.subplots()
ax.set_title("Power")
ax.boxplot(power.values())
ax.set_xticklabels(power.keys())

plt.show()
