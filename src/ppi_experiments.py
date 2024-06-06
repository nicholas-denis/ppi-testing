import numpy as np
import matplotlib.pyplot as plt
import ppi_py
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import pandas as pd

import yaml
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging


# Fitting functions

def build_fit_slr(x,y):
    """
    y_hat = model.predict(x)
    """
    model = LinearRegression()
    model.fit(x,y)
    return model

def build_fit_dt(x,y):
    """
    y_hat = model.predict(x)
    """
    model = DecisionTreeRegressor()
    model.fit(x,y)
    return model

def build_fit_rf(x,y):
    """
    y_hat = model.predict(x)
    """
    model = RandomForestRegressor()
    model.fit(x,y)
    return model

# Population sampling functions

def sample_y_linear(x, m, beta, rho):
    """
    Given a sample of X, we generate Y as a linear function (SLR).
    """

    # compute sigma squared
    #sigma_squared = (beta**2)*gamma*(1.-rho)/rho
    sigma_squared = abs(m)*beta*(1.-(rho**2))/(rho**2)

    # sample noise (e_i)
    sampled_e = np.random.normal(loc=0,
                                 scale=sigma_squared*x,
                                 size=x.shape
                                 )

    # compute y as a linear function of x
    y = m*x + sampled_e

    return y

def sample_gamma_population(alpha,
                            beta,
                            sample_size,
                            m,
                            rho = None
                            ):
    """
    Sample <sample_size> instances, x_i ~ G(alpha, gamma)

    Then creates y_i ~ beta*x_i + e_i,
      where e_i ~ N(0, sigma^2_e*x_i), where
      sigma^2_e = (beta^2)gamma[(1-r^2)/)r^2]
      for rho(x_i, y_i) = r ==> desired correlation
    """

    assert np.any(rho != 0), "rho must be different from 0"

    # sample x
    x = np.random.gamma(alpha, beta, sample_size)

    # sample y
    y = sample_y_linear(x, m, beta, rho)

    return x, y

def sample_mv_y_linear(x, vec, beta, rho):
    sigma_squared = abs(m)*beta*(1.-(rho**2))/(rho**2)

    # sample noise (e_i)
    sampled_e = np.random.normal(loc=0,
                                 scale=sigma_squared,
                                 size=(x.shape[0], 1)
                                 )

    # compute y as a linear function of x
    y = x @ vec + sampled_e

    return y

def sample_mv_gamma_population(alpha, beta, sample_size, vec, rho=None):
    assert np.any(rho != 0), "rho must be different from 0"

    # sample x
    x = np.random.gamma(alpha, beta, (sample_size, vec.shape[0]))
    # sample y
    y = sample_mv_y_linear(x, vec, beta, rho)

    return x, y


def basic_experiment(config):

    alpha = config['experiment']['parameters']['alpha']
    beta = config['experiment']['parameters']['beta']
    m = config['experiment']['parameters']['m']
    rho_vals = config['experiment']['parameters']['rho']
    n_its = config['experiment']['parameters']['n_its']
    train_sample = config['experiment']['parameters']['train_sample']
    small_sample = config['experiment']['parameters']['small_sample']
    large_sample = config['experiment']['parameters']['large_sample']

    # Check if reproducibility is required
    if config.get('reproducibility'):
        np.random.seed(config['reproducibility'].get('seed'))

    # Prepare storage for results
    ppi_mean_widths = []
    naive_mean_widths = []
    classical_mean_widths = []

    true_value = m * alpha * beta

    for r in rho_vals:
        rho = r

        ppi_preds = []
        ppi_lowers = []
        ppi_uppers = []
        ppi_true_bias = []
        ppi_covered = []

        naive_preds = []
        naive_lowers = []
        naive_uppers = []
        naive_covered = []

        classical_preds = []
        classical_lowers = []
        classical_uppers = []
        classical_covered = []

        ppi_widths = []
        naive_widths = []
        classical_widths = []

        residuals = []

        for i in range(n_its):
            # Sampling
            x_train, y_train = sample_gamma_population(alpha, beta, train_sample, m, rho)
            x_gold, y_gold = sample_gamma_population(alpha, beta, small_sample, m, rho)
            x_ppi, y_ppi = sample_gamma_population(alpha, beta, large_sample, m, rho)

            # Reshaping for scikitlearn
            x_train, y_train = np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1)
            x_gold, y_gold = np.array(x_gold).reshape(-1, 1), np.array(y_gold).reshape(-1, 1)
            x_ppi, y_ppi = np.array(x_ppi).reshape(-1, 1), np.array(y_ppi).reshape(-1, 1)

            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
            model = build_fit_slr(x_train, y_train)  # Training

            # Residual testing
            y_test_pred = model.predict(x_test)
            residual = np.sqrt(np.mean((y_test_pred - y_test)**2))  # Mean squared deviation
            residuals.append(residual)

            y_gold_fitted = model.predict(x_gold)  # Gold standard fitted
            y_fitted = model.predict(x_ppi)  # Unlabelled fitted

            # Manual rectifier computation
            rectifier = np.mean(y_gold_fitted - y_gold)

            # PPI section
            ppi_theta = ppi_py.ppi_mean_pointestimate(y_gold, y_gold_fitted, y_fitted)
            ppi_theta_ci = ppi_py.ppi_mean_ci(y_gold, y_gold_fitted, y_fitted)

            ppi_preds.append(ppi_theta[0])
            ppi_lowers.append(ppi_theta_ci[0][0])
            ppi_uppers.append(ppi_theta_ci[1][0])
            ppi_true_bias.append(np.mean(y_fitted - y_ppi))
            if true_value >= ppi_theta_ci[0][0] and true_value <= ppi_theta_ci[1][0]:
                ppi_covered.append(1)
            else:
                ppi_covered.append(0)
            ppi_widths.append(ppi_theta_ci[1][0] - ppi_theta_ci[0][0])

            # Naive imputation
            concat = np.vstack((y_gold, y_fitted)).flatten()  # Concatenate samples
            naive_theta = np.mean(concat)  # Compute mean
            naive_sigma = np.std(concat)  # Compute std dev
            n_tot = concat.shape[0]
            naive_theta_ci = stats.norm.interval(0.9, loc=naive_theta, scale=naive_sigma/np.sqrt(n_tot))  # Use norm as N is large

            naive_preds.append(naive_theta)
            naive_lowers.append(naive_theta_ci[0])
            naive_uppers.append(naive_theta_ci[1])
            if true_value >= naive_theta_ci[0] and true_value <= naive_theta_ci[1]:
                naive_covered.append(1)
            else:
                naive_covered.append(0)
            naive_widths.append(naive_theta_ci[1] - naive_theta_ci[0])

            # Classical prediction (Gold standard only)
            classical_theta, classical_se = np.mean(y_gold.flatten()), stats.sem(y_gold.flatten())
            h = classical_se * stats.t.ppf((1 + .9) / 2., small_sample-1)  # Highly stolen code, uses t-dist here

            classical_preds.append(classical_theta)
            classical_lowers.append(classical_theta - h)
            classical_uppers.append(classical_theta + h)
            if true_value >= classical_theta - h and true_value <= classical_theta + h:
                classical_covered.append(1)
            else:
                classical_covered.append(0)
            classical_widths.append(2 * h)

        print("Rho: ", rho)
        print("Average residual: ", np.mean(residuals))
        print("PPI.", "Average pred: ", np.mean(ppi_preds), "Average lower bound: ", np.mean(ppi_lowers), "Average upper bound: ", np.mean(ppi_uppers), "Average width: ", np.mean(ppi_widths), "Covered percent: ", np.mean(ppi_covered))
        print("Naive.", "Average pred: ", np.mean(naive_preds), "Average lower bound: ", np.mean(naive_lowers), "Average upper bound: ", np.mean(naive_uppers), "Average width: ", np.mean(naive_widths), "Covered percent: ", np.mean(naive_covered))
        print("Classical.",  "Average pred: ", np.mean(classical_preds), "Average lower bound: ", np.mean(classical_lowers), "Average upper bound: ", np.mean(classical_uppers), "Average width: ", np.mean(classical_widths), "Covered percent: ", np.mean(classical_covered))

        ppi_mean_widths.append(np.mean(ppi_widths))
        naive_mean_widths.append(np.mean(naive_widths))
        classical_mean_widths.append(np.mean(classical_widths))

    plt.figure(dpi=400)

    # Plot each y against x
    plt.plot(rho_vals, ppi_mean_widths, label='PPI Mean Widths', marker='o')
    plt.plot(rho_vals, naive_mean_widths, label='Naive Mean Widths', marker='x')
    plt.plot(rho_vals, classical_mean_widths, label='Classical Mean Widths', marker='s')

    # Add labels and title
    plt.xlabel('Rho Value (Level of noise)')
    plt.ylabel('Confidence Interval Width')
    plt.title('Confidence Interval Widths vs Amount of noise')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Save the plot in new experiment folder
    plt.savefig(os.path.join(config['paths']['plotting_path'], 'noiseplot.png'))

    # Save the widths in a pandas dataframe

    data = {'PPI Mean Widths': ppi_mean_widths, 'Naive Mean Widths': naive_mean_widths, 'Classical Mean Widths': classical_mean_widths}
    row_labels = [f"Rho: {r}" for r in rho_vals]

    df = pd.DataFrame(data, index=row_labels)

    return df
