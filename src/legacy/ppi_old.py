import numpy as np
import matplotlib.pyplot as plt
import ppi_py
import scipy.stats as stats
import pandas as pd

import yaml
import os
import sys
import argparse
import matplotlib.pyplot as plt
import datetime
import logging
import warnings
import time

import distributions as dist
import ml_models as ml

# colored text
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

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
    sigma_squared = np.linalg.norm(vec)*beta*(1.-(rho**2))/(rho**2)

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


def sample_population(population_dict):
    """
    Sample a population from a given dictionary of parameters

    TODO:
    - Add custom distributions, not modular right now

    Args:
    x_dict (dict): dictionary of parameters for x
    y_dict (dict): dictionary of parameters for y

    Returns:
    x (np.array): sampled x
    y (np.array): sampled y
    """
    x_dict = population_dict['x_population']
    y_dict = population_dict['y_population']
    # Sample x
    if x_dict['distribution'] == 'gamma_univariate':
        x = np.random.gamma(x_dict['alpha'], x_dict['beta'], x_dict['size']).reshape(-1, 1)
    else:
        raise ValueError("Distribution not supported")
    # Sample y based on x
    if y_dict['transformation'] == 'linear_univariate':
        y = sample_y_linear(x, y_dict['m'], x_dict['beta'], y_dict['rho']).reshape(-1, 1)

    return x, y

def train_model(x_train, y_train, model_config):
    """
    Train a model based on the configuration

    Args:
    x_train (np.array): training x
    y_train (np.array): training y
    model_config (dict): configuration for the model

    Returns:
    model: trained model
    """
    if model_config['name'] == 'linear_regression':
        model = build_fit_slr(x_train, y_train)
    elif model_config['name'] == 'decision_tree':
        model = build_fit_dt(x_train, y_train)
    elif model_config['name'] == 'random_forest':
        model = build_fit_rf(x_train, y_train)
    else:
        raise ValueError("Model not supported")
    return model
    

def compute_metrics(metric_list, estimates_d, config, secondary=False):
    """
    A general metric function to be used in the experiment.
    BIG TODO, WILL NEED A LOT OF WORK
    Currently only computes mean and CI for width

    Will probably split this function into estimation + CI function
    and a function to compute the metrics

    params:
    metric_list (list): list of metrics to compute
    estimates_d (dict): dictionary of estimates
    config (dict): configuration for the experiment (probably not needed)
    """
    metrics = {}
    if secondary:
        metrics['ppi_preds'] = estimates_d['ppi_theta']
        metrics['ppi_lowers'] = estimates_d['ppi_theta_ci'][0]
        metrics['ppi_uppers'] = estimates_d['ppi_theta_ci'][1]

        metrics['naive_preds'] = estimates_d['naive_theta']
        metrics['naive_lowers'] = estimates_d['naive_theta_ci'][0]
        metrics['naive_uppers'] = estimates_d['naive_theta_ci'][1]

        metrics['classical_preds'] = estimates_d['classical_theta']
        metrics['classical_lowers'] = estimates_d['classical_ci'][0]
        metrics['classical_uppers'] = estimates_d['classical_ci'][1]
    for metric in metric_list:
        if metric == 'widths':
            metrics['ppi_widths'] = estimates_d['ppi_theta_ci'][1] - estimates_d['ppi_theta_ci'][0]
            metrics['naive_widths'] = estimates_d['naive_theta_ci'][1] - estimates_d['naive_theta_ci'][0]
            metrics['classical_widths'] = estimates_d['classical_ci'][1] - estimates_d['classical_ci'][0]
        elif metric == 'coverages':
            true_value = config['experiment']['parameters'].get('true_value', None)
            if true_value is None:
                raise ValueError("True value not provided for coverage computation")
            metrics['ppi_coverages'] = np.mean([1 if true_value >= estimates_d['ppi_theta_ci'][0] and true_value <= estimates_d['ppi_theta_ci'][1] else 0])
            metrics['naive_coverages'] = np.mean([1 if true_value >= estimates_d['naive_theta_ci'][0] and true_value <= estimates_d['naive_theta_ci'][1] else 0])
            metrics['classical_coverages'] = np.mean([1 if true_value >= estimates_d['classical_ci'][0] and true_value <= estimates_d['classical_ci'][1] else 0])
    
    return metrics


def compute_ci(config, y_gold, y_gold_fitted, y_fitted):
    """
    Computes confidence interval and estimate.

    Currently only computes 1 estimate, but hopefully will compute multiple in the future

    return: tuple of (ppi_theta, ppi_theta_ci, naive_theta, naive_theta_ci, classical_theta, classical_ci)
    """
    d = {}
    if config['experiment']['estimate'] == 'mean':
        # PPI CI
        ppi_theta = ppi_py.ppi_mean_pointestimate(y_gold, y_gold_fitted, y_fitted)
        ppi_theta_ci = ppi_py.ppi_mean_ci(y_gold, y_gold_fitted, y_fitted)
        
        # Naive CI
        concat = np.vstack((y_gold, y_fitted)).flatten()  # Concactenate samples
        naive_theta = np.mean(concat)  # Compute mean
        naive_sigma = np.std(concat)  # Compute std dev
        n_tot = concat.shape[0]
        naive_theta_ci = stats.norm.interval(0.9, loc=naive_theta, scale=naive_sigma/np.sqrt(n_tot))  # Use norm as N is large

        # Classical CI
        small_sample = y_gold.shape[0]
        classical_theta, classical_se = np.mean(y_gold.flatten()), stats.sem(y_gold.flatten())
        h = classical_se * stats.t.ppf((1 + .9) / 2., small_sample-1)  # Highly stolen code, uses t-dist here

        d['ppi_theta'] = ppi_theta
        d['ppi_theta_ci'] = ppi_theta_ci
        d['naive_theta'] = naive_theta
        d['naive_theta_ci'] = naive_theta_ci
        d['classical_theta'] = classical_theta
        d['classical_ci'] = (classical_theta - h, classical_theta + h)
    
    return d

def single_iteration(config):
    """
    A single iteration of the experiment

    Note to self: Return all the variables for appending later

    Experiment components:
    - Sampling
    - Reshaping for scikitlearn
    - Training
    - Residual testing
    """
    x_train, y_train = dist.sample_population(config['experiment']['parameters']['training_population'])
    x_gold, y_gold = dist.sample_population(config['experiment']['parameters']['gold_population'])
    x_ppi, y_ppi = dist.sample_population(config['experiment']['parameters']['unlabelled_population'])

    x_train, x_test, y_train, y_test = ml.train_test_split(x_train, y_train, test_size=config['experiment']['parameters'].get('test_size', 0.2))

    model = ml.train_model(x_train, y_train, config['experiment']['model'])  # Placeholder

    # Residual testing
    y_test_pred = model.predict(x_test)
    residual = np.sqrt(np.mean((y_test_pred - y_test)**2))  # Mean squared deviation

    # Fitting
    y_gold_fitted = model.predict(x_gold)  # Gold standard fitted
    y_fitted = model.predict(x_ppi)  # Unlabelled fitted

    # Manual rectifier computation
    rectifier = np.mean(y_gold_fitted - y_ppi)

    # True bias computation if called for
    if config['experiment']['parameters']['unlabelled_population'].get('include', False):
        true_bias = np.mean(y_fitted - y_ppi)

    # Confidence interval computation

    ci_results = compute_ci(config, y_gold, y_gold_fitted, y_fitted)

    # Metric computation

    primary_metrics = compute_metrics(config['experiment']['metrics']['primary'], ci_results, config, secondary=False)
    secondary_metrics = compute_metrics(config['experiment']['metrics']['secondary'], ci_results, config, secondary=True)

    return primary_metrics, secondary_metrics

def extend_metrics(metrics_dict, metrics_d):
    """
    Extend the metrics dictionary with the new metrics

    Might not need this anymore

    Args:
    metrics_dict (dict): dictionary of dictionaries of metrics each key is the independent variable
    metrics_d (dict): dictionary of metrics
    """
    for key, value in metrics_d.items():
        metrics_dict[key].append(value)

    return metrics_dict

def plot_metrics(metrics_dict, config):
    """
    Plot the metrics from the experiment, and saves them in the plotting folder

    Args:
    metrics_dict (dict): dictionary of dictionaries of metrics each key is the independent variable
    config (dict): configuration for the experiment
    """
    plt_title = config['experiment']['ind_var']['plot_description']
    for metric in config['experiment']['metrics']['primary']:
        if metric == 'widths':
            plt.figure(dpi=400)

            # Plot each y against x
            plt.plot(config['experiment']['ind_var']['vals'], metrics_dict['ppi_widths'], label='PPI Widths', marker='o')
            plt.plot(config['experiment']['ind_var']['vals'], metrics_dict['naive_widths'], label='Naive Widths', marker='x')
            plt.plot(config['experiment']['ind_var']['vals'], metrics_dict['classical_widths'], label='Classical Widths', marker='s')

            # Add labels and title
            plt.xlabel(config['experiment']['ind_var']['label_description'])
            plt.ylabel('Confidence Interval Width')
            plt.title('Confidence Interval Widths vs Amount of noise')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()  # Adjusts the spacing to prevent legend cutoff

            # Save the plot in new experiment folder
            plt.savefig(os.path.join(config['paths']['plotting_path'], 'widthsplot.png'), bbox_inches='tight')

            # plt.show()

            
    return 


def create_metrics_dict(config):
    """
    Create a dictionary of metrics to be updated in the experiment
    """
    primary_metrics = {}  # metrics to be graphed

    for metric in config['experiment']['metrics'].get('primary', []):
        primary_metrics["ppi_" + metric] = []
        primary_metrics["naive_" + metric] = []
        primary_metrics["classical_" + metric] = [] 

    secondary_metrics = {}  # metrics not to be graphed but to be stored

    # these will always be computed

    secondary_metrics['ppi_preds'] = []
    secondary_metrics['ppi_lowers'] = []
    secondary_metrics['ppi_uppers'] = []
    secondary_metrics['naive_preds'] = []
    secondary_metrics['naive_lowers'] = []
    secondary_metrics['naive_uppers'] = []
    secondary_metrics['classical_preds'] = []
    secondary_metrics['classical_lowers'] = []
    secondary_metrics['classical_uppers'] = []

    for metric in config['experiment']['metrics'].get('secondary', []):
        secondary_metrics["ppi_" + metric] = []
        secondary_metrics["naive_" + metric] = []
        secondary_metrics["classical_" + metric] = []
    
    return primary_metrics, secondary_metrics

def experiment(config):
    """
    WIP - This is the main experiment function
    Intended use: modular framework for running ppi experiments

    Each experiment has the following components:
    - Variable unpacking
    - Checking reproducibility
    - Metric storage
    - Iterating over independent variables
    - A single iteration of the experiment
    - Graphing the results
    """

    # unpack experiment parameters

    params = config['experiment']['parameters'] # dictionary

    # check if reproducibility is required
    if config.get('reproducibility'):
        np.random.seed(config['reproducibility'].get('seed'))

    # prepare metric results storage

    primary_metrics, secondary_metrics = create_metrics_dict(config)  # These will be means

    # iterate over independent variables

    print(f"{GREEN}Starting experiment{RESET}")

    for x in config['experiment']['ind_var']['vals']:
        print(f"{YELLOW}Running experiment with {config['experiment']['ind_var']['name']} = {x}{RESET}")
        # begin timing
        start = time.time()
        ind_var_primary, ind_var_secondary = create_metrics_dict(config)
        for path in config['experiment']['ind_var']['paths']:
            keys = path.split('.')  # Split the path
            # update the independent variable, I'm about to update the original config in place.
            temp = config
            for key in keys[:-1]:
                temp = temp[key]
            # Copilot code, I don't know if this is correct. It should be.
            # Assign a new value to the last key
            temp[keys[-1]] = x
        for i in range(params['n_its']):
            # single iteration of the experiment
            iter_primary, iter_secondary = single_iteration(config) 
            # update the metrics
            # there's a clever thing happening where the dict outside of the loop
            # has the same keys as the dict inside the loop even if they do differnt things
            # the outside loop has mean values per iter, the inside loop has all the values
            for key, value in iter_primary.items():
                ind_var_primary[key].append(value)
            for key, value in iter_secondary.items():
                ind_var_secondary[key].append(value)
        # compute the metrics
        for key, value in ind_var_primary.items():
            primary_metrics[key].append(np.mean(value))
        for key, value in ind_var_secondary.items():
            secondary_metrics[key].append(np.mean(value))
        # end timing
        end = time.time()
        print(f"{GREEN}Finished experiment with {config['experiment']['ind_var']['name']} = {x} took {end - start} seconds{RESET}")

    # Plot figures from metrics, and save them in the plotting folder

    plot_metrics(primary_metrics, config)

    # Save the plot in new experiment folder

    row_labels = [f"Rho: {r}" for r in config['experiment']['ind_var']['vals']]
    primary_df = pd.DataFrame(primary_metrics, index=row_labels)
    secondary_df = pd.DataFrame(secondary_metrics, index=row_labels)

    # Write to csv

    metrics_df = pd.concat([primary_df, secondary_df], axis=1)

    # Save results in a pandas dataframe

    return metrics_df


# TODO
# Finish metrics function
# Finish the rest of the experiment function
# Write README

##########################################################
# Legacy code
##########################################################

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