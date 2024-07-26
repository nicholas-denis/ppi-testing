import numpy as np
import matplotlib.pyplot as plt
import ppi_py
import scipy.stats as stats
import pandas as pd

import yaml
import os
import sys
import argparse
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

def create_metrics_dict(config):
    """
    Parse through config file and create empty metrics dict based off methods and metrics
    """
    metrics = {}
    metrics['estimate'] = []
    metrics['true_parameter'] = []
    metrics['ci_low'] = []
    metrics['ci_high'] = []
    metrics['ci_width'] = []
    metrics['empirical_coverage'] = []
    metrics['desired_coverage'] = []
    metrics['noise'] = []
    metrics['technique'] = []
    metrics['model'] = []
    metrics['iteration'] = []
    metrics['test_error'] = []
    metrics['rectifier'] = []
    metrics['relative_bias'] = []

    if config['experiment']['parameters'].get('model_bias', False):
        metrics['model_bias'] = []

    return metrics

def compute_metrics(config, conf_int):
    """
    Given a confidence interval tuple, computes and returns metrics
    """
    metrics = {} 
    metrics['estimate'] = [conf_int[0]]
    metrics['true_parameter'] = [config['experiment']['parameters'].get('true_value', None)]
    metrics['ci_low'] = [conf_int[1][0]]
    metrics['ci_high'] = [conf_int[1][1]]
    metrics['desired_coverage'] = [config['experiment']['parameters'].get('confidence_level', None)]
    metrics['noise'] = [config['experiment']['parameters'].get('rho', None)]  # Temporary, needs to be changed later
    if config['experiment']['parameters'].get('true_value', None) != 0:
        metrics['relative_bias'] = [(conf_int[0] - config['experiment']['parameters']['true_value'])/config['experiment']['parameters']['true_value']]
    else:
        metrics['relative_bias'] = [np.nan]
    for metric in config['experiment']['metrics']:
        if metric == 'widths':
            metrics['ci_width'] = [conf_int[1][1] - conf_int[1][0]]
        elif metric == 'coverages':
            true_value = [config['experiment']['parameters'].get('true_value', None)]
            if true_value is None:
                raise ValueError("True value not provided for coverage computation")
            metrics['empirical_coverage'] = [1 if true_value >= conf_int[1][0] and true_value <= conf_int[1][1] else 0]
    return metrics


def do_ppi_ci_mean(y_gold, y_gold_fitted, y_fitted, conf, lhat=1.0):
    alpha = 1 - conf
    ci = ppi_py.ppi_mean_ci(y_gold, y_gold_fitted, y_fitted, alpha=alpha, lhat=lhat)
    ci = (ci[0][0], ci[1][0])  # Remove the array
    return ppi_py.ppi_mean_pointestimate(y_gold, y_gold_fitted, y_fitted , lhat=lhat)[0], ci

def do_naive_ci_mean(y_gold, y_gold_fitted, y_fitted, conf):
    y_gold_fitted = y_gold_fitted.reshape(-1, 1)
    y_fitted = y_fitted.reshape(-1, 1)
    concat = np.vstack((y_gold, y_fitted)).flatten()  # Concactenate samples
    naive_theta = np.mean(concat)  # Compute mean
    naive_sigma = np.std(concat)  # Compute std dev
    n_tot = concat.shape[0]
    return naive_theta, stats.norm.interval(conf, loc=naive_theta, scale=naive_sigma/np.sqrt(n_tot))

def do_classical_ci_mean(y_gold, y_gold_fitted, y_fitted, conf):
    small_sample = y_gold.shape[0]
    classical_theta, classical_se = np.mean(y_gold.flatten()), stats.sem(y_gold.flatten())
    h = classical_se * stats.t.ppf((1 + conf) / 2., small_sample-1)  # Highly stolen code, uses t-dist here
    return classical_theta, (classical_theta - h, classical_theta + h)

def ratio_estimator_variance(x_ppi, x_gold, y_gold):
    # sample sizes
    x_ppi = x_ppi.reshape(1, -1)
    x_gold = x_gold.reshape(1, -1)
    y_gold = y_gold.reshape(1, -1)
    n_ppi = x_ppi.shape[1]
    n_gold = x_gold.shape[1]

    # means
    x_ppi_bar = np.mean(x_ppi)
    x_gold_bar = np.mean(x_gold)
    y_gold_bar = np.mean(y_gold)
    r_hat = y_gold_bar / x_gold_bar

    # variances
    x_sample_var = np.var(x_gold, axis=1, ddof=1)
    y_sample_var = np.var(y_gold, axis=1, ddof=1)
    xy_cov = np.sum((x_gold - x_gold_bar) * (y_gold - y_gold_bar)) / (n_gold - 1)

    var = (1 - n_gold / n_ppi) * (1 / n_gold) * (y_sample_var**2 + r_hat**2 * x_sample_var**2 - 2 * r_hat * xy_cov)
    return var

def do_ratio_ci_mean(x_ppi, x_gold, y_gold, conf, dof=1, t_dist=True):
    x_bar_gold = np.mean(x_gold)
    y_bar_gold = np.mean(y_gold)
    x_bar_ppi = np.mean(x_ppi)
    mean_estimate = x_bar_ppi * y_bar_gold / x_bar_gold
    var = ratio_estimator_variance(x_ppi, x_gold, y_gold)

    # Shouldn't use t-distribution for this, as the estimate is generally skewed.
    # See: https://en.wikipedia.org/wiki/Ratio_estimator
    # We use Vysochanskij-Petunin inequality to get a conservative estimate
    # Aka, statistical assumptions are violated. 

    if t_dist:
        lower = mean_estimate - stats.t.ppf(1 - (1 - conf) / 2, dof) * np.sqrt(var)
        upper = mean_estimate + stats.t.ppf(1 - (1 - conf) / 2, dof) * np.sqrt(var)
    else:
        lamb = np.sqrt(4 / (9 * conf))
        lower = lamb * np.sqrt(var) - mean_estimate
        upper = lamb * np.sqrt(var) + mean_estimate
    
    return mean_estimate, (lower[0], upper[0])

def compute_ci_singular(config, y_gold, y_gold_fitted, y_fitted, method, x_ppi=None, x_gold=None):
    """
    Computes confidence interval and estimate for a single method
    """
    method_type = method['type']
    conf = config['experiment']['parameters'].get('confidence_level', 0.9)
    if config['experiment']['estimate'] == 'mean':
        if method_type == 'ppi':
            return do_ppi_ci_mean(y_gold, y_gold_fitted, y_fitted, conf, lhat=1.0)
        elif method_type == 'naive':
            return do_naive_ci_mean(y_gold, y_gold_fitted, y_fitted, conf)
        elif method_type == 'classical':
            return do_classical_ci_mean(y_gold, y_gold_fitted, y_fitted, conf)
        elif method_type == 'ppi_pp':
            return do_ppi_ci_mean(y_gold, y_gold_fitted, y_fitted, conf, lhat=method['lhat'])
        elif method_type == 'stratified_ppi':
            print("Yet to be implemented")
        elif method_type == 'ratio':
            return do_ratio_ci_mean(x_ppi, x_gold, y_gold, conf, t_dist=method.get('t_dist', False))
        elif method_type == 'classical_test':
            return do_ppi_ci_mean(y_gold, y_gold_fitted, y_fitted, conf, lhat=0)
        else:
            print("Method not recognized")
    else:
        print("Estimation method not recognized")

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
    num_methods = len(config['experiment']['methods'])

    x_train, y_train = dist.sample_population(config['experiment']['parameters']['training_population'])
    x_gold, y_gold = dist.sample_population(config['experiment']['parameters']['gold_population'])
    x_ppi, y_ppi = dist.sample_population(config['experiment']['parameters']['unlabelled_population'])

    if config['experiment']['parameters'].get('true_value', None) is None:
        # Do not change this to an if not statement in case 'true_value' is 0
        true_value = np.mean(y_ppi)
        config['experiment']['parameters']['true_value'] = true_value

    x_train, x_test, y_train, y_test = ml.train_test_split(x_train, y_train, test_size=config['experiment']['parameters'].get('test_size', 0.2))

    model = ml.train_model(x_train, y_train, config['experiment']['model'])  # Placeholder

    # Residual testing
    y_test_pred = model.predict(x_test)
    residual = np.mean(np.abs(y_test_pred - y_test))  # Mean abs error

    # Fitting
    y_gold_fitted = model.predict(x_gold)  # Gold standard fitted
    y_fitted = model.predict(x_ppi)  # Unlabelled fitted

    # Manual rectifier computation
    rectifier = np.mean(y_gold_fitted - y_gold)

    # Model bias computation if called for
    if config['experiment']['parameters']['unlabelled_population'].get('include', False):
        model_bias = np.mean(y_fitted - y_ppi)

    # Confidence interval computation

    metrics = create_metrics_dict(config)

    if 'model_bias' in metrics:
        metrics['model_bias'] = [model_bias] * num_methods
    metrics['test_error'] = [residual] * num_methods
    metrics['rectifier'] = [rectifier] * num_methods


    for method in config['experiment']['methods']:
        ci_results = compute_ci_singular(config, y_gold, y_gold_fitted,
                                          y_fitted, method,
                                            x_ppi=x_ppi, x_gold=x_gold)
        method_metrics = compute_metrics(config, ci_results)
        method_metrics['technique'] = [method['type']]
        method_metrics['model'] = [config['experiment']['model'].get('name', None)] 
        metrics = extend_metrics(metrics, method_metrics)

    return metrics

def extend_metrics(metrics_dict, metrics_d):
    """
    Extend the metrics dictionary with the new metrics

    Might not need this anymore

    Args:
    metrics_dict (dict): dictionary of dictionaries of metrics each key is the independent variable
    metrics_d (dict): dictionary of metrics
    """
    for key, value in metrics_d.items():
        if key in metrics_dict:
            metrics_dict[key].extend(value)

    return metrics_dict

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

    metrics = create_metrics_dict(config) 
    ind_var = config['experiment']['ind_var']['name']
    metrics[ind_var] = []

    num_methods = len(config['experiment']['methods'])

    # iterate over independent variables

    print(f"{GREEN}Starting experiment{RESET}")

    # approximate true mean if necessary

    if config['experiment']['parameters'].get('true_value', None) is None:
        pop_dict = config['experiment']['parameters']['gold_population']
        pop_dict['size'] = 100000
        x_sample, y_sample = dist.sample_population(pop_dict)
        config['experiment']['parameters']['true_value'] = np.mean(y_sample)

    for x in config['experiment']['ind_var']['vals']:
        print(f"{YELLOW}Running experiment with {config['experiment']['ind_var']['name']} = {x}{RESET}")
        # begin timing
        start = time.time()
        for path in config['experiment']['ind_var']['paths']:
            keys = path.split('.')  # Split the path
            # update the independent variable, I'm about to update the original config in place.
            temp = config
            for key in keys[:-1]:
                temp = temp[key]
            temp[keys[-1]] = x
        for i in range(params['n_its']):
            # single iteration of the experiment
            iter_metrics = single_iteration(config) 
            iter_metrics['iteration'] = [i] * num_methods
            iter_metrics[ind_var] = [x] * num_methods
            metrics = extend_metrics(metrics, iter_metrics)
        # end timing
        end = time.time()
        print(f"{GREEN}Finished experiment with {config['experiment']['ind_var']['name']} = {x} took {end - start} seconds{RESET}")

    # Plot figures from metrics, and save them in the plotting folder

    #plot_metrics(primary_means, config)

    # Create a dataframe of the metrics

    metrics_df = pd.DataFrame(metrics)

    if config['experiment']['parameters'].get('cut_interval', False):
        metrics_df['ci_low'] = np.maximum(metrics_df['ci_low'], 0)
        metrics_df['ci_high'] = np.maximum(metrics_df['ci_high'], 0)

    return metrics_df


# - Ratio estimator, constructing CI, I have some but not sure if they're the best