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

def compute_metrics_old(metric_list, estimates_d, config):
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
    for metric in config['experiment']['metrics']:
        if metric == 'widths':
            metrics['ci_width'] = [conf_int[1][1] - conf_int[1][0]]
        elif metric == 'coverages':
            true_value = [config['experiment']['parameters'].get('true_value', None)]
            if true_value is None:
                raise ValueError("True value not provided for coverage computation")
            metrics['empirical_coverage'] = [1 if true_value >= conf_int[1][0] and true_value <= conf_int[1][1] else 0]

    return metrics


def do_ppi_ci_mean(y_gold, y_gold_fitted, y_fitted, conf, lam=1.0):
    alpha = 1 - conf
    ci = ppi_py.ppi_mean_ci(y_gold, y_gold_fitted, y_fitted, alpha=alpha)
    ci = (ci[0][0], ci[1][0])  # Remove the array
    return ppi_py.ppi_mean_pointestimate(y_gold, y_gold_fitted, y_fitted)[0], ppi_py.ppi_mean_ci(y_gold, y_gold_fitted, y_fitted, alpha=alpha)

def do_naive_ci_mean(y_gold, y_gold_fitted, y_fitted, conf):
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


def compute_ci(config, y_gold, y_gold_fitted, y_fitted):
    """
    Computes confidence interval and estimate.

    Currently only computes 1 estimate, but hopefully will compute multiple in the future

    return: tuple of (ppi_theta, ppi_theta_ci, naive_theta, naive_theta_ci, classical_theta, classical_ci)
    """
    d = {}
    if config['experiment']['estimate'] == 'mean':
        conf = config['experiment']['parameters'].get('confidence', 0.9)

        for method in config['experiment']['methods']:
            if method['type'] == 'ppi':
                # PPI CI
                ppi_theta, ppi_theta_ci = do_ppi_ci_mean(y_gold, y_gold_fitted, y_fitted, conf)
                d['ppi'] = (ppi_theta, ppi_theta_ci)
            elif method['type'] == 'naive':
                # Naive CI
                naive_theta, naive_theta_ci = do_naive_ci_mean(y_gold, y_gold_fitted, y_fitted, conf)
                d['naive'] = (naive_theta, naive_theta_ci)
            elif method['type'] == 'classical':
                # Classical CI
                classical_theta, classical_ci = do_classical_ci_mean(y_gold, y_gold_fitted, y_fitted, conf)
                d['classical'] = (classical_theta, classical_ci)
            elif method['type'] == 'ppi_pp':
                # PPI++ CI
                ppi_theta, ppi_theta_ci = do_ppi_ci_mean(y_gold, y_gold_fitted, y_fitted, conf, lam=method['lam'])
                d['ppi_pp'] = (ppi_theta, ppi_theta_ci)
            elif method['type'] == 'stratified_ppi':
                print("Yet to be implemented")
            else:
                print("Method not recognized")
    
    return d

def compute_ci_singular(config, y_gold, y_gold_fitted, y_fitted, method):
    """
    Computes confidence interval and estimate for a single method
    """
    conf = config['experiment']['parameters'].get('confidence', 0.9)
    if config['experiment']['estimate'] == 'mean':
        if method == 'ppi':
            return do_ppi_ci_mean(y_gold, y_gold_fitted, y_fitted, conf)
        elif method == 'naive':
            return do_naive_ci_mean(y_gold, y_gold_fitted, y_fitted, conf)
        elif method == 'classical':
            return do_classical_ci_mean(y_gold, y_gold_fitted, y_fitted, conf)
        elif method == 'ppi_pp':
            return do_ppi_ci_mean(y_gold, y_gold_fitted, y_fitted, conf, lam=config['experiment']['parameters']['lam'])
        elif method == 'stratified_ppi':
            print("Yet to be implemented")
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
    x_train, y_train = dist.sample_population(config['experiment']['parameters']['training_population'])
    x_gold, y_gold = dist.sample_population(config['experiment']['parameters']['gold_population'])
    x_ppi, y_ppi = dist.sample_population(config['experiment']['parameters']['unlabelled_population'])

    x_train, x_test, y_train, y_test = ml.train_test_split(x_train, y_train, test_size=config['experiment']['parameters'].get('test_size', 0.2))

    model = ml.train_model(x_train, y_train, config['experiment']['model'])  # Placeholder

    # Residual testing
    y_test_pred = model.predict(x_test)
    residual = np.mean(np.abs(y_test_pred - y_test))  # Mean abs error

    # Fitting
    y_gold_fitted = model.predict(x_gold)  # Gold standard fitted
    y_fitted = model.predict(x_ppi)  # Unlabelled fitted

    # Manual rectifier computation
    rectifier = np.mean(y_gold_fitted - y_ppi)

    # True bias computation if called for
    if config['experiment']['parameters']['unlabelled_population'].get('include', False):
        true_bias = np.mean(y_fitted - y_ppi)

    # Confidence interval computation

    metrics = create_metrics_dict(config)

    for method in config['experiment']['methods']:
        ci_results = compute_ci_singular(config, y_gold, y_gold_fitted, y_fitted, method['type'])
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

def plot_metrics(metrics_dict, config):
    """
    Plot the metrics from the experiment, and saves them in the plotting folder

    Add alpha (transparency values)

    Include a few more plots.

    Look at violin plots

    Error bars

    CI Plots

    Biases

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


def create_metrics_dict_old(config):
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

    return metrics

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
            # Copilot code, I don't know if this is correct. It should be.
            # Assign a new value to the last key
            temp[keys[-1]] = x
        for i in range(params['n_its']):
            # single iteration of the experiment
            iter_metrics = single_iteration(config) 
            iter_metrics['iteration'] = [i] * num_methods
            iter_metrics[ind_var] = [x] * (params['n_its'] * num_methods)
            print(iter_metrics)
            metrics = extend_metrics(metrics, iter_metrics)
        # end timing
        end = time.time()
        print(f"{GREEN}Finished experiment with {config['experiment']['ind_var']['name']} = {x} took {end - start} seconds{RESET}")

    # Plot figures from metrics, and save them in the plotting folder

    #plot_metrics(primary_means, config)

    # Create a dataframe of the metrics

    metrics_df = pd.DataFrame(metrics)

    print(metrics_df)

    return metrics_df

# TODO
# Change the plotting to be done after everything is put in the dataframe
# Also, add a plotting section to config

# 1: PPI++ implementation DONE
# 2: Check if .tocsv is a quick fix DONE
# 3: Read the experiments email to make sure if code can run the experiments needed.