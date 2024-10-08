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
import copy

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
    metrics['prediction_variance'] = []
    metrics['true_value'] = []
    metrics['relative_improvement'] = []

    if config['experiment']['parameters'].get('model_bias', False):
        metrics['model_bias'] = []

    if config['experiment'].get('distances', False):
        for distance in config['experiment']['distances']:
            metrics[distance] = []

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
    h = classical_se * stats.t.ppf((1 + conf) / 2., small_sample-1)
    return classical_theta, (classical_theta - h, classical_theta + h)

def do_classical_ci_mean_norm(y_gold, y_gold_fitted, y_fitted, conf):
    """
    Classical CI using normal distribution
    """
    small_sample = y_gold.shape[0]
    classical_theta, classical_se = np.mean(y_gold.flatten()), stats.sem(y_gold.flatten())
    h = classical_se * stats.norm.ppf((1 + conf) / 2.)  # Normal dist
    return classical_theta, (classical_theta - h, classical_theta + h)

def ratio_estimator_variance(x_ppi, x_gold, y_gold):
    """
    An estimate of the variance of the ratio estimator

    Given by: sigma = 1/(n - 1) * sum(y - r_hat * x)^2
    var = (N - n)/N * sigma^2 / n 
    
    Here: N = n_ppi, n = n_gold * (x_ppi_bar/x_gold_bar)^2

    This is a conservative estimate, and requires n to be somewhat large.

    If it is known that x_ppi_bar = x_gold_bar, one can remove the x_ppi_bar/x_gold_bar term, however in practice, this is not the case.
    """
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
    sigma_sq = np.sum((y_gold - r_hat * x_gold) ** 2) / (n_gold - 1)
    var = (1 - n_gold / n_ppi) * sigma_sq / n_gold * (x_ppi_bar / x_gold_bar) ** 2

    return var

def do_ratio_ci_mean(x_ppi, x_gold, y_gold, conf, dof=1, t_dist=True):
    x_bar_gold = np.mean(x_gold)
    y_bar_gold = np.mean(y_gold)
    x_bar_ppi = np.mean(x_ppi)
    mean_estimate = x_bar_ppi * y_bar_gold / x_bar_gold
    var = ratio_estimator_variance(x_ppi, x_gold, y_gold)

    n_gold = x_gold.shape[0]

    # Shouldn't use t-distribution for this, as the estimate is generally skewed.
    # See: https://en.wikipedia.org/wiki/Ratio_estimator
    # We use Vysochanskij-Petunin inequality to get a conservative estimate
    # Aka, statistical assumptions are violated. 

    if t_dist:
        lower = mean_estimate - stats.t.ppf((1 + conf) / 2, n_gold) * np.sqrt(var)
        upper = mean_estimate + stats.t.ppf((1 + conf) / 2, n_gold) * np.sqrt(var)
    else:
        lamb = np.sqrt(4 / (9 * conf))
        lower = lamb * np.sqrt(var) - mean_estimate
        upper = lamb * np.sqrt(var) + mean_estimate
    
    return mean_estimate, (lower, upper)

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
            return do_ppi_ci_mean(y_gold, y_gold_fitted, y_fitted, conf, lhat=None)
        elif method_type == 'stratified_ppi':
            print("Yet to be implemented")
        elif method_type == 'ratio':
            return do_ratio_ci_mean(x_ppi, x_gold, y_gold, conf, t_dist=method.get('t_dist', False))
        elif method_type == 'classical_test':
            return do_classical_ci_mean_norm(y_gold, y_gold_fitted, y_fitted, conf)
        else:
            print("Method not recognized")
    else:
        print("Estimation method not recognized")

def single_iteration(config, iter_num=0, model_dict=None):
    """
    A single iteration of the experiment

    Experiment components:
    - Sampling
    - Reshaping for scikitlearn
    - Training
    - Residual testing
    """
    
    num_methods = len(config['experiment']['methods'])

    # print(model_dict)

    if model_dict is None:
        x_train, y_train = dist.sample_population(config['experiment']['parameters']['training_population'])
        x_train, x_test, y_train, y_test = ml.train_test_split(x_train, y_train, test_size=config['experiment']['parameters'].get('test_size', 0.2))
        max_train_x = x_train.max()
        min_train_x = x_train.min()
        model = ml.train_model(x_train, y_train, config['experiment']['model'])
        # Residual testing
        y_test_pred = model.predict(x_test)
        residual = np.mean(np.abs(y_test_pred - y_test))  # Mean abs error
    else:
        model = model_dict['model']
        x_train = model_dict['x_train']
        y_train = model_dict['y_train']
        x_test = model_dict['x_test']
        y_test = model_dict['y_test']
        max_train_x = model_dict['max_train_x']
        min_train_x = model_dict['min_train_x']
        residual = model_dict['residual']

    x_gold, y_gold = dist.sample_population(config['experiment']['parameters']['gold_population'])
    x_ppi, y_ppi = dist.sample_population(config['experiment']['parameters']['unlabelled_population'])

    # Fitting
    y_gold_fitted = model.predict(x_gold)  # Gold standard fitted
    y_fitted = model.predict(x_ppi)  # Unlabelled fitted

    # Manual rectifier, variance computation
    rectifier = np.var(y_gold_fitted - y_gold)
    predictions_var = np.var(y_fitted)

    if config['experiment'].get('clipping', False):
        # remove all x values in x_ppi that are outside the range and their corresponding y values
        x_ppi_clip = x_ppi[(x_ppi >= min_train_x) & (x_ppi <= max_train_x)]
        y_ppi_clip = y_ppi[(x_ppi >= min_train_x) & (x_ppi <= max_train_x)]

        # do the same for x_gold if config says so
        if config['experiment'].get('remove_gold', False):
            x_gold_clip = x_gold[(x_gold >= min_train_x) & (x_gold <= max_train_x)]
            y_gold_clip = y_gold[(x_gold >= min_train_x) & (x_gold <= max_train_x)]
            # check if either is nearly empty
            if (x_ppi_clip.shape[0] >= 5 and x_gold_clip.shape[0] >= 5):
                x_ppi, y_ppi = np.array(x_ppi_clip).reshape(-1, 1), np.array(y_ppi_clip).reshape(-1, 1)
                x_gold, y_gold = np.array(x_gold_clip).reshape(-1, 1), np.array(y_gold_clip).reshape(-1, 1)
            else:
                # log clipping removed too many points:
                # logging.warning(f"Iteration {iter_num}: Clipping removed too many points, reverting to original data")
                pass
        else:
            # check if either is empty
            if  (x_ppi_clip.shape[0] >= 5):
                x_ppi, y_ppi = np.array(x_ppi_clip).reshape(-1, 1), np.array(y_ppi_clip).reshape(-1, 1)
            else:
                # log clipping removed too many points:
                # log in the file that is referred to by logging_path
                logger = logging.getLogger('main')
                logger.warning(f"Iteration {iter_num}: Clipping removed too many points, reverting to original data")
                pass


    if config['experiment']['parameters'].get('true_value', None) is None:
        # Do not change this to an if not statement in case 'true_value' is 0
        true_value = np.mean(y_ppi)
        config['experiment']['parameters']['true_value'] = true_value


    # Model bias computation if called for
    if config['experiment']['parameters']['unlabelled_population'].get('include', False):
        model_bias = np.mean(y_fitted - y_ppi)

    # Confidence interval computation

    metrics = create_metrics_dict(config)

    if 'model_bias' in metrics:
        metrics['model_bias'] = [model_bias] * num_methods
    metrics['test_error'] = [residual] * num_methods
    metrics['rectifier'] = [rectifier] * num_methods
    metrics['prediction_variance'] = [predictions_var] * num_methods

    for method in config['experiment']['methods']:
        ci_results = compute_ci_singular(config, y_gold, y_gold_fitted,
                                          y_fitted, method,
                                            x_ppi=x_ppi, x_gold=x_gold)
        method_metrics = compute_metrics(config, ci_results)
        if method['type'] == 'classical':
            classical_width = ci_results[1][1] - ci_results[1][0]
        tech_width = ci_results[1][1] - ci_results[1][0]
        method_metrics['relative_improvement'] = [(classical_width - tech_width) / classical_width]
        method_metrics['technique'] = [method['type']]
        method_metrics['model'] = [config['experiment']['model'].get('name', None)] 
        metrics = extend_metrics(metrics, method_metrics)

    return metrics

def extend_metrics(metrics_dict, metrics_d):
    """
    Extend the metrics dictionary with the new metrics

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
    
    for ind_var in config['experiment']['ind_var']['name']:
        metrics[ind_var] = []

    num_methods = len(config['experiment']['methods'])

    # iterate over independent variables

    print(f"{GREEN}Starting experiment{RESET}")

    # approximate true mean if necessary

    if config['experiment']['parameters'].get('true_value', None) is None:
        pop_dict = copy.deepcopy(config['experiment']['parameters']['gold_population'])
        pop_dict['x_population']['size'] = 10000
        x_sample, y_sample = dist.sample_population(pop_dict)
        config['experiment']['parameters']['true_value'] = np.mean(y_sample)

    # plot the distributions if necessary
    if config['experiment'].get('plot_distributions', False):
        if config['experiment'].get('dist_family', None) == 'gamma':
            lin_space = np.linspace(0, 40, 4000)
            for vals in config['experiment']['ind_var']['vals']:
                for key in vals.keys():
                    if key == 'alpha':
                        alpha = vals[key]
                    elif key == 'beta':
                        beta = vals[key]
                y = stats.gamma.pdf(lin_space, a=alpha, scale=beta)
                plt.plot(lin_space, y, label=f"Alpha: {alpha}, Beta: {beta}")
                plt.legend()

            # save plot
            plt.savefig(os.path.join(config['paths']['plotting_path'], "distplots.png"), bbox_inches='tight')

            plt.close()

        elif config['experiment'].get('dist_family', None) == 'normal':
            lin_space = np.linspace(-20, 20, 4000)
            for vals in config['experiment']['ind_var']['vals']:
                for key in vals.keys():
                    if key == 'mean':
                        mean = vals[key]
                    elif key == 'std':
                        std = vals[key]
                y = stats.norm.pdf(lin_space, loc=mean, scale=std)
                plt.plot(lin_space, y, label=f"Mean: {mean}, Std: {std}")
                plt.legend()

            # save plot
            plt.savefig(os.path.join(config['paths']['plotting_path'], "distplots.png"), bbox_inches='tight')

            plt.close()
        
        else:
            print(f"{RED}Distribution family not recognized{RESET}")
            

    for collection in config['experiment']['ind_var']['vals']:
        ind_vars_str = ""
        for x in collection.keys():  # Looks at alpha or beta
            ind_vars_str += f"{x} = {collection[x]} "
            for path in config['experiment']['ind_var']['paths'][x]:
                keys = path.split('.')  # Split the path
                # update the independent variable, I'm about to update the original config in place.
                temp = config
                for key in keys[:-1]:
                    temp = temp[key]
                temp[keys[-1]] = collection[x]

        # compute new true value if necessary
        if config['experiment'].get('varying_true_value', None):
            pop_dict = copy.deepcopy(config['experiment']['parameters']['gold_population'])
            pop_dict['x_population']['size'] = 10000
            x_sample, y_sample = dist.sample_population(pop_dict)
            config['experiment']['parameters']['true_value'] = np.mean(y_sample)

        # Compute distribution distances

        train_pop_copy = copy.deepcopy(config['experiment']['parameters']['training_population'])
        train_pop_copy['x_population']['size'] = 100000
        gold_pop_copy = copy.deepcopy(config['experiment']['parameters']['gold_population'])
        gold_pop_copy['x_population']['size'] = 100000
        train_x_sample, train_y_sample = dist.sample_population(train_pop_copy)
        gold_x_sample, gold_y_sample = dist.sample_population(gold_pop_copy)

        for distance in config['experiment'].get('distances', []):
            if distance == 'tv':
                tv_distance = dist.total_variation_distance(train_x_sample, gold_x_sample)
                print(f"{YELLOW}Total Variation Distance: {tv_distance}{RESET}")
            elif distance == 'wasserstein':
                wasserstein_distance = dist.wasserstein_distance(train_x_sample, gold_x_sample)
                print(f"{YELLOW}Wasserstein Distance: {wasserstein_distance}{RESET}")
        print(f"{YELLOW}Running experiment with {ind_vars_str}{RESET}")

        # A very bandaid fix, because I don't want to keep rearranging my config files (also good for the future in case...
        # ...someone else makes the same mistake)

        methods_list = config['experiment']['methods']
        # move classical to the front
        # this is needed to be done because we need to compute relative improvement over classical
        # so classical needs to be computed first
        for i, method in enumerate(methods_list):
            if method['type'] == 'classical':
                methods_list.insert(0, methods_list.pop(i))

        # begin timing
        start = time.time()
        # print current time
        print(datetime.datetime.now())

        model_dict = None

        if config['experiment'].get('train_once', False):
            x_train, y_train = dist.sample_population(config['experiment']['parameters']['training_population'])
            x_train, x_test, y_train, y_test = ml.train_test_split(x_train, y_train, test_size=config['experiment']['parameters'].get('test_size', 0.2))
            model = ml.train_model(x_train, y_train, config['experiment']['model'])
            if config['experiment'].get('clipping', False):
                max_train_x = x_train.max()
                min_train_x = x_train.min()
            else:
                max_train_x = None
                min_train_x = None
            y_test_pred = model.predict(x_test)
            residual = np.mean(np.abs(y_test_pred - y_test))  # Mean abs error

            model_dict = {'model': model, 'x_train': x_train, 'y_train': y_train,
                           'x_test': x_test, 'y_test': y_test,
                             'max_train_x': max_train_x, 'min_train_x': min_train_x, 
                             'residual': residual}

        for i in range(params['n_its']):

            # single iteration of the experiment
            iter_metrics = single_iteration(config, iter_num=i, model_dict=model_dict)
            iter_metrics['iteration'] = [i] * num_methods
            iter_metrics['true_value'] = [config['experiment']['parameters']['true_value']] * num_methods

            # this is not a very smart way of doing it, but it works, also not very modular
            if 'tv' in config['experiment'].get('distances', []):
                iter_metrics['tv'] = [tv_distance] * num_methods
            if 'wasserstein' in config['experiment'].get('distances', []):
                iter_metrics['wasserstein'] = [wasserstein_distance] * num_methods
            for x in collection.keys():
                iter_metrics[x] = [collection[x]] * num_methods
            metrics = extend_metrics(metrics, iter_metrics)
            
        # end timing
        end = time.time()
        print(f"{GREEN}Finished experiment with {ind_vars_str}{RESET} in {end - start} seconds")

    # Create a dataframe of the metrics

    # print length of metric lists

    metrics_df = pd.DataFrame(metrics)

    if config['experiment']['parameters'].get('cut_interval', False):
        metrics_df['ci_low'] = np.maximum(metrics_df['ci_low'], 0)
        metrics_df['ci_high'] = np.maximum(metrics_df['ci_high'], 0)

    return metrics_df
