import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import pandas as pd

import yaml
import os
import sys
import argparse


# x sampling functions

def sample_gamma(params_dict):
    try:
        x = np.random.gamma(params_dict['alpha'], params_dict['beta'], params_dict['size']).reshape(-1, 1)
    except:
        raise KeyError("Incorrect parameters for gamma distribution, required: alpha, beta, size")
    return x

def sample_gamma_mv(params_dict):
    try:
        x = np.random.gamma(params_dict['alpha'], params_dict['beta'], (params_dict['size'], params_dict['n_features']))
    except:
        raise KeyError("Incorrect parameters for gamma distribution, required: alpha, beta, size, n_features")
    return x

def sample_normal(params_dict):
    try:
        x = np.random.normal(params_dict['mean'], params_dict['std'], params_dict['size']).reshape(-1, 1)
    except:
        raise KeyError("Incorrect parameters for normal distribution, required: loc, scale, size")
    return x

# y sampling functions

def sample_y_linear(x, params_dict):
    """
    Simple linear transformation of x, with gaussian noise
    """
    std = params_dict.get('std', 1)
    rho = params_dict.get('rho', 1)
    m = params_dict.get('m', 1)
    b = params_dict.get('b', 0)
    # sample noise (e_i)
    sampled_e = np.random.normal(loc=0,
                                scale=rho*m*std,
                                size=x.shape
                                )
    
    # compute y as a linear function of x
    y = m*x + sampled_e + b
    
    #except:
    #    raise KeyError("Incorrect parameters for linear transformation, required: m, rho, x, b")

    return y

def sample_y_linear_gamma(x, params_dict):
    """
    Sample y as a linear function of x, with noise depending on gamma
    """
    alpha = params_dict['alpha']
    beta = params_dict['beta']
    m = params_dict['m']
    rho = params_dict['rho']

    sigma_squared = m**2 * alpha * beta**2 * ((1 - rho)/rho)

    sample_e = np.random.normal(loc=0,
                                scale=np.sqrt(sigma_squared),
                                size=x.shape
                                )
    
    y = m * x + sample_e

    return y

def sample_y_squared_gamma(x, params_dict):
    """
    Sample y as a linear function of x, with noise depending on gamma
    """
    alpha = params_dict['alpha']
    beta = params_dict['beta']
    m = params_dict['m']
    rho = params_dict['rho']

    sample_e = np.random.normal(loc=0,
                                scale=np.sqrt(rho),
                                size=x.shape
                                )
    
    
    y = m * x**2 + sample_e

    return y

def sample_y_squared(x, params_dict):
    """
    Sample y as a linear function of x, with noise depending on gamma
    """
    m = params_dict.get('m', 1)
    rho = params_dict.get('rho', 1)
    std = params_dict.get('std', 1)

    sample_e = np.random.normal(loc=0,
                                scale=rho * m * 10 * std,
                                size=x.shape
                                )
    
    y = m * x**2 + sample_e

    return y

def sample_y_linear_mv(x, y_dict):
    try:
        sigma_squared = abs(y_dict['m'])*(1.-(y_dict['rho']**2))/(y_dict['rho']**2)

        # sample noise (e_i)
        sampled_e = np.random.normal(loc=0,
                                     scale=sigma_squared,
                                     size=(x.shape[0], 1)
                                     )

        # compute y as a linear function of x
        y = x @ y_dict['vec'] + sampled_e

        return y
    except:
        raise KeyError("Incorrect parameters for linear transformation, required: m, rho, vec")


def sample_y_linear_mult_noise(x, params_dict):
    """
    Same as sample_y_linear, but with multiplicative noise terms
    """
    sigma_squared = abs(params_dict['m'])*(1.-(params_dict['rho']**2))/(params_dict['rho']**2) 

    # sample noise (e_i)
    sampled_e = np.random.normal(loc=1,
                                scale=sigma_squared*x,
                                size=x.shape
                                )
    
    # compute y as a linear function of x
    y = params_dict['m']* x * sampled_e  # Can still get negative values.
    
    #except:
    #    raise KeyError("Incorrect parameters for linear transformation, required: m, rho, x, b")

    return y

def sample_y_linear_mult_noise_mv_squared(x, y_dict):
    return sample_y_linear_mult_noise(x, y_dict)**2

# Distribution distance functions

def total_variation_distance(x, y, bins=100):
    """
    Compute the total variation distance between two distributions using
    monte carlo methods

    Args:
    x (np.array): x
    y (np.array): y
    """
    # Compute the histogram for x
    hist_x, bins_x = np.histogram(x, bins, density=True)
    hist_y, bins_y = np.histogram(y, bins, density=True)

    # Compute the total variation distance
    tvd = np.max(np.abs(hist_x - hist_y))  # can be either max or abs

    return tvd

def wasserstein_distance(x, y, cost=None):
    """
    Compute the optimal transport distance between two distributions using
    monte carlo methods
    
    Will implement cost later.
    """
    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x.flatten()
            y = y.flatten()
            return stats.wasserstein_distance(x, y)
        else:
            return stats.wasserstein_distance_nd(x, y)
    else:
        return stats.wasserstein_distance(x, y)



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
    distributions = {
    'gamma_univariate': sample_gamma,
    'gamma_multivariate': sample_gamma_mv,
    'normal_univariate': sample_normal,
    }

    transformations = {
        'linear_univariate': sample_y_linear,
        'linear_multivariate': sample_y_linear_mv,
        'linear_mult_noise': sample_y_linear_mult_noise,
        'linear_squared': sample_y_squared,
        'linear_mult_noise_squared': sample_y_linear_mult_noise_mv_squared,
        'linear_gamma': sample_y_linear_gamma,
        'linear_gamma_squared': sample_y_squared_gamma,
    }

    x_dict = population_dict['x_population']
    y_dict = population_dict['y_population']
    # Sample x
    if x_dict['distribution'] in distributions:
        x = distributions[x_dict['distribution']](x_dict).reshape(-1, 1)
    else:
        raise ValueError("Distribution not supported")
 
    # Sample y
    if y_dict['transformation'] in transformations:
        y = transformations[y_dict['transformation']](x, y_dict).reshape(-1, 1)
    else:
        raise ValueError("Transformation not supported")

    return x, y
