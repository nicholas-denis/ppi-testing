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

# y sampling functions

def sample_y_linear_old(x, m, beta, rho):
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

def sample_y_linear(x, params_dict):
    """
    Same as sample_y_linear_old, but with no beta parameter, and b is the intercept

    If you want the hold one, just set m = m_old * beta
    """
    sigma_squared = abs(params_dict['m'])*(1.-(params_dict['rho']**2))/(params_dict['rho']**2) 

    # sample noise (e_i)
    sampled_e = np.random.normal(loc=0,
                                scale=sigma_squared*x,
                                size=x.shape
                                )
    
    # compute y as a linear function of x
    y = params_dict['m']*x + sampled_e + params_dict['b']
    
    #except:
    #    raise KeyError("Incorrect parameters for linear transformation, required: m, rho, x, b")

    return y

def sample_y_squared(x, params_dict):
    return sample_y_linear(x, params_dict)**2

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

def optimal_transport_distance(x, y, cost):
    """
    Compute the optimal transport distance between two distributions using
    monte carlo methods
    
    Big todo I guess, will read paper first
    """
    pass


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
    }

    transformations = {
        'linear_univariate': sample_y_linear,
        'linear_multivariate': sample_y_linear_mv,
        'linear_mult_noise': sample_y_linear_mult_noise,
        'linear_squared': sample_y_squared,
        'linear_mult_noise_squared': sample_y_linear_mult_noise_mv_squared,
    }

    x_dict = population_dict['x_population']
    y_dict = population_dict['y_population']
    # Sample x
    # Sample x
    if x_dict['distribution'] in distributions:
        x = distributions[x_dict['distribution']](x_dict).reshape(-1, 1)
    else:
        raise ValueError("Distribution not supported")

    # Sample y based on x
    if y_dict['transformation'] in transformations:
        y = transformations[y_dict['transformation']](x, y_dict).reshape(-1, 1)
    else:
        raise ValueError("Transformation not supported")

    return x, y
