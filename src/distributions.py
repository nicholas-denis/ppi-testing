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


# x sampling functions

def sample_gamma_univariate(params_dict):
    try:
        x = np.random.gamma(params_dict['alpha'], params_dict['beta'], params_dict['size']).reshape(-1, 1)
    except:
        raise KeyError("Incorrect parameters for gamma distribution, required: alpha, beta, size")
    return x

# y sampling functions

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
    y = m*x + b + sampled_e

    return y

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