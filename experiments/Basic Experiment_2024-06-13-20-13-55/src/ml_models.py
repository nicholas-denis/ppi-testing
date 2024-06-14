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
import matplotlib.pyplot as plt
import datetime
import logging
import warnings
import time


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
