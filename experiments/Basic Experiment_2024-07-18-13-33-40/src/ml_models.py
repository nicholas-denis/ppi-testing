import numpy as np
import matplotlib.pyplot as plt
import ppi_py
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import pandas as pd
import optuna
import xgboost as xgb

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

def build_fit_slr(x, y, model_config):
    """
    y_hat = model.predict(x)
    """
    model = LinearRegression()
    model.fit(x, y)
    return model

def build_fit_dt(x, y, model_config):
    """
    y_hat = model.predict(x)
    """
    model = DecisionTreeRegressor()
    model.fit(x ,y)
    return model

def build_fit_rf(x, y, model_config):
    """
    y_hat = model.predict(x)
    """
    model = RandomForestRegressor()
    model.fit(x ,y)
    return model

def build_rf_optuna(x, y, model_config):
    """
    Build a random forest model with optuna hyperparameter optimization
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 150)
        max_depth = trial.suggest_int('max_depth', 1, 32)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        y_train, y_val = y_train.ravel(), y_val.ravel()
        model.fit(x_train, y_train)
        return np.mean((model.predict(x_val) - y_val) ** 2)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    model = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
    y = y.ravel()
    model.fit(x, y)
    return model

def build_xgb(x, y, model_config):
    """
    Build an xgboost model
    """
    model = xgb.XGBRegressor()
    model.fit(x, y)
    return model

def build_xgb_optuna(x, y, model_config):
    """
    Build an xgboost model with optuna hyperparameter optimization
    """
    # turn off logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 150)
        max_depth = trial.suggest_int('max_depth', 1, 32)
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        y_train, y_val = y_train.ravel(), y_val.ravel()
        model.fit(x_train, y_train)
        return np.mean((model.predict(x_val) - y_val) ** 2)

    n_trials = model_config.get('n_trials', 100)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    model = xgb.XGBRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
    y = y.ravel()
    model.fit(x, y)
    return model

def train_model_old(x_train, y_train, model_config):
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
        model = build_fit_slr(x_train, y_train, model_config)
    elif model_config['name'] == 'decision_tree':
        model = build_fit_dt(x_train, y_train, model_config)
    elif model_config['name'] == 'random_forest':
        if model_config.get('optuna', False):
            model = build_rf_optuna(x_train, y_train, model_config)
        else:
            model = build_fit_rf(x_train, y_train, model_config)
    elif model_config['name'] == 'xgboost':
        if model_config.get('optuna', False):
            model = build_xgb_optuna(x_train, y_train, model_config)
        else:
            model = build_xgb(x_train, y_train, model_config)
    else:
        raise ValueError("Model not supported")
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
    # Mapping of model names to their build functions
    model_builders = {
        'linear_regression': build_fit_slr,
        'decision_tree': build_fit_dt,
        'random_forest': build_rf_optuna if model_config.get('optuna', False) else build_fit_rf,
        'xgboost': build_xgb_optuna if model_config.get('optuna', False) else build_xgb,
    }

    # Get the build function based on the model name
    build_function = model_builders.get(model_config['name'])

    if build_function:
        model = build_function(x_train, y_train, model_config)
    else:
        raise ValueError("Model not supported")

    return model
