import numpy as np
import matplotlib.pyplot as plt
import ppi_py
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import yaml
import os
import argparse
import datetime

# colored text
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'



# Helper functions



# Build paths
def build_paths(config: dict):
    """
    summary:
    build paths for this experiment

    args:
    config: dict, config file

    returs:
    config: dict, updated config file
    
    """

    # get from config experiment name
    experiment_name = config.get('experiment', {}).get('name')
    
    # add a time stamp to experiment_name
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = f"{experiment_name}_{time_stamp}"


    # get the experiments path
    experiments_path = config.get('paths', {}).get('experiments_path')

    # check if it exists, if not create
    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)
        print(f"{YELLOW}Created experiments path: {experiments_path}{RESET}")

    # let's create a folder for THIS experiment
    experiment_path = os.path.join(experiments_path, experiment_name)

    config['paths']['experiment_path'] = experiment_path

    # check if it exists, if not create
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        print(f"{YELLOW}Created experiment path: {experiment_path}{RESET}")
    
    # create logging folder
    if not os.path.exists(os.path.join(experiment_path, 'logs')):
        os.makedirs(os.path.join(experiment_path, 'logs'))
        print(f"{YELLOW}Created logging folder{RESET}")

    # create plotting folder
    if not os.path.exists(os.path.join(experiment_path, 'plots')):
        os.makedirs(os.path.join(experiment_path, 'plots'))
        print(f"{YELLOW}Created plotting folder{RESET}")

    return config



# Create logger

def create_logger(config: dict):
    pass



# Main function

def main(config: dict):
    """
    summary:
    main function

    args:
    config: dict, config file

    returns:
    None

    """

    # Build paths
    config = build_paths(config)

    # Create logger




    return