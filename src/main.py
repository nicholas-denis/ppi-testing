import numpy as np
import matplotlib.pyplot as plt
import ppi_py
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import yaml
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging

# colored text
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'


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
        print(f"{YELLOW}Created logging folder: {os.path.join(experiment_path, 'logs')}{RESET}")
        # write the logging folder path to the config file
        config['paths']['logging_path'] = os.path.join(experiment_path, 'logs')


    # create plotting folder
    if not os.path.exists(os.path.join(experiment_path, 'plots')):
        os.makedirs(os.path.join(experiment_path, 'plots'))
        print(f"{YELLOW}Created plotting folder: {os.path.join(experiment_path, 'plots')}{RESET}")
        # write the plotting folder path to the config file
        config['paths']['plotting_path'] = os.path.join(experiment_path, 'plots')
    


    return config

def create_logger(config: dict):
    """
    Create a logger file
    """
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(os.path.join(config['paths']['logging_path'], 'log.log'), 'w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

def main(config: dict):
    
    # create paths
    config = build_paths(config)

    # create logger
    logger = create_logger(config)

    # copy the config file to the experiment folder
    with open(os.path.join(config['paths']['experiments_path'], 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    # do the ppi experiment


    # save results to disk

    return


if __name__ == '__main__':
    # arg parser
    parser = argparse.ArgumentParser(description='Main function')
    # add argument for config file path
    parser.add_argument('--config', 
                        type=str, 
                        default='../configs/config.yaml', 
                        help='Path to config file')

    # print argparser
    args = parser.parse_args()
    print(f"{RED}args: {CYAN}{args}{RESET}")

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    print(f"{RED}config: {CYAN}{config}{RESET}")
    print(f"{RED}type(config): {CYAN}{type(config)}{RESET}")

    paths = config.get('paths')
    print(f"{RED}paths: {CYAN}{paths}{RESET}")
    config['Nick_Huang'] = 'Nick Huang'

    # print configs again
    print(f"{RED}config: {CYAN}{config}{RESET}")
    main(config)

    print(f"{MAGENTA}All done! {RESET}")