import yaml
import plotting
import os
import pandas as pd
import numpy as np
import argparse

# Buggy right now: experiment folder must be in the experiment folder and not in a subfolder.

def plot_folder(folder_path):
    """
    Reruns the plotting function with the data and config file
    """
    # load the config file
    with open(os.path.join(folder_path, 'experiment_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    
    # load the data
    data = pd.read_csv(os.path.join(folder_path, 'results/results.csv'))

    plotting.plot_results(data, config)
    
    return

def plot_using_config_data(config, data):
    """
    Reruns the plotting function with the data and config file

    Use this if you do not to prefer to edit the config file in the experiment folder
    """
    plotting.plot_results(data, config)
    
    return

def main(folder_path = None, config = None, data = None):
    if folder_path is not None and (config is not None or data is not None):
        raise ValueError("Either folder_path or config and data must be provided, not both") 
    if folder_path is not None:
        plot_folder(folder_path)
        return
    if config is not None and data is not None:
        plot_using_config_data(config, data)
        return
    else:
        raise ValueError("Either folder_path or config and data must be provided")
    
    return

if __name__ == '__main__':
    # arg parser
    parser = argparse.ArgumentParser(description='Main function')
    # add argument for config file path
    parser.add_argument('--folder_path', 
                        type=str, 
                        default=None, 
                        help='Path to folder with experiment config and results')
    parser.add_argument('--config', 
                        type=str, 
                        default=None, 
                        help='Path to config file')
    parser.add_argument('--data', 
                        type=str, 
                        default=None, 
                        help='Path to data file')

    # print argparser
    args = parser.parse_args()
    print(f"args: {args}")

    main(folder_path = args.folder_path, config = args.config, data = args.data)