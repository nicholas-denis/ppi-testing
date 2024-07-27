import distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import ppi
import ppi_py
import scipy.stats as stats
import pandas as pd
import ml_models as ml

import yaml
import os
import sys
import argparse
import plotting
import copy

def test(config):
    rho_vals = [10, 100, 1000, 10000, 100000]
    pop_config = config['experiment']['parameters']['gold_population']
    for rho in rho_vals:
        pop_config['y_population']['rho'] = rho
        x_gold, y_gold = dist.sample_population(pop_config)

        print(pop_config['x_population']['size'])

        # print the gold population standard error
        print("Gold Population Standard Error: ", stats.sem(y_gold))

        # plot the x population and the y population
        fig, ax = plt.subplots(2)
        ax[0].hist(x_gold, bins=50)
        ax[0].set_title("X Population")
        ax[1].scatter(x_gold, y_gold, alpha=0.5)
        ax[1].set_title("Y Population")
        plt.show()

        ml_config = config['experiment']['model']

        train_config = copy.deepcopy(pop_config)
        train_config['x_population']['size'] = 10000

        x_train, y_train = dist.sample_population(train_config)

        model = ml.train_model(x_train, y_train, ml_config)

        y_gold_fitted = model.predict(x_gold)

        # plot the gold population and the fitted gold population

        fig, ax = plt.subplots(2)

        ax[0].scatter(x_gold, y_gold, alpha=0.5)
        ax[0].set_title("Gold Population")
        ax[1].scatter(x_gold, y_gold_fitted, alpha=0.5)
        ax[1].set_title("Fitted Gold Population")
        plt.show()
    

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a simulation.')
    parser.add_argument('--config', type=str, help='path to the config file')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    test(config)