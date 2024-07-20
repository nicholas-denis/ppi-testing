import pandas as pd
import numpy as np
import yaml

import os
import sys

def summarize(df, config):
    """
    Take the dataframe and config and summarize the data, write into a .txt file
    """

    # get the important parameters

    experiment_name = config['experiment']['name']

    training_dist_x = config['experiment']['parameters']['training_population']['x_population'].get('distribution', '')
    training_size = config['experiment']['parameters']['training_population']['x_population'].get('size', '')
    
    training_dist_y = config['experiment']['parameters']['training_population']['y_population'].get('distribution', '')

    gold_dist_x = config['experiment']['parameters']['gold_population']['x_population'].get('distribution', '')
    gold_size = config['experiment']['parameters']['gold_population']['x_population'].get('size', '')

    gold_dist_y = config['experiment']['parameters']['gold_population']['y_population'].get('distribution', '')

    unlabelled_dist_x = config['experiment']['parameters']['unlabelled_population']['x_population'].get('distribution', '')
    unlabelled_size = config['experiment']['parameters']['unlabelled_population']['x_population'].get('size', '')

    unlabelled_dist_y = config['data']['parameters']['unlabelled_population']['y_population'].get('distribution', '')

    true_value = config['experiment']['parameters'].get('true_value', '')
    experiment_iterations = config['experiment']['parameters'].get('n_its', '')

    confidence_level = config['experiment']['parameters'].get('confidence_level', '')

    ind_var = config['experiment']['ind_var'].get('name', '')

    model = config['experiment']['model'].get('name', '')

    estimate = config['experiment'].get('estimate', '')

    methods = ''
    for method in config['experiment'].get('methods', []):
        methods += method + ', '

    # get results:

    data_summaries = []

    for x in config['experiment']['ind_var'].get('vals', []):
        for method in config['experiment'].get('methods', []):
            # isolate the data for this method and x
            data = df[(df['method'] == method) & (df['ind_var'] == x)]

