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

    unlabelled_dist_y = config['experiment']['parameters']['unlabelled_population']['y_population'].get('distribution', '')

    true_value = config['experiment']['parameters'].get('true_value', '')
    experiment_iterations = config['experiment']['parameters'].get('n_its', '')

    confidence_level = config['experiment']['parameters'].get('confidence_level', '')

    ind_var = config['experiment']['ind_var'].get('name', '')

    model = config['experiment']['model'].get('name', '')

    estimate = config['experiment'].get('estimate', '')

    methods = ''
    for method in config['experiment']['methods']:
        methods += method.get('type', ' ') + ', '

    # get results:

    data_summaries = []

    ind_var = config['experiment']['ind_var']['name']

    for x in config['experiment']['ind_var'].get('vals', []):
        ind_var_df = df[df[ind_var] == x]
        avg_test_error = ind_var_df['test_error'].mean()
        error_statement = "The average test error for x = {} is {}".format(x, avg_test_error)
        data_summaries.append(error_statement)
        for method in config['experiment'].get('methods', []):
            # isolate the data for this method and x
            method_ind_var_df = df[(df['technique'] == method) & (df[ind_var] == x)]
            avg_ci_width = method_ind_var_df['ci_width'].mean()
            avg_coverage = method_ind_var_df['empirical_coverage'].mean()

            ci_statement = "The average CI width for x = {} and method {} is {}".format(x, method, avg_ci_width)
            coverage_statement = "The average empirical coverage for x = {} and method {} is {}".format(x, method, avg_coverage)
            data_summaries.append(ci_statement)
            data_summaries.append(coverage_statement)

    # write to experiment folder
    experiment_folder = config['paths']['experiment_path']

    with open(os.path.join(experiment_folder, 'summary.txt'), 'w') as f:
        f.write("Experiment Summary\n")
        f.write("Experiment Name: {}\n".format(experiment_name))
        f.write("Training Population X Distribution: {}\n".format(training_dist_x))
        f.write("Training Population X Size: {}\n".format(training_size))
        f.write("Training Population Y Distribution: {}\n".format(training_dist_y))
        f.write("Gold Population X Distribution: {}\n".format(gold_dist_x))
        f.write("Gold Population X Size: {}\n".format(gold_size))
        f.write("Gold Population Y Distribution: {}\n".format(gold_dist_y))
        f.write("Unlabelled Population X Distribution: {}\n".format(unlabelled_dist_x))
        f.write("Unlabelled Population X Size: {}\n".format(unlabelled_size))
        f.write("Unlabelled Population Y Distribution: {}\n".format(unlabelled_dist_y))
        f.write("True Value: {}\n".format(true_value))
        f.write("Experiment Iterations: {}\n".format(experiment_iterations))
        f.write("Confidence Level: {}\n".format(confidence_level))
        f.write("Independent Variable: {}\n".format(ind_var))
        f.write("Model: {}\n".format(model))
        f.write("Estimate: {}\n".format(estimate))
        f.write("Methods: {}\n".format(methods))
        f.write("\n")
        f.write("Data Summaries\n")
        for summary in data_summaries:
            f.write(summary + '\n')
        f.close()




