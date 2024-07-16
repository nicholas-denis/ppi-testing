import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os

def line_plot(data, plot_config, config, x=None):
    """
    summary:
    plot a line plot

    plot_config will control which columns in the data will be plotted

    params:
    data: pd dataframe
    plot_config: dict, plot configuration
    config: dict, general experiment config file
    """
    if x:
        for y in plot_config['y']:
            plt.plot(x, data[y], label=plot_config['y'][y]['label'])
    else:
        x = plot_config['x']
        for y in plot_config['y']:
            plt.plot(data[x], data[y], label=plot_config['y'][y]['label'])

    # Add labels and title
    plt.xlabel(plot_config['x_label'])
    plt.ylabel(plot_config['y_label'])
    plt.title(plot_config['title'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjusts the spacing to prevent legend cutoff

    plt.savefig(os.path.join(config['paths']['plotting_path'], plot_config['file_name']), bbox_inches='tight')

    plt.plot(data)

    if plot_config.get('show', None):
        plt.show()

def violin_plot(data, plot_config, config):
    """
    summary:
    plot a violin plot

    plot_config will control which columns in the data will be plotted

    params:
    data: pd dataframe
    plot_config: dict, plot configuration
    """
    for y in plot_config['y']:
        plt.violinplot(data[y], showmeans=True, showmedians=True)

    # Add labels and title
    plt.xlabel(plot_config['x_label'])
    plt.ylabel(plot_config['y_label'])
    plt.title(plot_config['title'])

    plt.savefig(os.path.join(config['paths']['plotting_path'], plot_config['file_name']), bbox_inches='tight')

    if plot_config.get('show'):
        plt.show()
