import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np

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
    style_ordering = ['-', '--', '-.', ':']
    style_num = 0
    if x:
        for y in plot_config['y']:
            plt.plot(x, data[y], label=plot_config['y'][y]['label'])
    else:
        x = plot_config['x']
        for tech in plot_config['y_techniques']:
            # create a df with only the data for the technique
            tech_data = data[data['technique'] == tech['technique']]
            x_values = config['experiment']['ind_var']['vals']
            ind_var = config['experiment']['ind_var']['name']
            y_means = []
            y_lower_percentiles = []
            y_upper_percentiles = []
            for x_val in x_values:
                y_series = tech_data.loc[tech_data[ind_var] == x_val, plot_config['y_metric']]
                y_means.append(np.mean(y_series))
                y_lower_percentiles.append(np.percentile(y_series, 10))
                y_upper_percentiles.append(np.percentile(y_series, 90))
            plt.plot(x_values, y_means, label=tech['label'], alpha=0.7, lw=0.7, linestyle=style_ordering[style_num % 4])
            plt.fill_between(x_values, y_lower_percentiles, y_upper_percentiles, alpha=0.5)
            style_num += 1


    # Add labels and title
    plt.xlabel(plot_config['x_label'])
    plt.ylabel(plot_config['y_label'])
    plt.title(plot_config['title'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjusts the spacing to prevent legend cutoff

    # create a plot file and save it a the plotting path

    plt.savefig(os.path.join(config['paths']['plotting_path'], plot_config['file_name']), bbox_inches='tight')

    if plot_config.get('show', None):
        plt.show()
    
    plt.close()

def coverage_plot(data, plot_config, config):
    """
    Plot the coverage of the confidence intervals
    """
    style_ordering = ['-', '--', '-.', ':']
    style_num = 0
    x = plot_config['x']
    for tech in plot_config['y_techniques']:
        # create a df with only the data for the technique
        tech_data = data[data['technique'] == tech['technique']]
        x_values = config['experiment']['ind_var']['vals']
        ind_var = config['experiment']['ind_var']['name']
        y_means = []
        for x_val in x_values:
            y_series = tech_data.loc[tech_data[ind_var] == x_val, plot_config['empirical_coverage']]
            y_means.append(np.mean(y_series))
        plt.plot(x_values, y_means, label=tech['label'], alpha=0.7, linestyle=style_ordering[style_num % 4])
        style_num += 1
    plt.plot(x_values, [config['experiment']['parameters']['confidence_level']] * len(x_values), color='black')

    # Add labels and title
    plt.xlabel(plot_config['x_label'])
    plt.ylabel(plot_config['y_label'])
    plt.title(plot_config['title'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjusts the spacing to prevent legend cutoff

    # create a plot file and save it a the plotting path

    plt.savefig(os.path.join(config['paths']['plotting_path'], plot_config['file_name']), bbox_inches='tight')

    if plot_config.get('show', None):
        plt.show()
    
    plt.close()


def line_plot_generic(data, x_col, y_cols):
    """
    Line plotting function that does not require a configuration file
    """
    for y in y_cols:
        plt.plot(data[x_col], np.mean(data[y]), label=y)
        plt.fill_between(data[x_col], np.percentile(data[y], 10), np.percentile(data[y], 90), alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel('Value')
    plt.title('Line Plot')
    plt.legend()
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
    # Create a figure 
    num_techs = len(plot_config['y_techniques'])
    figs, axs = plt.subplots(nrows = num_techs, ncols=1, figsize=(10, 10))
    for id, tech in enumerate(plot_config['y_techniques']):
        tech_data = data[data['technique'] == tech['technique']] # isolate the data for the technique
        x_values = config['experiment']['ind_var']['vals']
        ind_var = config['experiment']['ind_var']['name']
        list_of_y_series = []
        for x_val in x_values:
            y_series = tech_data.loc[tech_data[ind_var] == x_val, plot_config['y_metric']] # isolate the x values
            # turn y_series into a list
            y_series = y_series.tolist()
            list_of_y_series.append(y_series)
        axs[id].violinplot(list_of_y_series, x_values, showmeans=True, showmedians=True, widths=0.15)
        axs[id].set_title(tech['label'])

    # Save figure
    plt.savefig(os.path.join(config['paths']['plotting_path'], plot_config['file_name']), bbox_inches='tight')

    if plot_config.get('show', None):
        plt.show()

    # close the figure
    plt.close()

def plot_results(data, config):
    """
    summary:
    plot the results
    """
    for plot_config in config['plotting']['plots']:
        if plot_config['type'] == 'line':
            line_plot(data, plot_config, config)
        elif plot_config['type'] == 'violin':
            violin_plot(data, plot_config, config)
        elif plot_config['type'] == 'coverage':
            coverage_plot(data, plot_config, config)
        else:
            raise ValueError(f"Plot type {plot_config['type']} not supported")
    return

# Testing

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('./experiments/Keep_5/results/results.csv')
    config = yaml.load(open('./configs/basic_experiment.yaml'), Loader=yaml.FullLoader)

    # Plot the data
    plot_results(data, config)