import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np
import matplotlib.patches as mpatches

def line_plot(data, plot_config, config, x_lab=None):
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
    x = plot_config.get('x', None)
    if x_lab:
        x = x_lab
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
        plt.fill_between(x_values, y_lower_percentiles, y_upper_percentiles, alpha=0.3)
        style_num += 1


    # Add labels and title
    plt.xlabel(plot_config.get('x_label', ''))
    plt.title(plot_config.get('title', ''))
    plt.title(plot_config.get('title', ''))
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
            y_series = tech_data.loc[tech_data[ind_var] == x_val, 'empirical_coverage']
            y_means.append(np.mean(y_series))
        plt.plot(x_values, y_means, label=tech['label'], alpha=0.7, linestyle=style_ordering[style_num % 4])
        style_num += 1
    plt.plot(x_values, [config['experiment']['parameters']['confidence_level']] * len(x_values), color='black')

    # Add labels and title
    plt.xlabel(plot_config.get('x_label', ''))
    plt.ylabel('Empirical Coverage')
    plt.title(plot_config.get('title', ''))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjusts the spacing to prevent legend cutoff

    # create a plot file and save it a the plotting path

    plt.savefig(os.path.join(config['paths']['plotting_path'], plot_config['file_name']), bbox_inches='tight')

    if plot_config.get('show', None):
        plt.show()
    
    plt.close()

def sample_plot(data, plot_config, config):
    """
    Take 5 samples from each method, graph their confidence intervals
    """
    # ignorematplotlib warnings
    plt.rcParams.update({'figure.max_open_warning': 0})


    colour_ordering = ['b', 'g', 'r', 'c', 'm', 'k']
    colour_num = 0
    method_count = 0

    num_x_vals = len(config['experiment']['ind_var']['vals'])
    # create a figure with num_x_vals subplots
    fig, axs = plt.subplots(num_x_vals, 1, figsize=(10, 10))

    patches = []

    for tech in plot_config['y_techniques']:
        # create a df with only the data for the technique
        tech_data = data[data['technique'] == tech['technique']]
        x_values = config['experiment']['ind_var']['vals']
        ind_var = config['experiment']['ind_var']['name']
        color = colour_ordering[colour_num % len(colour_ordering)]
        for id, x_val in enumerate(x_values):
            y_series = tech_data.loc[tech_data[ind_var] == x_val, ['ci_low', 'ci_high']]
            y_series = y_series.sample(n=5)
            ci_list = [(y_series.iloc[i][0], y_series.iloc[i][1]) for i in range(5)]
            
            for i in range(5):
                axs[id].plot(ci_list[i], ((method_count + i)/5, (method_count + i)/5),
                             color=color)
                axs[id].axvline(x=config['experiment']['parameters']['true_value'], color='y', linestyle='--')
                # remove the y axis 
                axs[id].get_yaxis().set_visible(False)
        patch = mpatches.Patch(color=color, label=tech['label'])
        patches.append(patch)
        method_count += 5
        colour_num += 1

    # create a legend that captures the different methods with the respective colours

    fig.legend(handles=patches, loc='upper right')
    
    fig.suptitle(plot_config.get('title', ''))

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
    figs, axs = plt.subplots(nrows = num_techs, ncols=1, figsize=(15, 15))
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
        axs[id].violinplot(list_of_y_series, positions = x_values, showmeans=True, showmedians=True, widths=0.15)
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
        elif plot_config['type'] == 'sample':
            sample_plot(data, plot_config, config)
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