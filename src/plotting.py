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
        for tech in plot_config['y_techniques']:
            # create a df with only the data for the technique
            tech_data = data[data['technique'] == tech['technique']]
            plt.plot(tech_data[x], tech_data[plot_config.get('y_metric', None)], label=tech['label'])
            

    # Add labels and title
    plt.xlabel(plot_config['x_label'])
    plt.ylabel(plot_config['y_label'])
    plt.title(plot_config['title'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjusts the spacing to prevent legend cutoff

    # create a plot file and save it a the plotting path

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

def plot_results(data, config):
    """
    summary:
    plot the results
    """
    plot_config = config['plotting']
    for plot in plot_config['plots']:
        if plot['type'] == 'line':
            line_plot(data, plot, config)
        elif plot['type'] == 'violin':
            violin_plot(data, plot, config)
        else:
            raise ValueError(f"Plot type {plot['type']} not supported")
        
    return

# Testing

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('./experiments/Sample_experiment_keep_3/results/results.csv')
    config = yaml.load(open('./configs/basic_experiment.yaml'), Loader=yaml.FullLoader)

    # Plot the data
    plot_results(data, config)