# ppi-testing
Run the following script after navigating to the src folder.

python main.py --config ../configs/basic_experiment.yaml

Usage:

Create a yaml file to configure the experiment

Use example.yaml as a reference for what is needed.

Keep the path section as is.

### Experiments

experiment is a mandatory key

name - name of the file, the main.py function will *not* copy the name of the yaml file to use as the experiment folder name

description - optional experiment description (legacy feature for summarize.py, which is not working at the moment)

parameters is a mandatory key

training population requires
- training_population
- gold_population
- unlabelled_population

All 3 require keys x_population, y_population

x_population requires:

- distribution - the name of the distribution (see distributions.py for reference of supported strings, distributions)
- size - size of population set
- all required parameters of distribution

y_population requires:
- transformation - the name of the transformation (see distributions.py for reference of supported strings, transformation)
- all required parameters of transformation

true_value - optional, if set to null or excluded, will sample the gold_population 100k times to estimate the true parameter (currently only supports estimate type: mean)

n_its - number of iteration experiment

test_size - optional, test split size, default - 0.2

use_active_inference - I have no idea how that got there or what it does, probably can delete

confidence_level - optional, intended confidnece level (not alpha value!!) default - 0.95

cut_interval - optional, if True, will cut off all negative values of confidence interval, default - False

ind_var - independent variables that will be altered throughout experiment
- requires name, list of string names of independent variables
- requires vals, dictionary of key value pairs of name\[val\]
- requires paths, dictionary with keys of list(name)
    - Contains the path inside of the yaml file of the variable to be altered throughout experiment

Example usage:

```
ind_var:
    name: 
      - mean
      - std
    vals:
      - mean: 0
        std: 4
      - mean: 2
        std: 4
      - mean: -2
        std: 4
      - mean: 4
        std: 4
      - mean: -4
        std: 4
      - mean: 0
        std: 5
      - mean: 0
        std: 6
    paths:
      mean:
        - experiment.parameters.gold_population.x_population.mean
        - experiment.parameters.unlabelled_population.x_population.mean
        - experiment.parameters.gold_population.y_population.mean
        - experiment.parameters.unlabelled_population.y_population.mean
      std:
        - experiment.parameters.gold_population.x_population.std
        - experiment.parameters.unlabelled_population.x_population.std
        - experiment.parameters.gold_population.y_population.std
        - experiment.parameters.unlabelled_population.y_population.std
```

model - dictionary of settings of model to be trained
- name - name of model (see ml_models.py for reference of supported models)
- optuna - optional, if True, will use optuna hyperparameter tuning, default - False
- trials - optional, number of optuna trials, default - 10

model_bias - if True, will calculate model bias

estimate - type of estimate being estimate, currently only supports mean

methods - list of methods of constructing confidence intervals that will be tested

metrics - list of metrics to be computed (keep widths, coverages almost always)

distances - optional, distance between distributions metric that will be computed, only used for covariate shift experiments

plot_distributions - optional, if True, will plot the X distributions to be tested

clipping - optional, if True, will remove all unlabelled points that are outside of the training distribution

remove_gold - optional, will also remove gold values outside of training distribution

varying_true_value - optional, if True, will recompute true_value every new independent variable

train_once - optional, if True, trains a model once per independent variable instead per experiment iteration

### Plotting

Too many plot types to specify, use example.yaml as a reference. The way plotting is run is that first, main.py will run the experiment and create the results.csv, and a pd_dataframe, which will be sent to 
plotting functions, each plot under plotting\[plots\] creates a new plot, these plots have their own config, which is relatively straightforward. The only important thing to note is that x is the key of the pd_dataframe that you will want to use as plotting x variable (you do not have to in general worry about duplicates, plotting.py calls col.uniques())

If you have a bunch of data, and want to just rerun the plotting, run plot_only.py, there are two options
- run python plot_only.py --folder_path "folder path", it will automatically rerun only the plotting section of the experiment_config inside the folder, however, this requires altering the
config already copied from when the experiment was first run. You may use it if you do not mind it being change
- run python plot_only.py --config "config path" --data "data path", this is another way of rerunning the plotting section, however, this can be done with any config, data path, hence you 
do not have to alter the already saved experiment_config_folder.