import matplotlib.pyplot as plt
import ppi
import ppi_py
import scipy.stats as stats
import pandas as pd
import ml_models as ml
import copy
import yaml
import os
import sys
import argparse

def isolate_data(df, col, vals):
    """
    Isolate columns of interest and process the data

    Parameters
    df: pandas.DataFrame
        Dataframe containing the data
    col: column name
    vals: list of values to isolate
    """
    df_new = df[df[col].isin(vals)]
    return df_new

