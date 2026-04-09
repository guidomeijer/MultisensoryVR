# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 09/04/2026
"""
# %%

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style
colors, dpi = figure_style()

# Load in data
path_dict = paths()
neuron_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
neuron_df['subject'] = neuron_df['subject'].astype(str)
neuron_df['date'] = neuron_df['date'].astype(str)

waveform_df = pd.read_csv(path_dict['save_path'] / 'waveform_metrics.csv')
waveform_df['subject'] = waveform_df['subject'].astype(str)
waveform_df['date'] = waveform_df['date'].astype(str)
waveform_df = waveform_df.rename(columns={'unit_id': 'neuron_id'})