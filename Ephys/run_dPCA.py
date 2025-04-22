# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:28:35 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from dPCA import dPCA
from msvr_functions import paths, load_subjects, figure_style, combine_regions
colors, dpi = figure_style()

# Load in data
path_dict = paths()
with open(join(path_dict['save_path'], 'env_act_dict.pkl'), 'rb') as f:
    env_act_dict = pickle.load(f)
    
# Loop over regions
dpca_region = dict()
for i, region in enumerate(env_act_dict.keys()):
    
    # Center the data
    this_act = env_act_dict[region]  # neurons x distance bins x context
    this_act -= np.mean(this_act.reshape((this_act.shape[0], -1)), 1)[:, None, None]
    
    # Fit dPCA
    dpca = dPCA.dPCA(labels='ds', n_components=1)
    Z = dpca.fit_transform(this_act)
    dpca_region[region] = np.squeeze(Z['s'])[:, 0]
    
# %% Plot

f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
for i, region in enumerate(dpca_region.keys()):
    ax.plot(dpca_region[region], label=region)