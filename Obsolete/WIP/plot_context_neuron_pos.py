# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 12:42:18 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style

BIN_SIZE = 50
Y_BINS = np.arange(-2590, -3620, step=-BIN_SIZE)

# Load in data
path_dict = paths()
neuron_pos_df = pd.read_csv(path_dict['save_path'] / 'neuron_position.csv')
peri_df = neuron_pos_df[(neuron_pos_df['region'] == 'PERI 36') | (neuron_pos_df['region'] == 'PERI 35')]

# Drop nans
peri_df = peri_df[~peri_df['p_context'].isnull()]

perc_sig = np.empty(Y_BINS.shape[0] - 1)
for i in range(Y_BINS.shape[0] - 1):
    bin_df = peri_df[(peri_df['y'] > Y_BINS[i+1]) & (peri_df['y'] < Y_BINS[i])]
    perc_sig[i] = (np.sum(bin_df['p_context'] < 0.05) / bin_df.shape[0]) * 100
    
# %%
colors, dpi = figure_style()
f, axs = plt.subplots(1, 2, figsize=(1.75 * 2, 1.75), dpi=dpi)

axs[0].plot(Y_BINS[:-1] - (BIN_SIZE/2), perc_sig)
axs[0].set(xticks=[-3000, -3600], xticklabels=[-3000, 3600], xlabel='AP (um)')
axs[0].invert_xaxis()

sc = axs[1].scatter(peri_df['y'], peri_df['z'], c=peri_df['p_context'], s=1, cmap='turbo_r')
f.colorbar(sc)
axs[1].invert_xaxis()

sns.despine(trim=True)
plt.tight_layout()
