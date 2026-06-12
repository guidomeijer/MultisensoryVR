# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 12/06/2026
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from msvr_functions import paths, figure_style
colors, dpi = figure_style()

# Load in data
path_dict = paths()
decode_df = pd.read_csv(path_dict['google_drive_data_path'] / 'cortex_ca1_pla_decoding.csv')

# Only look at CA1
#decode_df = decode_df[decode_df['region'] == 'CA1']

# %% Plot object decoding
plot_df = decode_df[decode_df['decoder'] == 'object']
f, axs = plt.subplots(1, 5, figsize=(7, 1.75), dpi=dpi)
for i, region in enumerate(np.unique(decode_df['cortical_region'])):
    sns.lineplot(data=plot_df[decode_df['cortical_region'] == region], x='time', y='accuracy', errorbar='se', ax=axs[i],
                 hue='region')
    axs[i].set(title=region)

sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %% Plot object decoding
plot_df = decode_df[decode_df['decoder'] == 'context']
f, axs = plt.subplots(1, 5, figsize=(7, 1.75), dpi=dpi)
for i, region in enumerate(np.unique(decode_df['cortical_region'])):
    sns.lineplot(data=plot_df[decode_df['cortical_region'] == region], x='time', y='accuracy', errorbar='se', ax=axs[i],
                 hue='region')
    axs[i].set(title=region)

sns.despine(trim=True)
plt.tight_layout()
plt.show()
