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
angles_df = pd.read_csv(path_dict['google_drive_data_path'] / 'cortex_ca1_pla_subspace_angles.csv')

# Only look at CA1
#decode_df = decode_df[decode_df['region'] == 'CA1']

# %% Plot object decoding
plot_df = decode_df[decode_df['decoder'] == 'object']
f, axs = plt.subplots(1, 5, figsize=(6, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(np.unique(decode_df['cortical_region'])):
    axs[i].plot([-2, 2], [0.33, 0.33], ls='--', color='grey')
    sns.lineplot(data=plot_df[plot_df['cortical_region'] == region], x='time', y='accuracy', errorbar='se', ax=axs[i],
                 hue='region', palette=[colors['Cortex'], colors['CA1']], err_kws={'lw': 0}, legend=None)
    axs[i].set(title=region, ylabel='', xlabel='')
axs[0].set(ylabel='Object decoding accuracy (%)')

sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %% Plot context decoding
plot_df = decode_df[decode_df['decoder'].str.startswith('context_obj1')]
f, axs = plt.subplots(1, 5, figsize=(6, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(np.unique(decode_df['cortical_region'])):
    axs[i].plot([-2, 2], [0.5, 0.5], ls='--', color='grey')
    sns.lineplot(data=plot_df[plot_df['cortical_region'] == region], x='time', y='accuracy', errorbar='se', ax=axs[i],
                 hue='region', palette=[colors['Cortex'], colors['CA1']], err_kws={'lw': 0}, legend=None)
    axs[i].set(title=region, ylabel='', xlabel='')
axs[0].set(ylabel='Context decoding accuracy (%)')
f.suptitle('Object 1')

sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %% Plot context decoding
plot_df = decode_df[decode_df['decoder'].str.startswith('context_obj2')]
f, axs = plt.subplots(1, 5, figsize=(6, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(np.unique(decode_df['cortical_region'])):
    axs[i].plot([-2, 2], [0.5, 0.5], ls='--', color='grey')
    sns.lineplot(data=plot_df[plot_df['cortical_region'] == region], x='time', y='accuracy', errorbar='se', ax=axs[i],
                 hue='region', palette=[colors['Cortex'], colors['CA1']], err_kws={'lw': 0}, legend=None)
    axs[i].set(title=region, ylabel='', xlabel='')
axs[0].set(ylabel='Context decoding accuracy (%)')
f.suptitle('Object 2')

sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %% Plot context decoding
plot_df = decode_df[decode_df['decoder'].str.startswith('context_obj3')]
f, axs = plt.subplots(1, 5, figsize=(6, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(np.unique(decode_df['cortical_region'])):
    axs[i].plot([-2, 2], [0.5, 0.5], ls='--', color='grey')
    sns.lineplot(data=plot_df[plot_df['cortical_region'] == region], x='time', y='accuracy', errorbar='se', ax=axs[i],
                 hue='region', palette=[colors['Cortex'], colors['CA1']], err_kws={'lw': 0}, legend=None)
    axs[i].set(title=region, ylabel='', xlabel='')
axs[0].set(ylabel='Context decoding accuracy (%)')
f.suptitle('Object 3')

sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %% Plot angles
f, ax = plt.subplots(figsize=(2.5, 1.75), dpi=dpi)
sns.barplot(data=angles_df, x='region_pair', y='angle_1', errorbar='se', ax=ax)
ax.tick_params(axis='x', labelrotation=90)
ax.set(ylabel='Angle 1 (%)', xlabel='')

sns.despine(trim=True)
plt.tight_layout()
plt.show()

