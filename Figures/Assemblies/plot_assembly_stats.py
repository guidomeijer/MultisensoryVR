# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 05/03/2026
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
assembly_df = pd.read_csv(join(path_dict['save_path'], 'assembly_sig.csv'))
regions = np.unique(assembly_df['region'])

# Calculate overlap between ripple and obj2 assemblies
assembly_df['ripple_sig'] = assembly_df['p_ripples'] < 0.05
assembly_df['obj1_sig'] = assembly_df['p_obj1'] < 0.05
assembly_df['obj2_sig'] = assembly_df['p_obj2'] < 0.05

f, axs = plt.subplots(2, len(regions), figsize=(1.75 * len(regions), 3.5), dpi=dpi)

for i, region in enumerate(regions):
    region_df = assembly_df[assembly_df['region'] == region]

    # Get the observed percentage of ripple assemblies that are also obj2 modulated
    n_ripple = np.sum(region_df['ripple_sig'])
    if n_ripple == 0:
        continue

    for j, obj in enumerate(['obj1', 'obj2']):
        n_overlap = np.sum(region_df['ripple_sig'] & region_df[f'{obj}_sig'])
        obs_pct = (n_overlap / n_ripple) * 100

        # Permutation test
        n_shuffles = 10000
        shuffled_overlap = np.zeros(n_shuffles)
        obj_labels = region_df[f'{obj}_sig'].values.copy()
        ripple_sig = region_df['ripple_sig'].values
        for k in range(n_shuffles):
            np.random.shuffle(obj_labels)
            shuffled_overlap[k] = np.sum(ripple_sig & obj_labels)

        # Calculate p-value
        p_value = (np.sum(shuffled_overlap >= n_overlap) + 1) / (n_shuffles + 1)

        # Plot results
        ax = axs[j, i]
        ax.hist(shuffled_overlap / n_ripple * 100, bins=20, color='grey')
        ax.axvline(obs_pct, color='r', linestyle='--')
        if j == 0:
            ax.set(title=f'{region}\n{obj}\np = {p_value:.3f}')
        else:
            ax.set(title=f'{obj}\np = {p_value:.3f}')
        ax.set(xlabel=f'% Ripple assemblies\nmodulated by {obj}', ylabel='Count')

plt.tight_layout()
sns.despine(trim=True)
plt.show()
plt.savefig(join(path_dict['save_path'], 'ripple_obj_overlap.pdf'))