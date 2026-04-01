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

p_val_df = pd.DataFrame(index=regions, columns=['obj1', 'obj2'])

for i, region in enumerate(regions):
    region_df = assembly_df[assembly_df['region'] == region]

    # Get the observed percentage of ripple assemblies that are also obj2 modulated
    n_ripple = np.sum(region_df['ripple_sig'])
    if n_ripple == 0:
        continue

    for j, obj in enumerate(['obj1', 'obj2']):
        n_overlap = np.sum(region_df['ripple_sig'] & region_df[f'{obj}_sig'])

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
        p_val_df.loc[region, obj] = p_value

# %% Plot
f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.heatmap(p_val_df.astype(float), annot=True, fmt='.3f', cmap='magma_r', vmin=0, vmax=1, ax=ax,
            cbar_kws={'label': 'p-value'})
ax.set(ylabel='', xlabel='', xticks=[0.5, 1.5], xticklabels=['Object 1', 'Object 2'])
ax.tick_params(axis='y', labelrotation=0)

plt.tight_layout()
plt.show()
#plt.savefig(join(path_dict['google_drive_fig_path'], 'ripple_obj_overlap.pdf'))