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
assembly_df = assembly_df[~np.isnan(assembly_df['p_ripples'])]

# Calculate overlap between ripple and obj2 assemblies
assembly_df['ripple_sig'] = assembly_df['p_ripples'] < 0.05
assembly_df['obj1_sig'] = assembly_df['p_obj1'] < 0.05
assembly_df['obj2_sig'] = assembly_df['p_obj2'] < 0.05
assembly_df['ripple_strong'] = ((assembly_df['p_ripples'] < 0.05)
                                & (assembly_df['amp_ripples'] > 0))
assembly_df['diff'] = 'no_diff'
assembly_df.loc[assembly_df['dprime_obj2'] > 0, 'diff'] = 'Pos.'
assembly_df.loc[assembly_df['dprime_obj2'] < 0, 'diff'] = 'Neg.'

p_val_df = pd.DataFrame(index=regions, columns=['obj1', 'obj2'])

for i, region in enumerate(regions):
    region_df = assembly_df[assembly_df['region'] == region]

    # Get the observed percentage of ripple assemblies that are also obj2 modulated
    n_ripple = np.sum(region_df['ripple_strong'])
    if n_ripple == 0:
        continue

    for j, obj in enumerate(['obj1', 'obj2']):
        # Chi-squared test
        contingency_table = pd.crosstab(region_df['ripple_strong'], region_df[f'{obj}_sig'])
        if contingency_table.shape == (2, 2):
            _, p_value, _, _ = stats.chi2_contingency(contingency_table)
        else:
            # Cannot perform chi-squared if a category is entirely missing
            p_value = 1.0
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

# %%
f, axs = plt.subplots(1, len(regions), figsize=(8, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(regions):
    region_df = assembly_df[assembly_df['region'] == region]
    r, p = stats.pearsonr(region_df['amp_ripples'], -np.log10(region_df['p_obj2']))
    print(f'{region}: p={p:.3f}, r={r:.3f}')
    sns.regplot(x='amp_ripples', y=-np.log10(region_df['p_obj2']), data=region_df,
                ax=axs[i], scatter=False, color='black', ci=None)
    axs[i].scatter(region_df['amp_ripples'], -np.log10(region_df['p_obj2']))
    axs[i].set(xlim=[-1, 1], xticks=[-1, 0, 1], ylim=[-0.1, 4], ylabel='', title=region)
axs[0].set(ylabel='Object 2 significance (-log10[p])')

sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %%
f, axs = plt.subplots(1, len(regions), figsize=(8, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(regions):
    region_df = assembly_df[(assembly_df['region'] == region) & (assembly_df['diff'] != 'no_diff')]
    _, p = stats.ttest_ind(region_df.loc[region_df['diff'] == 'Pos.', 'amp_ripples'],
                           region_df.loc[region_df['diff'] == 'Neg.', 'amp_ripples'])
    sns.barplot(data=region_df, x='region', y='amp_ripples', hue='diff', ax=axs[i], errorbar='se',
                hue_order=['Pos.', 'Neg.'], palette=[colors['goal'], colors['no-goal']])
    axs[i].set(title=f'{region}, p={p:.3f}')
plt.show()

# %%

plot_df = assembly_df[assembly_df['ripple_sig'] & (assembly_df['region'] == 'CA1')]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.75 * 2, 1.75), dpi=dpi)

ax1.scatter(plot_df['dprime_obj1'], plot_df['amp_ripples'])

ax2.scatter(plot_df['dprime_obj2'], plot_df['amp_ripples'])

plt.show()
