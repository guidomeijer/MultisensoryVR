# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:27:02 2025

By Guido Meijer
"""


import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
from scipy import stats
import mne
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style, add_significance
colors, dpi = figure_style()
mne.set_log_level('WARNING')

# Settings
CLASSIFIER = 'randomforest'
WIN_NEAR = [900-150, 900]
WIN_FAR = [1350-150, 1350]

# Load in data
path_dict = paths()
subjects = load_subjects()
context_df = pd.read_csv(join(path_dict['save_path'], f'decode_context_cortex_25neurons_{CLASSIFIER}.csv'),
                         dtype={'subject': str, 'region': str})
shuffle_df = pd.read_csv(join(path_dict['save_path'], f'decode_context_cortex_25neurons_{CLASSIFIER}_shuffle.csv'),
                         dtype={'subject': str, 'region': str})

def run_stats(df, shuffle_df):
    test_matrix = df.pivot_table(index=['subject', 'date'], columns='position', values='accuracy', aggfunc='mean')
    control_matrix = shuffle_df.pivot_table(index=['subject', 'date'], columns='position', values='accuracy', aggfunc='mean')
    positions = test_matrix.columns.values
    X = test_matrix.values - control_matrix.values
    t_threshold = stats.t.ppf(1 - 0.3 / 2, test_matrix.shape[0]-1)
    t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
       X,
       threshold=t_threshold,
       n_permutations=1000, 
       tail=0,          # Two-tailed test
       out_type='mask'  # Returns boolean masks for positions
       )
    p_cortex = np.ones(len(positions))
    for cluster_mask, p_val in zip(clusters, cluster_p_values):
        if p_val < 0.05:
            # Assign the cluster-level p-value to all positions in this cluster
            p_cortex[cluster_mask] = p_val
    return p_cortex, positions

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi, sharey=True)

# Near object
this_df = context_df[np.isin(context_df['subject'].values,
                     subjects.loc[subjects['Far'] == 0, 'SubjectID'].astype(str).values)]
this_shuffle_df = shuffle_df[np.isin(shuffle_df['subject'].values,
                        subjects.loc[subjects['Far'] == 0, 'SubjectID'].astype(str).values)]
# Do stats
p_cortex, positions = run_stats(this_df[this_df['region'] == 'Cortex'], 
                                this_shuffle_df[this_shuffle_df['region'] == 'Cortex'])
p_ca1, positions = run_stats(this_df[this_df['region'] == 'CA1'], 
                             this_shuffle_df[this_shuffle_df['region'] == 'CA1'])

# Plot
sns.lineplot(this_shuffle_df, x='position', y='accuracy', errorbar=('ci', 95), linewidth=0,
             ax=ax1, err_kws={'lw': 0, 'alpha': 1}, legend=None, zorder=0, color='lightgrey')
sns.lineplot(this_df, x='position', y='accuracy', hue='region', errorbar='se',
             ax=ax1, err_kws={'lw': 0}, legend=None, zorder=2,
             hue_order=['Cortex', 'CA1'], palette=[colors['PERI'], colors['CA1']])
ax1.plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
ax1.plot([900, 900], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
add_significance(positions, p_cortex, ax1, y_pos=0.88, alpha=0.05, color=colors['PERI'])
add_significance(positions, p_ca1, ax1, y_pos=0.86, alpha=0.05, color=colors['CA1'])
ax1.set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
       yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90],
       ylim=[0.3, 0.9], xlabel='', ylabel='', title='Near')

# Far objects
this_df = context_df[np.isin(context_df['subject'].values,
                     subjects.loc[subjects['Far'] == 1, 'SubjectID'].astype(str).values)]
this_shuffle_df = shuffle_df[np.isin(shuffle_df['subject'].values,
                        subjects.loc[subjects['Far'] == 1, 'SubjectID'].astype(str).values)]

# Do stats
p_cortex, positions = run_stats(this_df[this_df['region'] == 'Cortex'], 
                                this_shuffle_df[this_shuffle_df['region'] == 'Cortex'])
p_ca1, positions = run_stats(this_df[this_df['region'] == 'CA1'], 
                             this_shuffle_df[this_shuffle_df['region'] == 'CA1'])

# Add rectangles
ax1.axvspan(WIN_NEAR[0], WIN_NEAR[1], color='grey', alpha=0.2, lw=0)
ax2.axvspan(WIN_FAR[0], WIN_FAR[1], color='grey', alpha=0.2, lw=0)

sns.lineplot(this_shuffle_df, x='position', y='accuracy', errorbar=('ci', 95), linewidth=0,
             ax=ax2, err_kws={'lw': 0, 'alpha': 1}, legend=None, zorder=0, color='lightgrey')
sns.lineplot(this_df, x='position', y='accuracy', hue='region', errorbar='se',
             ax=ax2, err_kws={'lw': 0}, legend='brief', zorder=2,
             hue_order=['Cortex', 'CA1'], palette=[colors['Cortex'], colors['CA1']])
ax2.plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
ax2.plot([1350, 1350], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
add_significance(positions, p_cortex, ax2, y_pos=0.88, alpha=0.05, color=colors['Cortex'])
add_significance(positions, p_ca1, ax2, y_pos=0.86, alpha=0.05, color=colors['CA1'])
ax2.set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
       yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90],
       ylim=[0.3, 0.9], xlabel='', ylabel='', title='Far')
ax2.legend(title='', bbox_to_anchor=(0.38, 0.3))

f.text(0.5, 0.04, 'Position (cm)', ha='center')
f.text(0.06, 0.55, 'Context decoding accuracy (%)', ha='center', va='center', rotation='vertical')
plt.subplots_adjust(left=0.14, bottom=0.2, right=0.98, top=0.9, wspace=0.05, hspace=0.2)
sns.despine(trim=True)

ax2.spines['left'].set_visible(False)
ax2.yaxis.set_ticks_position('none')
ax2.tick_params(labelleft=False)

plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / f'decode_context_cortex_{CLASSIFIER}.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' /f'decode_context_cortex_{CLASSIFIER}.jpg', dpi=600)
plt.show()

# %%

f, ax1 = plt.subplots(1, 1, figsize=(1.9, 1.75), dpi=dpi)

# Near object
this_df = context_df[np.isin(context_df['subject'].values,
                     subjects.loc[subjects['Far'] == 0, 'SubjectID'].astype(str).values)]
this_shuffle_df = shuffle_df[np.isin(shuffle_df['subject'].values,
                        subjects.loc[subjects['Far'] == 0, 'SubjectID'].astype(str).values)]

sns.lineplot(this_shuffle_df, x='position', y='accuracy', errorbar=('ci', 95), linewidth=0,
             ax=ax1, err_kws={'lw': 0, 'alpha': 1}, legend=None, zorder=0, color='w')
sns.lineplot(this_df[this_df['region'] == 'CA1'], x='position', y='accuracy', color='w',
             errorbar='se', ax=ax1, err_kws={'lw': 0}, legend=None, zorder=2)
ax1.plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
ax1.plot([900, 900], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
ax1.set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
       yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90],
       ylim=[0.3, 0.9], xlabel='Position (cm)', ylabel='Context decoding accuracy (%)')

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(path_dict['paper_fig_path'] / 'Presentations' / 'decoding_only_lines.jpg', dpi=600)
plt.show()

f, ax1 = plt.subplots(1, 1, figsize=(1.9, 1.75), dpi=dpi)

# Do stats
p_cortex, positions = run_stats(this_df[this_df['region'] == 'Cortex'], 
                                this_shuffle_df[this_shuffle_df['region'] == 'Cortex'])
p_ca1, positions = run_stats(this_df[this_df['region'] == 'CA1'], 
                             this_shuffle_df[this_shuffle_df['region'] == 'CA1'])

# Plot
sns.lineplot(this_shuffle_df, x='position', y='accuracy', errorbar=('ci', 95), linewidth=0,
             ax=ax1, err_kws={'lw': 0, 'alpha': 1}, legend=None, zorder=0, color='lightgrey')
sns.lineplot(this_df[this_df['region'] == 'CA1'], x='position', y='accuracy', color=colors['CA1'],
             errorbar='se', ax=ax1, err_kws={'lw': 0}, legend=None, zorder=2)
ax1.plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
ax1.plot([900, 900], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
#add_significance(positions, p_cortex, ax1, y_pos=0.88, alpha=0.05, color=colors['PERI'])
add_significance(positions, p_ca1, ax1, y_pos=0.86, alpha=0.05, color=colors['CA1'])
ax1.set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
       yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90],
       ylim=[0.3, 0.9], xlabel='Position (cm)', ylabel='Context decoding accuracy (%)')

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(path_dict['paper_fig_path'] / 'Presentations' / 'decoding_only_CA1.jpg', dpi=600)
plt.show()

f, ax1 = plt.subplots(1, 1, figsize=(1.9, 1.75), dpi=dpi)

# Do stats
p_cortex, positions = run_stats(this_df[this_df['region'] == 'Cortex'], 
                                this_shuffle_df[this_shuffle_df['region'] == 'Cortex'])
p_ca1, positions = run_stats(this_df[this_df['region'] == 'CA1'], 
                             this_shuffle_df[this_shuffle_df['region'] == 'CA1'])

# Plot
sns.lineplot(this_shuffle_df, x='position', y='accuracy', errorbar=('ci', 95), linewidth=0,
             ax=ax1, err_kws={'lw': 0, 'alpha': 1}, legend=None, zorder=0, color='lightgrey')
sns.lineplot(this_df, x='position', y='accuracy', hue='region', hue_order=['Cortex', 'CA1'],
             palette=[colors['Cortex'], colors['CA1']],
             errorbar='se', ax=ax1, err_kws={'lw': 0}, legend=None, zorder=2)
ax1.plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
ax1.plot([900, 900], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
add_significance(positions, p_cortex, ax1, y_pos=0.88, alpha=0.05, color=colors['PERI'])
add_significance(positions, p_ca1, ax1, y_pos=0.86, alpha=0.05, color=colors['CA1'])
ax1.set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
       yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90],
       ylim=[0.3, 0.9], xlabel='Position (cm)', ylabel='Context decoding accuracy (%)')

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(path_dict['paper_fig_path'] / 'Presentations' / 'decoding_CA1_and_cortex.jpg', dpi=600)
plt.show()