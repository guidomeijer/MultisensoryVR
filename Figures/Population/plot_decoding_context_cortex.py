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
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style, add_significance
colors, dpi = figure_style()

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

def run_stats(df):
    """
    Runs a paired t-test between decoding accuracy and shuffled accuracy.
    """
    cortex_acc = df.loc[df['region'] == 'Cortex', 'accuracy']
    cortex_shuf = df.loc[df['region'] == 'Cortex', 'accuracy_shuf']
    ca1_acc = df.loc[df['region'] == 'CA1', 'accuracy']
    ca1_shuf = df.loc[df['region'] == 'CA1', 'accuracy_shuf']

    # Paired t-test: actual vs shuffled
    p_cortex = stats.ttest_ind(cortex_acc, cortex_shuf)[1] if len(cortex_acc) > 1 else np.nan
    p_ca1 = stats.ttest_ind(ca1_acc, ca1_shuf)[1] if len(ca1_acc) > 1 else np.nan

    return pd.Series([p_cortex, p_ca1], index=['p_cortex', 'p_ca1'])


# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.75 * 2, 1.75), dpi=dpi, sharey=True)

# Do statistics
this_df = context_df[np.isin(context_df['subject'].values,
                     subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int))]
this_shuffle_df = shuffle_df[np.isin(shuffle_df['subject'].values,
                        subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int))]
merged_df = pd.merge(this_df, this_shuffle_df, on=['subject', 'region', 'position'], suffixes=('', '_shuf'))
results_df = merged_df.groupby('position').apply(run_stats).reset_index()

# Plot
sns.lineplot(this_shuffle_df, x='position', y='accuracy', errorbar=('ci', 95), linewidth=0,
             ax=ax1, err_kws={'lw': 0, 'alpha': 1}, legend=None, zorder=0, color='lightgrey')
sns.lineplot(this_df, x='position', y='accuracy', hue='region', errorbar='se',
             ax=ax1, err_kws={'lw': 0}, legend=None, zorder=2,
             hue_order=['Cortex', 'CA1'], palette=[colors['PERI'], colors['CA1']])
ax1.plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
ax1.plot([900, 900], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
_, p_cortex, _,  _ = multipletests(results_df['p_cortex'].values, alpha=0.05, method='bonferroni')
_, p_ca1, _, _ = multipletests(results_df['p_ca1'].values, alpha=0.05, method='bonferroni')
add_significance(results_df['position'].values, p_cortex, ax1, y_pos=0.88, alpha=0.05, color=colors['PERI'])
add_significance(results_df['position'].values, p_ca1, ax1, y_pos=0.86, alpha=0.05, color=colors['CA1'])
ax1.set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
       yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90],
       ylim=[0.3, 0.9], xlabel='', ylabel='', title='Near')

# Do statistics
this_df = context_df[np.isin(context_df['subject'].values,
                     subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
this_shuffle_df = shuffle_df[np.isin(shuffle_df['subject'].values,
                        subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
merged_df = pd.merge(this_df, this_shuffle_df, on=['subject', 'region', 'position'], suffixes=('', '_shuf'))
results_df = merged_df.groupby('position').apply(run_stats).reset_index()

# Add rectangles
ax1.axvspan(WIN_NEAR[0], WIN_NEAR[1], color='grey', alpha=0.2, lw=0)
ax2.axvspan(WIN_FAR[0], WIN_FAR[1], color='grey', alpha=0.2, lw=0)

sns.lineplot(this_shuffle_df, x='position', y='accuracy', errorbar=('ci', 95), linewidth=0,
             ax=ax2, err_kws={'lw': 0, 'alpha': 1}, legend=None, zorder=0,
             hue_order=['Cortex', 'CA1'], color='lightgrey')
sns.lineplot(this_df, x='position', y='accuracy', hue='region', errorbar='se',
             ax=ax2, err_kws={'lw': 0}, legend='brief', zorder=2,
             hue_order=['Cortex', 'CA1'], palette=[colors['PERI'], colors['CA1']])
ax2.plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
ax2.plot([1350, 1350], [0.3, 0.9], ls='--', color='grey', zorder=1, lw=0.5)
_, p_cortex, _,  _ = multipletests(results_df['p_cortex'].values, alpha=0.05, method='bonferroni')
_, p_ca1, _, _ = multipletests(results_df['p_ca1'].values, alpha=0.05, method='bonferroni')
add_significance(results_df['position'].values, p_cortex, ax2, y_pos=0.88, alpha=0.05, color=colors['PERI'])
add_significance(results_df['position'].values, p_ca1, ax2, y_pos=0.86, alpha=0.05, color=colors['CA1'])
ax2.set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
       yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90],
       ylim=[0.3, 0.9], xlabel='', ylabel='', title='Far')
ax2.legend(title='', bbox_to_anchor=(0.38, 0.3))

f.text(0.5, 0.04, 'Position (cm)', ha='center')
f.text(0.06, 0.5, 'Context decoding accuracy (%)', ha='center', va='center', rotation='vertical')
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.98, top=0.9, wspace=0, hspace=0.2)
sns.despine(trim=True)

ax2.spines['left'].set_visible(False)
ax2.yaxis.set_ticks_position('none')
ax2.tick_params(labelleft=False)

plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / f'decode_context_cortex_{CLASSIFIER}.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' /f'decode_context_cortex_{CLASSIFIER}.jpg', dpi=600)
plt.show()