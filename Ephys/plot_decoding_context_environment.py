# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:27:02 2025

By Guido Meijer
"""


import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style, combine_regions
colors, dpi = figure_style()

# Load in data
path_dict = paths()
subjects = load_subjects()
context_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_GLM_position.csv'))
context_df['region'] = context_df['region'].astype(str)
context_df = context_df[context_df['region'] != 'iCA1']
context_df.loc[context_df['region'] == 'dCA1', 'region'] = 'CA1'


# %%

f, axs = plt.subplots(1, 6, figsize=(7, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'CA1']):
    sns.lineplot(context_df[(context_df['region'] == region)
                            & np.isin(context_df['subject'].values, subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int))],
                 x='position', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None, zorder=1)
    axs[i].plot([0, 1500], [0.5, 0.5], ls='--', color='grey', zorder=0)
    axs[i].plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=0, lw=0.5)
    axs[i].plot([900, 900], [0.3, 0.9], ls='--', color='grey', zorder=0, lw=0.5)
    axs[i].set(title=region, xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
               yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90], xlabel='',
               ylim=[0.3, 0.9])
axs[0].set(ylabel='Decoding accuracy (%)')
f.text(0.5, 0.04, 'Position (cm)', ha='center')
plt.subplots_adjust(left=0.075, bottom=0.21, right=0.98, top=0.85, wspace=0.1)
sns.despine(trim=True)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decode_context_GLM_position_near.pdf'))


# %%

f, axs = plt.subplots(1, 6, figsize=(7, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'CA1']):
    sns.lineplot(context_df[(context_df['region'] == region)
                            & np.isin(context_df['subject'].values, subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))],
                 x='position', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None, zorder=1)
    axs[i].plot([0, 1500], [0.5, 0.5], ls='--', color='grey', zorder=0)
    axs[i].plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=0, lw=0.5)
    axs[i].plot([1350, 1350], [0.3, 0.9], ls='--', color='grey', zorder=0, lw=0.5)
    axs[i].set(title=region, xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
               yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90], xlabel='',
               ylim=[0.3, 0.9])
axs[0].set(ylabel='Decoding accuracy (%)')
f.text(0.5, 0.04, 'Position (cm)', ha='center')
plt.subplots_adjust(left=0.075, bottom=0.21, right=0.98, top=0.85, wspace=0.1)
sns.despine(trim=True)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decode_context_GLM_position_far.pdf'))
