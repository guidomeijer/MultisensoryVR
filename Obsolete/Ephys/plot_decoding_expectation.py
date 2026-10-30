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
decode_df = pd.read_csv(join(path_dict['save_path'], 'decode_expectation.csv'))



# %%

f, axs = plt.subplots(1, 8, figsize=(8, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'LEC', 'dCA1', 'iCA1']):
    axs[i].plot([-4, 0], [0.5, 0.5], ls='--', color='grey')
    axs[i].plot([0,0], [0.4, 0.8], ls='--', color='grey')
    sns.lineplot(decode_df[decode_df['region'] == region],
                 x='time', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None)
    axs[i].set_title(region)
    axs[i].axis('off')

axs[0].plot([-2, -1], [0.4, 0.4], color='k', lw=1)
axs[0].text(-1.5, 0.375, '1s', ha='center', va='center')
axs[0].plot([-2, -2], [0.4, 0.5], color='k', lw=1)
axs[0].text(-2.4, 0.45, '10%', ha='center', va='center', rotation=90)
axs[0].text(-3.2, 0.6, 'Context decoding accuracy', ha='center', va='center', rotation=90)
plt.subplots_adjust(left=0.05, bottom=None, right=0.99, top=0.85, wspace=0, hspace=None)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_expectation.pdf'))
