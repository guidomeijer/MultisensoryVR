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
decode_df = pd.read_csv(join(path_dict['save_path'], 'decode_object.csv'))


# %%

f, axs = plt.subplots(1, 8, figsize=(8, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'LEC', 'dCA1', 'iCA1']):
    sns.lineplot(decode_df[decode_df['region'] == region],
                 x='time', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None)
    axs[i].set(title=region, xticks=[-3, -2, -1, 0, 1], xlabel='', xlim=[-3, 1], ylim=[0.5, 1])
    if i == 0:
        axs[i].set(ylabel='Object decoding accuracy (%)', yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
    else:        
        axs[i].get_yaxis().set_visible(False)      
        axs[i].spines['left'].set_visible(False)    
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)

f.text(0.5, 0.04, 'Time from object entry (s)', ha='center', va='center')
#sns.despine(trim=True)
plt.subplots_adjust(left=0.05, bottom=0.2, right=0.99, top=0.85)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_object.pdf'))

