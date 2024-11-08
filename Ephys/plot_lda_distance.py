# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:56:35 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style
colors, dpi = figure_style()

# Load in data
path_dict = paths()
context_env_df = pd.read_csv(join(path_dict['save_path'], 'lda_distance_context.csv')) 


# %%

f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
sns.lineplot(context_env_df, x='distance', y='lda_distance', hue='region', ax=ax1, errorbar='se',
             zorder=1, err_kws={'lw': 0}, style='control')
#ax1.plot(ax1.get_xlim(), [0.5, 0.5], ls='--', color='grey', zorder=0, lw=0.75)
ax1.set(ylabel='Context LDA distance', xticks=[0, 50, 100, 150],
        xlabel='Distance in environment (cm)')
#ax1.set(yscale='log')
ax1.legend().set_title('')
ax1.plot([45, 45], ax1.get_ylim(), ls='--', color='grey')
ax1.plot([90, 90], ax1.get_ylim(), ls='--', color='grey')
ax1.plot([135, 135], ax1.get_ylim(), ls='--', color='grey')

sns.despine(trim=True)
plt.tight_layout()


