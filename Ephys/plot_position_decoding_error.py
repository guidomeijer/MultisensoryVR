# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 09:46:12 2025 by Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style
colors, dpi = figure_style()

path_dict = paths()

error_df = pd.read_csv(path_dict['save_path'] / 'position_decoding_error.csv')
error_df['error_cm'] = error_df['error_mm'] / 10
error_df['error_cm_shuffled'] = error_df['error_mm_shuffled'] / 10
order_regions = error_df[['region', 'error_cm']].groupby('region').mean().sort_values('error_cm').index

f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
sns.boxplot(data=error_df, y='region', x='error_cm', order=order_regions, linewidth=0.75,
            fliersize=0, ax=ax1)
ax1.fill_between([np.quantile(error_df['error_cm_shuffled'], 0.025),
                  np.quantile(error_df['error_cm_shuffled'], 0.975)],
                 [-0.5, -0.5], [7.5, 7.5], color='lightgrey')
#ax1.invert_yaxis()
ax1.text(38.9, -0.8, 'Chance', ha='center', va='center', color='grey')
ax1.set(ylabel='', xlabel='Position decoding error (cm)')
#ax1.tick_params(axis='x', labelrotation=90)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['fig_path'] / 'position_decoding_error.jpg', dpi=600)


