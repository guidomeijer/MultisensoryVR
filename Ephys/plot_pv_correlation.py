# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:17:20 2024
@author: Guido
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from msvr_functions import paths, load_subjects

# Load data
path_dict = paths()
subjects = load_subjects()
pv_corr_df = pd.read_pickle(path_dict['google_drive_data_path'] / 'pv_correlation.pickle')

# Plot
regions = pv_corr_df['region'].unique()
f, ax = plt.subplots(1, len(regions), figsize=(5 * len(regions), 4), dpi=300, squeeze=False)
ax = ax.flatten()

for i, region in enumerate(regions):
    region_df = pv_corr_df[(pv_corr_df['region'] == region)
                            & np.isin(pv_corr_df['subject'].values,
                                      subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
    pv_corr = np.mean(np.stack(region_df['pv_corr'].values), axis=0)
    sns.heatmap(pv_corr, cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax[i])
    ax[i].set(title=f'{region} (n={region_df.shape[0]})',
              xticks=[0, pv_corr.shape[0]], yticks=[0, pv_corr.shape[0]],
              xticklabels=[1, pv_corr.shape[0]], yticklabels=[1, pv_corr.shape[0]],
              xlabel='Spatial bin', ylabel='Spatial bin')

plt.tight_layout()
plt.savefig(path_dict['fig_path'] / 'pv_correlation.pdf')
plt.show()
