# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 30/04/2026
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyqtgraph.examples.PColorMeshItem import edgecolors

from msvr_functions import paths, figure_style
path_dict = paths()
colors, dpi = figure_style()

# Load in data
pla_df = pd.read_csv(path_dict['google_drive_data_path'] / 'pla_n_components.csv')

g = sns.FacetGrid(pla_df, col='region_pair', col_wrap=4, height=3)
g.map_dataframe(sns.lineplot, x='n_components', y='r_norm')
g.set_axis_labels('Number of components', 'Normalized R')
g.set(ylim=(0.6, 1))
g.set_titles(col_template='{col_name}')
plt.tight_layout()
plt.show()

# %%
f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=pla_df, x='n_components', y='r', ax=ax, errorbar='se', err_kws={'lw': 0}, color='k', marker='o',
             markeredgecolor='k')
ax.set(ylabel='Correlation coefficient (r)', xlabel='Number of components', xticks=np.arange(1, 11),
       yticks=[0.14, 0.16, 0.18, 0.2, 0.22])

sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'pla_n_components.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'pla_n_components.jpg', dpi=600)