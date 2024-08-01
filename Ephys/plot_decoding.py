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
decode_df = pd.read_csv(join(path_dict['save_path'], 'decode_goal_distractor.csv'))

# Plot
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 1.75), dpi=dpi)

sns.lineplot(data=decode_df[decode_df['object'] == 1], x='time', y='accuracy', hue='region', ax=ax1)

sns.lineplot(data=decode_df[decode_df['object'] == 2], x='time', y='accuracy', hue='region', ax=ax2)

sns.lineplot(data=decode_df[decode_df['object'] == 3], x='time', y='accuracy', hue='region', ax=ax3)
