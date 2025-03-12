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
per_obj_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_per_object.csv'))
#per_obj_df = per_obj_df[per_obj_df['region'] != 'ENT']
per_obj_df = per_obj_df[(per_obj_df['region'] == 'PERI 35') | (per_obj_df['region'] == 'vCA1')]
onset_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_onset.csv'))
onset_df = onset_df[onset_df['region'] != 'ENT']

# Plot
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75*3, 1.75), dpi=dpi)
sns.lineplot(per_obj_df[per_obj_df['object'] == 1], x='time', y='accuracy', hue='region', errorbar='se',
             ax=ax1, err_kws={'lw': 0}, palette=colors)

sns.lineplot(per_obj_df[per_obj_df['object'] == 2], x='time', y='accuracy', hue='region', errorbar='se',
             ax=ax2, err_kws={'lw': 0}, palette=colors)

sns.lineplot(per_obj_df[per_obj_df['object'] == 3], x='time', y='accuracy', hue='region', errorbar='se',
             ax=ax3, err_kws={'lw': 0}, palette=colors)

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(onset_df, x='time', y='accuracy', hue='region', errorbar='se', ax=ax1,
             err_kws={'lw': 0}, palette=colors)