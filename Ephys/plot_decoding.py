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
obj_context_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_per_object.csv'))
context_onset_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_onset.csv'))
obj_id_df = pd.read_csv(join(path_dict['save_path'], 'decode_object_identity.csv'))
context_env_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_environment.csv')) 

# Subtract the decoding accuracy of the control object 
for region in np.unique(obj_context_df['region']):
    mean_control = obj_context_df.loc[(obj_context_df['object'] == 3) & (obj_context_df['region'] == region),
                                      ['time', 'accuracy']].groupby('time').mean()
    for time_point in mean_control.index:
        obj_context_df.loc[obj_context_df['time'] == time_point, 'acc_subtr'] = (
            obj_context_df.loc[obj_context_df['time'] == time_point, 'accuracy']
            - mean_control.loc[time_point, 'accuracy'])

# %% Plot
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4.5, 1.75), dpi=dpi, sharey=True)

sns.lineplot(data=obj_context_df[obj_context_df['object'] == 1], x='time', y='accuracy', hue='region', ax=ax1,
             errorbar='se', zorder=1)
ax1.plot(ax1.get_xlim(), [0.5, 0.5], ls='--', color='grey', zorder=0, lw=0.75)
ax1.set(ylim=[0.4, 0.95], ylabel='Context decoding accuracy (%)', title='Object 1',
        yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], xlabel='')
ax1.legend().set_title('')

sns.lineplot(data=obj_context_df[obj_context_df['object'] == 2], x='time', y='accuracy', hue='region', ax=ax2,
             errorbar='se', legend=None, zorder=1)
ax2.plot(ax1.get_xlim(), [0.5, 0.5], ls='--', color='grey', zorder=0, lw=0.75)
ax2.set(title='Object 2', xlabel='')

sns.lineplot(data=obj_context_df[obj_context_df['object'] == 3], x='time', y='accuracy', hue='region', ax=ax3,
             errorbar='se', legend=None, zorder=1)
ax3.plot(ax1.get_xlim(), [0.5, 0.5], ls='--', color='grey', zorder=0, lw=0.75)
ax3.set(title='Control object', xlabel='')
    

f.text(0.5, 0.06, 'Time from object entry (s)', ha='center')
sns.despine(trim=True)
plt.subplots_adjust(bottom=0.22)

"""
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=obj_context_df[obj_context_df['object'] == 2], x='time', y='rel_acc', hue='region', ax=ax1,
             errorbar='se', zorder=1)
"""

# %%

f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)

sns.lineplot(data=obj_context_df[obj_context_df['object'] == 2], x='time', y='acc_subtr', hue='region', ax=ax1,
             errorbar='se', zorder=1, palette=colors)
ax1.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey', zorder=0, lw=0.75)
ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='grey', zorder=0, lw=0.75)
ax1.set(ylim=[-0.2, 0.4], ylabel='Context decoding accuracy (%)', xlabel='Time from object entry (s)',
        yticks=[-0.2, 0, 0.2, 0.4])
ax1.legend(prop={'size': 5}).set_title('')

sns.despine(trim=True)
plt.tight_layout()


# %%

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(context_onset_df, x='time', y='accuracy', hue='region', ax=ax1, errorbar='se',
             zorder=1, err_kws={'lw': 0})
ax1.plot(ax1.get_xlim(), [0.5, 0.5], ls='--', color='grey', zorder=0, lw=0.75)
ax1.set(ylim=[0.4, 0.95], ylabel='Context decoding accuracy (%)', xticks=[-1, 0, 1, 2, 3],
        yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], xlabel='Time from sound onset (s)')
ax1.legend().set_title('')

sns.despine(trim=True)
plt.tight_layout()

# %%

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(obj_id_df, x='time', y='accuracy', hue='region', ax=ax1, errorbar='se',
             zorder=1, err_kws={'lw': 0})
ax1.plot(ax1.get_xlim(), [0.33, 0.33], ls='--', color='grey', zorder=0, lw=0.75)
ax1.set(ylim=[0.3, 0.7], ylabel='Object decoding accuracy (%)', xticks=[-2, -1, 0, 1],
        yticks=[0.3, 0.4, 0.5, 0.6, 0.7], xlabel='Time from object entry (s)')
ax1.legend().set_title('')

sns.despine(trim=True)
plt.tight_layout()

# %%

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(context_env_df, x='time', y='accuracy', hue='region', ax=ax1, errorbar='se',
             zorder=1, err_kws={'lw': 0})
ax1.plot(ax1.get_xlim(), [0.5, 0.5], ls='--', color='grey', zorder=0, lw=0.75)
ax1.set(ylim=[0.3, 0.9], ylabel='Object decoding accuracy (%)', xticks=[0, 50, 100, 150],
        yticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], xlabel='Distance from environment entry (cm)')
ax1.legend().set_title('')

sns.despine(trim=True)
plt.tight_layout()


