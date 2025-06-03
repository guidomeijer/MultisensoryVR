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
per_obj_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_per_object_reward_times.csv'))
per_obj_df = per_obj_df[per_obj_df['region'] != 'ENT']

per_obj_dist_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_per_object_distance.csv'))
per_obj_dist_df = per_obj_dist_df[per_obj_dist_df['region'] != 'ENT']

# %%
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75*3, 1.75), dpi=dpi)
sns.lineplot(per_obj_df[per_obj_df['object'] == 1], x='time', y='accuracy', hue='region', errorbar='se',
             ax=ax1, err_kws={'lw': 0}, palette=colors, legend=None)

sns.lineplot(per_obj_df[per_obj_df['object'] == 2], x='time', y='accuracy', hue='region', errorbar='se',
             ax=ax2, err_kws={'lw': 0}, palette=colors)

sns.lineplot(per_obj_df[per_obj_df['object'] == 3], x='time', y='accuracy', hue='region', errorbar='se',
             ax=ax3, err_kws={'lw': 0}, palette=colors)

sns.despine(trim=True)
plt.tight_layout()

# %%

f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'dCA1', 'vCA1']):
    axs[i].plot([-2, 2], [0.5, 0.5], ls='--', color='grey')
    axs[i].plot([0,0], [0.4, 0.8], ls='--', color='grey')
    sns.lineplot(per_obj_df[(per_obj_df['object'] == 1) & (per_obj_df['region'] == region)],
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

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_context_obj1.pdf'))

# %%

f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'dCA1', 'vCA1']):
    axs[i].plot([-2, 2], [0.5, 0.5], ls='--', color='grey')
    axs[i].plot([0,0], [0.4, 0.8], ls='--', color='grey')
    sns.lineplot(per_obj_df[(per_obj_df['object'] == 2) & (per_obj_df['region'] == region)],
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

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_context_obj2.pdf'))

# %%

f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'dCA1', 'vCA1']):
    axs[i].plot([-2, 2], [0.5, 0.5], ls='--', color='grey')
    axs[i].plot([0,0], [0.4, 0.8], ls='--', color='grey')
    sns.lineplot(per_obj_df[(per_obj_df['object'] == 3) & (per_obj_df['region'] == region)],
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

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_context_obj3.pdf'))
    

# %%

f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'dCA1', 'vCA1']):
    axs[i].plot([-150, 150], [0.5, 0.5], ls='--', color='grey')
    axs[i].plot([0,0], [0.4, 0.8], ls='--', color='grey')
    sns.lineplot(per_obj_dist_df[(per_obj_dist_df['object'] == 1) & (per_obj_dist_df['region'] == region)],
                 x='distance', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None)
    axs[i].set_title(region)
    axs[i].axis('off')

axs[0].plot([-150, -100], [0.34, 0.34], color='k', lw=1)
axs[0].text(-125, 0.3, '5 cm', ha='center', va='center')
axs[0].plot([-150, -150], [0.4, 0.5], color='k', lw=1)
axs[0].text(-170, 0.45, '10%', ha='center', va='center', rotation=90)
axs[0].text(-200, 0.6, 'Context decoding accuracy', ha='center', va='center', rotation=90)
plt.subplots_adjust(left=0.05, bottom=None, right=0.99, top=0.85, wspace=0, hspace=None)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_context_dist_obj1.pdf'))

# %%

f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'dCA1', 'vCA1']):
    axs[i].plot([-150, 150], [0.5, 0.5], ls='--', color='grey')
    axs[i].plot([0,0], [0.4, 0.8], ls='--', color='grey')
    sns.lineplot(per_obj_dist_df[(per_obj_dist_df['object'] == 2) & (per_obj_dist_df['region'] == region)],
                 x='distance', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None)
    axs[i].set_title(region)
    axs[i].axis('off')

axs[0].plot([-2, -1], [0.4, 0.4], color='k', lw=1)
axs[0].text(-1.5, 0.375, '1s', ha='center', va='center')
axs[0].plot([-2, -2], [0.4, 0.5], color='k', lw=1)
axs[0].text(-2.4, 0.45, '10%', ha='center', va='center', rotation=90)
axs[0].text(-3.2, 0.6, 'Context decoding accuracy', ha='center', va='center', rotation=90)
plt.subplots_adjust(left=0.05, bottom=None, right=0.99, top=0.85, wspace=0, hspace=None)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_context_dist_obj2.pdf'))

# %%

f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'dCA1', 'vCA1']):
    axs[i].plot([-150, 150], [0.5, 0.5], ls='--', color='grey')
    axs[i].plot([0,0], [0.4, 0.8], ls='--', color='grey')
    sns.lineplot(per_obj_dist_df[(per_obj_dist_df['object'] == 3) & (per_obj_df['region'] == region)],
                 x='distance', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None)
    axs[i].set_title(region)
    axs[i].axis('off')

axs[0].plot([-2, -1], [0.4, 0.4], color='k', lw=1)
axs[0].text(-1.5, 0.375, '1s', ha='center', va='center')
axs[0].plot([-2, -2], [0.4, 0.5], color='k', lw=1)
axs[0].text(-2.4, 0.45, '10%', ha='center', va='center', rotation=90)
axs[0].text(-3.2, 0.6, 'Context decoding accuracy', ha='center', va='center', rotation=90)
plt.subplots_adjust(left=0.05, bottom=None, right=0.99, top=0.85, wspace=0, hspace=None)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_context_dist_obj3.pdf'))
    

