# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 20/02/2026
"""
# %%

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from msvr_functions import paths, load_objects, figure_style, bin_signal, peri_event_trace

# Settings
MIN_RIPPLES = 10
SMOOTHING = 1
OBJ_WIN = 0.3
RIPPLE_WIN_CENTERS = np.arange(-1.5, 1.6, 0.1)
RIPPLE_WIN = 0.3
REGIONS = ['CA1', 'LEC', 'PERI', 'TEa', 'AUD', 'VIS']

# Initialize
path_dict = paths()
colors, dpi = figure_style()

# Load in all ripples
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)

# Load in significant assemblies
assembly_sig = pd.read_csv(path_dict['save_path'] / 'assembly_sig.csv')
assembly_sig['subject'] = assembly_sig['subject'].astype(str)
assembly_sig['date'] = assembly_sig['date'].astype(str)

# Get paths to data of this session
amp_paths = list((path_dict['google_drive_data_path'] / 'Assemblies').glob(f'*.amplitudes.npy'))

# Loop over regions and get spike pattern amplitudes
lda_df = pd.DataFrame()
for i, amp_path in enumerate(amp_paths):

    # Load in data for this region
    subject = amp_path.stem.split('.')[0].split('_')[0]
    date = amp_path.stem.split('.')[0].split('_')[1]
    probe = amp_path.stem.split('.')[0].split('_')[2]
    region = amp_path.stem.split('.')[0].split('_')[3]
    if np.mod(i, 10) == 0:
        print(f'Processing {i} of {len(amp_paths)}')
    amplitudes = np.load(amp_path)
    time_ax = np.load(amp_path.parent / (amp_path.stem.split('.')[0] + '.times.npy'))
    all_obj_df = load_objects(subject, date)
    all_obj_df = all_obj_df[all_obj_df['object'] == 1]
    #all_obj_df = all_obj_df[(all_obj_df['object'] == 1) | (all_obj_df['object'] == 2)]

    # Get ripples for this session
    ripples_ses = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
    if ripples_ses.shape[0] < MIN_RIPPLES:
        continue
    ripples_ses = ripples_ses[ripples_ses['start_times'] < time_ax[-1] - RIPPLE_WIN_CENTERS[-1]]

    # Select signficant assemblies
    these_assemblies = assembly_sig[(assembly_sig['region'] == region) & (assembly_sig['subject'] == subject)
                                    & (assembly_sig['date'] == date) & (assembly_sig['probe'] == probe)]
    amplitudes = amplitudes[these_assemblies.loc[these_assemblies['p_ripples'] < 0.05, 'assembly'].values - 1]
    n_assemblies = amplitudes.shape[0]
    if n_assemblies == 0:
        continue

    # Apply Gaussian smoothing to the assembly activation rates
    if SMOOTHING > 0:
        amplitudes = gaussian_filter1d(amplitudes, sigma=SMOOTHING, axis=1)

    # Fit LDA projection to hit-miss axis during object entries
    # Construct X array
    obj_amp = np.full((all_obj_df.shape[0], n_assemblies), np.nan)
    for asmbl in range(n_assemblies):

        obj_goal = all_obj_df.loc[all_obj_df['goal'] == 1, 'times'].values
        obj_no_goal = all_obj_df.loc[all_obj_df['goal'] == 0, 'times'].values
        obj_goal_amp = bin_signal(time_ax, amplitudes[asmbl, :], obj_goal, bin_size=OBJ_WIN)
        obj_no_goal_amp = bin_signal(time_ax, amplitudes[asmbl, :], obj_no_goal, bin_size=OBJ_WIN)
        obj_goal_no_goal = np.concatenate((obj_goal_amp, obj_no_goal_amp))
        obj_amp[:, asmbl] = obj_goal_no_goal

    # Get trial labels
    y_goal = np.concatenate((np.ones(all_obj_df[all_obj_df['goal'] == 1].shape[0]),
                             np.zeros(all_obj_df[all_obj_df['goal'] == 0].shape[0])))

    # Fit LDA to object entry
    obj_lda = LDA(n_components=1, priors=[0.5, 0.5])
    obj_lda.fit(obj_amp, y_goal)

    # Project ripples to LDA axis
    lda_align_abs, lda_align = np.zeros(RIPPLE_WIN_CENTERS.shape[0]), np.zeros(RIPPLE_WIN_CENTERS.shape[0])
    for k, win_center in enumerate(RIPPLE_WIN_CENTERS):

        # Get assembly amplitudes around ripples
        ripple_amp = np.full((ripples_ses.shape[0], n_assemblies), np.nan)
        for asmbl in range(n_assemblies):
            ripple_amp[:, asmbl] = bin_signal(time_ax, amplitudes[asmbl, :], ripples_ses['start_times'] + win_center,
                                              bin_size=RIPPLE_WIN)

        # Project to LDA axis
        obj_proj = obj_lda.transform(ripple_amp)
        lda_axis = obj_lda.coef_[0]

        # Calculate alignment per ripple
        alignment, abs_alignment = np.zeros(ripple_amp.shape[0]), np.zeros(ripple_amp.shape[0])
        for r in range(ripple_amp.shape[0]):
            dot_product = np.dot(ripple_amp[r, :], lda_axis)
            norm_pattern = np.linalg.norm(ripple_amp[r, :])
            alignment[r] = dot_product / (norm_pattern * np.linalg.norm(lda_axis))
            abs_alignment[r] = np.abs(dot_product / (norm_pattern * np.linalg.norm(lda_axis)))
        lda_align[k] = np.mean(alignment)
        lda_align_abs[k] = np.mean(abs_alignment)

    # Do baseline subtraction
    lda_align = lda_align - np.mean(lda_align[RIPPLE_WIN_CENTERS < -1])
    lda_align_abs = lda_align_abs - np.mean(lda_align_abs[RIPPLE_WIN_CENTERS < -1])

    # Add to dataframe
    lda_df = pd.concat((lda_df, pd.DataFrame(data={
        'lda_align': lda_align, 'lda_align_abs': lda_align_abs, 'time_ax': RIPPLE_WIN_CENTERS,
        'region': region, 'subject': subject, 'date': date})))


# %% Plot

f, axs = plt.subplots(1, 6, figsize=(7, 1.75), dpi=dpi)
for r, region in enumerate(REGIONS):
    axs[r].plot([-1, 1], [0, 0], color='darkgrey', ls='--')
    sns.lineplot(data=lda_df[lda_df['region'] == region], x='time_ax', y='lda_align_abs', errorbar='se', ax=axs[r],
                 err_kws={'lw': 0})
    axs[r].set(ylabel='', xlabel='', title=region, xlim=[-1, 1])
axs[0].set(ylabel='Absolute LDA alignment')

sns.despine(trim=True)
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.85, wspace=0.5)
plt.show()

f, axs = plt.subplots(1, 6, figsize=(7, 1.75), dpi=dpi)
for r, region in enumerate(REGIONS):
    axs[r].plot([-1, 1], [0, 0], color='darkgrey', ls='--')
    sns.lineplot(data=lda_df[lda_df['region'] == region], x='time_ax', y='lda_align', errorbar='se', ax=axs[r],
                 err_kws={'lw': 0})
    axs[r].set(ylabel='', xlabel='', title=region, xlim=[-1, 1])
axs[0].set(ylabel='Signed LDA alignment')

f.text(0.5, 0.04, 'Time from ripple onset (s)', ha='center', va='center')

sns.despine(trim=True)
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.85, wspace=0.5)
plt.show()
