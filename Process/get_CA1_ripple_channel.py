# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:55:05 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_neural_data, figure_style
colors, dpi = figure_style()

# Settings
PLOT = True
LOW_CUT = 150
HIGH_CUT = 250
FS = 2500
PERC_EXCL = 20  # percentage of bottom channels to exclude when picking maximum power depth

# Get paths
path_dict = paths()
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)

rec_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'Recording {i} of {len(rec)}')
 
    # Check if LFP is on disk
    ses_path = path_dict['local_data_path'] / 'Subjects' / subject / date 
    probe_path = path_dict['local_data_path'] / 'Subjects' / subject / date / probe 
    lfp_path = path_dict['local_data_path'] / 'Subjects' / subject / date / probe / 'lfp_raw_binary'
    if lfp_path.is_dir() == False:
        continue
 
    # Get channels in CA1
    _, _, channels = load_neural_data(ses_path, probe)
    ca1_channels = np.where(channels['acronym'] == 'CA1')[0]
    if ca1_channels.shape[0] == 0:
        print('No channels in CA1')
        continue
    ca1_ch_depth = channels['axial_um'][ca1_channels]
    
    # Load in LFP spectrum
    lf_psd = {'power': np.load(probe_path / '_iblqc_ephysSpectralDensityLF.power.npy'),
              'freqs': np.load(probe_path / '_iblqc_ephysSpectralDensityLF.freqs.npy')}
    ca1_ch_power = np.mean(lf_psd['power'][np.ix_((lf_psd['freqs'] >= LOW_CUT) & (lf_psd['freqs'] <= HIGH_CUT),
                                                  channels['acronym'] == 'CA1')], axis=0)
   
    # Loop over shanks
    max_power, max_power_depth, max_channel = np.empty(4), np.empty(4), np.zeros(4).astype(int)
    all_shank_power, all_shank_depth = [], []
    for (this_shank, this_lateral) in zip([0, 1, 2, 3], [[0, 32], [250, 282], [500, 532], [750, 782]]):
        shank_channels = np.where(np.isin(channels['lateral_um'], this_lateral))[0]
        shank_ca1_channels = np.isin(ca1_channels, shank_channels)
        if np.sum(shank_ca1_channels) < 10:
            max_power[this_shank] = np.nan
            all_shank_power.append([])
            all_shank_depth.append([])
            continue
        shank_power = ca1_ch_power[shank_ca1_channels]
        shank_depth = ca1_ch_depth[shank_ca1_channels]
        
        # Interpolate over outlier channels
        abs_diff = np.abs(np.diff(shank_power))
        outlier_ch = np.where(abs_diff > np.std(abs_diff)*3)[0] + 1
        for jj in outlier_ch:
            if jj + 1 >= shank_power.shape[0]:
                shank_power[jj] = shank_power[jj-1]
            else:
                shank_power[jj] = np.mean([shank_power[jj-1], shank_power[jj+1]])
    
        # Get mean per depth
        depth = np.unique(shank_depth)
        power_depth = np.array([np.mean(shank_power[shank_depth == i]) for i in depth])
    
        # Get max power channel         
        excl_n_ch = np.round(len(power_depth)*(PERC_EXCL/100)).astype(int)
        max_ind = np.argmax(power_depth[excl_n_ch:]) + excl_n_ch
        max_power[this_shank] = power_depth[max_ind]
        max_power_depth[this_shank] = depth[max_ind]
        max_channel[this_shank] = ca1_channels[shank_ca1_channels][shank_depth == depth[max_ind]][0]
        
        # Add to list
        all_shank_power.append(power_depth)
        all_shank_depth.append(depth)
        
    # Get the shank with the highest max power
    use_shank = np.nanargmax(max_power)
    use_channel = max_channel[use_shank]
    
    # Add to dataframe
    rec_df = pd.concat((rec_df, pd.DataFrame(index=[rec_df.shape[0]+1], data={
        'subject': subject, 'date': date, 'probe': probe, 'max_channel': use_channel})))
   
    # Plot
    if PLOT:
        f, axs = plt.subplots(1, 4, figsize=(1.75*4, 2.5), dpi=dpi, sharey=True, sharex=True)
        
        for sh in range(4):
            axs[sh].plot(all_shank_power[sh], all_shank_depth[sh], lw=1, zorder=0)
            axs[sh].scatter(max_power[sh], max_power_depth[sh], marker='x', color='red', lw=1.5, zorder=1)
            axs[sh].set(title=f'Shank {sh}', xlabel='Ripple band power')
            if sh == 0:
                axs[sh].set(ylabel='Distance from tip of probe (um)')
            if sh == use_shank:
                axs[sh].set(title=f'Shank {sh} (MAX)')
        
        sns.despine(trim=True)
        plt.tight_layout()
        
        plt.savefig(path_dict['fig_path'] / 'RipplePower' / f'{subject}_{date}_{probe}.jpg', dpi=600)
        plt.close(f)
     
# Save to disk
rec_df.to_csv(path_dict['save_path'] / 'ripple_channel.csv', index=False)

    