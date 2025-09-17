# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:38:04 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from spikeinterface import load_sorting_analyzer
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import seaborn as sns
from msvr_functions import paths, load_neural_data, figure_style

colors, dpi = figure_style()
path_dict = paths()
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)

waveform_si_df, waveform_calc_df = pd.DataFrame(), pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'Session {i} of {rec.shape[0]}')
    
    # Load in waveform metrics
    sort_path = path_dict['server_path'] / 'Subjects' / subject / date / probe / 'sorting'
    ses_path = path_dict['local_data_path'] / 'Subjects' / subject / date 
    waveform_metrics_si = pd.read_csv(sort_path / 'extensions' / 'template_metrics' / 'metrics.csv')
    waveform_metrics_si = waveform_metrics_si.rename(columns={'Unnamed: 0': 'neuron_id'})
        
    # Load in region
    spikes, clusters, channels = load_neural_data(ses_path, probe, only_good=False, min_fr=0)
        
    # Add to dataframe
    waveform_metrics_si['region'] = clusters['region']
    waveform_metrics_si['firing_rate'] = clusters['firing_rate']
    waveform_metrics_si['good'] = ((clusters['ibl_label'] == 1) | (clusters['ml_label'] == 1)).astype(int)
    waveform_metrics_si['subject'] = subject
    waveform_metrics_si['date'] = date
    waveform_metrics_si['probe'] = probe
    waveform_si_df = pd.concat((waveform_si_df, waveform_metrics_si), ignore_index=True)
    
    # Now calculate everything from the waveform
    sorting_analyzer = load_sorting_analyzer(sort_path)
    n_units = sorting_analyzer.get_num_units()
    wf_ext = sorting_analyzer.get_extension('waveforms')
    pt_ratio, spike_width, rp_slope, rc_slope = np.empty(n_units), np.empty(n_units), np.empty(n_units), np.empty(n_units)
    for n in range(n_units):
        unit_wfs = wf_ext.get_waveforms_one_unit(n, force_dense=False)
        mean_wf_all_ch = np.mean(unit_wfs, axis=0)
        max_channel = np.argmin(np.min(mean_wf_all_ch, axis=0))
        mean_wf = mean_wf_all_ch[:, max_channel]
        wf_time = np.linspace(0, (mean_wf.shape[0] / 30000) * 1000, mean_wf.shape[0])
        
        # Get part of spike from trough to first peak after the trough
        peak_after_trough = np.argmax(mean_wf[np.argmin(mean_wf):]) + np.argmin(mean_wf)
        repolarization = mean_wf[np.argmin(mean_wf):np.argmax(mean_wf[np.argmin(mean_wf):]) + np.argmin(mean_wf)]

        # Get peak-to-trough ratio
        pt_ratio[n] = mean_wf[peak_after_trough] / np.abs(np.min(mean_wf))
        
        # Get spike width in ms
        spike_width[n] = ((peak_after_trough - np.argmin(mean_wf)) / 30000) * 1000
        
        # Get repolarization slope
        if spike_width[n] <= 0.08:
            continue
        else:
            rp_slope[n], _, = np.polyfit(wf_time[np.argmin(mean_wf):peak_after_trough],
                                         mean_wf[np.argmin(mean_wf):peak_after_trough], 1)

        # Get recovery slope
        rc_slope[n], _ = np.polyfit(wf_time[peak_after_trough:], mean_wf[peak_after_trough:], 1)

    # Add to dataframe
    waveform_metrics_calc = pd.DataFrame(data={
        'unit_id': sorting_analyzer.unit_ids,
        'spike_width': spike_width,
        'pt_ratio': pt_ratio,
        'rp_slope': rp_slope,
        'rc_slope': rc_slope,
        'region': clusters['region'],
        'firing_rate': clusters['firing_rate'],
        'good': ((clusters['ibl_label'] == 1) | (clusters['ml_label'] == 1)).astype(int),
        'subject': subject,
        'date': date,
        'probe': probe
        })
    waveform_calc_df = pd.concat((waveform_calc_df, waveform_metrics_calc), ignore_index=True)
    
    # Save result
    waveform_si_df.to_csv(path_dict['save_path'] / 'waveform_metrics_si.csv', index=False)
    waveform_calc_df.to_csv(path_dict['save_path'] / 'waveform_metrics_calc.csv', index=False)

