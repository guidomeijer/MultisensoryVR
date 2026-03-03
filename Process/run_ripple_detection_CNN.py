# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:14:58 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow.keras as kr
from msvr_functions import paths, load_lfp, load_neural_data, load_trials, get_ripple_channels
from cnn_load_data import z_score_normalization, downsample_data, generate_overlapping_windows
from format_predictions import get_predictions_indexes


# Settings
WIN_SIZE = 0.0128
STRIDE = 0.0064
THRESHOLD = 0.98
ORIG_FS = 2500
DS_FS = 1250
MERGE_TIME = 100  # ms
MIN_DURATION = 15 # ms
N_CHANNEL_SLIDE = 8
PLOT = True

# Get paths
path_dict = paths()
rec = pd.read_csv(path_dict['save_path'] / 'ripple_channel.csv')
rec['subject'] = rec['subject'].astype(str)
rec['date'] = rec['date'].astype(str)

#rec = rec.loc[(rec['subject'] == '462910') & (rec['date'] == '20240815')]
rec = rec.iloc[1:]

# Initialize model
optimizer = kr.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                               amsgrad=False)
model = kr.models.load_model(r'C:\Users\Guido1\Repositories\cnn-ripple\model', compile=False)
model.compile(loss="binary_crossentropy", optimizer=optimizer)

ripple_df = pd.DataFrame()
for i, (subject, date, probe, max_channel) in enumerate(zip(rec['subject'], rec['date'], rec['probe'], rec['max_channel'])):
    print(f'\nRecording {i} of {len(rec)}')
    
    # Load in data for this recording
    ses_path = path_dict['local_data_path'] / 'Subjects' / subject / date
    trials = load_trials(subject, date)
    _, _, channels = load_neural_data(ses_path, probe)
    
    # Take channels around max channel, alternating left and right per row
    shank_channels = np.where(np.abs(channels['lateral_um'] - channels['lateral_um'][max_channel]) < 50)[0]
    column_channels = np.where(channels['lateral_um'] == channels['lateral_um'][max_channel])[0]
    
    # Do SWR detection for a bunch of channels around the max channel to see which one gives the best result
    max_ch_ind = np.where(column_channels == max_channel)[0][0]
    all_pred_indexes, all_pred_times, all_pred_prob, processed_channels = [], [], [], []
    for this_channel in column_channels[max_ch_ind-N_CHANNEL_SLIDE:max_ch_ind+N_CHANNEL_SLIDE]:
        print(f'Processing channel {this_channel}..')
        
        
        """
        # Select channels above and below in the same column while skipping each other channel vertically
        idx = np.where(column_channels == this_channel)[0][0]
        channels_above = column_channels[idx-8 : idx : 2]
        channels_below = column_channels[idx+2 : idx+7 : 2]
        use_channels = np.concatenate((channels_above, [max_channel], channels_below))
        """
        # Select channels around max channel in alteranating pattern
        use_channels = get_ripple_channels(this_channel, 3, 4, channels, shank_channels)
        if use_channels.shape[0] != 8:
            continue
        
        # Load in lfp
        lfp, timestamps = load_lfp(ses_path / probe, use_channels,
                                   start_time=trials['exitEnvTime'].values[-1] + 60)
        if len(lfp) == 0:
            continue
        
        # Downsample and Z-score
        lfp_ds = downsample_data(lfp, ORIG_FS, DS_FS)
        timestamps_ds = np.linspace(0, timestamps[-1] - timestamps[0], lfp_ds.shape[0])
        lfp_zscore = z_score_normalization(lfp_ds)
        
        # Separate the data into 12.8ms windows with 6.4ms overlapping
        X = generate_overlapping_windows(lfp_zscore, WIN_SIZE, STRIDE, DS_FS)
        
        # Detect ripples
        predictions = model.predict(X, verbose=True)
        
        # Threshold predictions
        pred_indexes, pred_prob = get_predictions_indexes(lfp_zscore, predictions, window_size=WIN_SIZE,
                                                          stride=STRIDE, fs=DS_FS, threshold=THRESHOLD)
        if pred_indexes.shape[0] == 0:
            print('No ripples detected')
            continue
        
        # Merge overlapping predicted time windows
        merged_begin_bool = np.concatenate(([True], ~(np.diff(pred_indexes[:,0]) <= MERGE_TIME)))
        merged_end_bool = np.concatenate((~(np.diff(pred_indexes[:,0]) <= MERGE_TIME), [True]))
        merged_begin_ind = np.where(merged_begin_bool)[0]
        merged_end_ind = np.where(merged_end_bool)[0]
        pred_prob = np.array([np.mean(pred_prob[i[0]:(i[1]+1)]) for i in zip(merged_begin_ind, merged_end_ind)])
        pred_indexes = np.vstack((pred_indexes[merged_begin_bool, 0], pred_indexes[merged_end_bool, 1])).T
           
        # Get times 
        pred_times = (pred_indexes / DS_FS) + timestamps[0]
        
        # Drop ripples that are too short
        keep_ripples = (pred_times[:, 1] - pred_times[:, 0]) > (MIN_DURATION / 1000)
        pred_times = pred_times[keep_ripples, :]
        pred_indexes = pred_indexes[keep_ripples, :]
        pred_prob = pred_prob[keep_ripples]
        
        # Add to lists
        print(f'Detected {pred_times.shape[0]} ripples')
        all_pred_times.append(pred_times)
        all_pred_indexes.append(pred_indexes)
        all_pred_prob.append(pred_prob)
        processed_channels.append(this_channel)
    
    # Select channel with most sharp wave ripples
    if len(all_pred_prob) == 0:
        continue
    n_ripples = np.array([i.shape[0] for i in all_pred_prob])
    best_channel = processed_channels[np.argmax(n_ripples)]
    pred_times = all_pred_times[np.argmax(n_ripples)]
    pred_indexes = all_pred_indexes[np.argmax(n_ripples)]
    pred_prob = all_pred_prob[np.argmax(n_ripples)]
    
    # Add to dataframe and save
    ripple_df = pd.concat((ripple_df, pd.DataFrame(data={
        'start_times': pred_times[:,0], 'end_times': pred_times[:,1],
        'start_indexes': pred_indexes[:,0], 'end_indexes': pred_indexes[:,1],
        'prob': pred_prob, 'best_channel': best_channel,
        'subject': subject, 'date': date, 'probe': probe})))
    ripple_df.to_csv(path_dict['save_path'] / 'ripples.csv', index=False)
        
      
    # %% Plot a selection of detected ripples
    if PLOT:
        """
        # Plot traces with ripples
        import spikeinterface.full as si
        rec_lfp = si.read_binary(ses_path / probe / 'lfp_raw_binary' / 'traces_cached_seg0.raw',
                                 sampling_frequency=2500, dtype='int16', num_channels=384)
        w = si.plot_traces(recording=rec_lfp, 
                           backend='matplotlib',
                           mode='line',
                           channel_ids=column_channels,
                           time_range=[rec_lfp.get_duration()-800, rec_lfp.get_duration()-790],
                           events=pred_times[:, 0],
                           show_channel_ids=True)
        """
        # Plot individual ripples
        if pred_prob.shape[0] > 200:
            plot_ripples = np.random.choice(pred_prob.shape[0], size=100, replace=False)
        elif pred_prob.shape[0] > 0:
            plot_ripples = np.arange(pred_prob.shape[0])
        else:
            continue
        
        (path_dict['fig_path'] / 'RippleDetection' / f'{subject}_{date}_{probe}').mkdir(exist_ok=True)
        for j in plot_ripples:
            f, ax = plt.subplots(figsize=(3, 7))
            ax.add_patch(Rectangle((100, -35), pred_indexes[j, 1]-pred_indexes[j, 0], 40,
                                   color='royalblue', alpha=0.25, lw=0))
            for jj in range(lfp_zscore.shape[1]):
                this_lfp = lfp_zscore[pred_indexes[j, 0]-100:pred_indexes[j, 1]+100, jj].copy()
                this_lfp -= jj*4
                ax.plot(this_lfp, color='k')
            ax.set(title=f'prob: {pred_prob[j]:.2f}, duration: {(pred_times[j,1]-pred_times[j,0])*1000:.1f} ms')
            ax.axis('off')
            plt.savefig(path_dict['fig_path'] / 'RippleDetection' / f'{subject}_{date}_{probe}' / f'{j}.jpg',
                        dpi=300)
            plt.close(f)
            
    # Clear memory
    del lfp, lfp_ds, lfp_zscore
        