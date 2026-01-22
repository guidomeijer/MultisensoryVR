# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:14:58 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import brainbox.io.one as bbone
import tensorflow.keras as kr
from msvr_functions import paths, load_lfp, load_neural_data
from cnn_load_data import z_score_normalization, downsample_data, generate_overlapping_windows
from format_predictions import get_predictions_indexes

# Settings
WIN_SIZE = 0.0128
STRIDE = 0.0064
THRESHOLD = 0.95
ORIG_FS = 2500
DS_FS = 1250
MERGE_TIME = 15  # ms
PLOT = False
OVERWRITE = True

# Get paths
path_dict = paths()
rec = pd.read_csv(path_dict['save_path'] / 'ripple_channel.csv')
rec['subject'] = rec['subject'].astype(str)
rec['date'] = rec['date'].astype(str)

# Initialize model
optimizer = kr.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                               amsgrad=False)
model = kr.models.load_model(r'C:\Users\Guido1\Repositories\cnn-ripple\model', compile=False)
model.compile(loss="binary_crossentropy", optimizer=optimizer)

ripple_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'Recording {i} of {len(rec)}')
    
    # Load in data for this recording
    ses_path = path_dict['local_data_path'] / 'Subjects' / subject / date
    _, _, channels = load_neural_data(ses_path, probe)
    max_channel = rec.loc[i, 'max_channel']
    
    # TO DO: select 8 channels with 40 um between
    # Select 8 channels including and beneath the channel with the max LFP amplitude
    these_channels = np.where(((channels['lateral_um'] == channels['lateral_um'][max_channel])
                               & (channels['axial_um'] < channels['axial_um'][max_channel]+(4*15))
                               & (channels['axial_um'] >= channels['axial_um'][max_channel]-(4*15))))[0]
    
    # Load in lfp
    print('Loading in LFP')
    lfp, timestamps = load_lfp(ses_path / probe, np.flip(these_channels))
    if len(lfp) == 0:
        continue
    
    # Downsample and Z-score
    lfp_ds = downsample_data(lfp, ORIG_FS, DS_FS)
    timestamps_ds = np.linspace(0, timestamps[-1] - timestamps[0], lfp_ds.shape[0])
    lfp_zscore = z_score_normalization(lfp_ds)
    
    # Generate overlapping windows
    print('Generating windows..')
    
    # Separate the data into 12.8ms windows with 6.4ms overlapping
    X = generate_overlapping_windows(lfp_zscore, WIN_SIZE, STRIDE, DS_FS)
    
    # Detect ripples
    print('Detecting ripples...')
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
    
    # Add to dataframe and save
    print(f'Detected {pred_times.shape[0]} ripples')
    ripple_df = pd.concat((ripple_df, pd.DataFrame(data={
        'start_times': pred_times[:,0], 'end_times': pred_times[:,1],
        'start_indexes': pred_indexes[:,0], 'end_indexes': pred_indexes[:,1],
        'prob': pred_prob, 'subject': subject, 'date': date, 'probe': probe})))
    ripple_df.to_csv(path_dict['save_path'] / 'ripples.csv', index=False)
        
      
    # %% Plot detected ripples
    if PLOT:
        if not (path_dict['fig_path'] / 'RippleDetection' / f'{subject}_{date}').is_dir():
            (path_dict['fig_path'] / 'RippleDetection' / f'{subject}_{date}_{probe}').mk_dir()
        for j in range(pred_indexes.shape[0]):
            f, ax = plt.subplots(figsize=(3, 7))
            ax.add_patch(Rectangle((100, -35), pred_indexes[j, 1]-pred_indexes[j, 0], 40,
                                   color='royalblue', alpha=0.25, lw=0))
            for jj in range(lfp_zscore.shape[1]):
                this_lfp = lfp_zscore[pred_indexes[j, 0]-100:pred_indexes[j, 1]+100, jj].copy()
                this_lfp -= jj*4
                ax.plot(this_lfp, color='k')
            ax.set(title=f'prob: {pred_prob[j]:.2f}, duration: {(pred_times[j,1]-pred_times[j,0])*1000:.1f} ms')
            ax.axis('off')
            plt.savefig(path_dict['fig_path'] / 'RippleDetection' / f'{subject}_{date}_{probe}', f'{j}.jpg',
                             dpi=600)
            plt.close(f)
            
    # Clear memory
    del lfp, lfp_ds, lfp_zscore
        