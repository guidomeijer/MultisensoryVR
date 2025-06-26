# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:07:37 2025

By Guido Meijer
"""

from os import path
import numpy as np
import pandas as pd
from glob import glob
from msvr_functions import paths
import spikeinterface.full as si


def main():

    path_dict = paths(sync=False)
    rec = pd.read_csv(path.join(path_dict['repo_path'], 'recordings.csv')).astype(str)
    
    for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
        print(f'{subject} {date} {probe}')
        
        # Get paths
        server_path = path.join(path_dict['server_path'], 'Subjects', subject, date, 'raw_ephys_data', probe)
        local_path = path.join(path_dict['local_data_path'], 'Subjects', subject, date, probe)
        
        # Check if there is a bin file
        if len(glob(path.join(server_path, '*bin'))) == 0:
            print('No bin file')
            continue
        
        # Load in raw data using SpikeInterface 
        if len(glob(path.join(server_path, '*.cbin'))) > 0:
            rec = si.read_cbin_ibl(server_path)
        else:
            rec = si.read_spikeglx(server_path, stream_id=si.get_neo_streams('spikeglx', server_path)[0][0])
        
        # Check if NP2 recording
        if np.unique(rec.get_property('group')).shape[0] == 1:
            print('Single shank recording')
            continue
        
        # Check if already done
        if path.isdir(path.join(local_path, 'lfp_raw_binary')):
            print('Already done')
            continue
        
        # Filter out LFP band
        rec_lfp = si.bandpass_filter(rec, freq_min=1, freq_max=400)
        
        # Correct for inter-sample shift
        rec_shifted = si.phase_shift(rec_lfp)    
        
        # Interpolate over bad channels  
        rec_car_temp = si.common_reference(rec_lfp)
        _, all_channels = si.detect_bad_channels(
            rec_car_temp, method='mad', std_mad_threshold=3, seed=42)
        noisy_channel_ids = rec_car_temp.get_channel_ids()[all_channels == 'noise']           
        rec_interpolated = si.interpolate_bad_channels(rec_shifted, noisy_channel_ids)
        
        # Do common average reference
        rec_car = si.common_reference(rec_interpolated)
        
        # Downsample to 2500 Hz
        rec_final = si.resample(rec_car, 2500)
        
        # Save raw binary to disk
        rec_final.save(folder=path.join(local_path, 'lfp_raw_binary'), format='binary', chunk_duration='1s',
                       dtype='int16', n_jobs=-1)
    
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()    
    main()