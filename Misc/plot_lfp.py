# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 10:24:57 2026

By Guido Meijer
"""

import numpy as np
import pandas as pd
from msvr_functions import paths, load_neural_data
import spikeinterface.full as si
path_dict = paths()

# Settings
SUBJECT = '478154'
DATE = '20251003'
PROBE = 'probe01'

# Get max channel data
rec_df = pd.read_csv(path_dict['save_path'] / 'ripple_channel.csv')
rec_df['subject'] = rec_df['subject'].astype(str)
rec_df['date'] = rec_df['date'].astype(str)

ses_path = path_dict['local_data_path'] / 'Subjects' / SUBJECT / DATE
rec = si.read_binary(ses_path / PROBE / 'lfp_raw_binary' / 'traces_cached_seg0.raw',
                     sampling_frequency=2500, dtype='int16', num_channels=384)
max_channel = rec_df.loc[(rec_df['subject'] == SUBJECT) & (rec_df['date'] == DATE) & (rec_df['probe'] == PROBE),
                         'max_channel'].values[0]
_, _, channels = load_neural_data(ses_path, PROBE)

# Get channels to plot
column_channels = np.where(channels['lateral_um'] == channels['lateral_um'][max_channel])[0]

# Plot
w = si.plot_traces(recording=rec, backend="matplotlib", mode='line', channel_ids=column_channels,
                   time_range=[rec.get_duration()-800, rec.get_duration()-790], show_channel_ids=True)





