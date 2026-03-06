# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 05/03/2026
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si

# Settings
SES_PATH = Path(r'V:\imaging1\guido\Subjects\466396\20241031\probe01')
PLOT_SHANK = 1

# Load in LFP data
rec = si.read_binary(SES_PATH / 'lfp_raw_binary' / 'traces_cached_seg0.raw',
                     sampling_frequency=2500, dtype='int16', num_channels=384)

# Load in channel locations
channel_locs = np.load(SES_PATH / 'channels.localCoordinates.npy').astype(int)
shank_map = {
    1: 0,
    2: 250,
    3: 500,
    4: 750
}
channel_ids = rec.get_channel_ids()
shank_channels = channel_ids[np.isin(channel_locs[:, 0], shank_map[PLOT_SHANK])]

# %% Plot
PLOT_DURATION = 3  # s
TIME_FROM_END = (12 * 60) + 10 # s
w = si.plot_traces(recording=rec, backend='matplotlib', mode='line', channel_ids=shank_channels,
                   time_range=[rec.get_duration()-TIME_FROM_END, rec.get_duration()-(TIME_FROM_END - PLOT_DURATION)],
                   show_channel_ids=True, figsize=(12, 8), vspacing_factor=0.8)
plt.show()

# %% This is how you get the LFP traces from certain channels
LAST_MIN = 5  # last minutes of the recording
GET_CHANNELS = [30, 32, 34]
start_frame = rec.time_to_sample_index(rec.get_duration()-(LAST_MIN * 60))
lfp_traces = rec.get_traces(start_frame=start_frame, end_frame=rec.get_total_samples(), channel_ids=GET_CHANNELS)