# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 09/06/2026
"""
# %%

import numpy as np
import pandas as pd
from msvr_functions import paths

# Load in ripple times
path_dict = paths()
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv', dtype={'subject': str, 'date': str})

# Loop over sessions
for date in np.unique(ripples['date']):
    print('Processing session', date)
    ses_df = ripples[ripples['date'] == date]
    for probe in np.unique(ses_df['probe']):

        # Get slice of df
        probe_df = ses_df[ses_df['probe'] == probe]
        subject = probe_df['subject'].values[0]

        # Save ripple times to local disk and server
        start_times = probe_df['start_times'].values
        np.save(path_dict['local_data_path'] / 'Subjects' / subject / date / probe / 'ripples.startTimes.npy', start_times)
        np.save(path_dict['server_path'] / 'Subjects' / subject / date / probe / 'ripples.startTimes.npy', start_times)
        end_times = probe_df['end_times'].values
        np.save(path_dict['server_path'] / 'Subjects' / subject / date / probe / 'ripples.endTimes.npy', end_times)

print('Done')



