# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:39:26 2023

By Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import gzip, pickle, json
import matplotlib.pyplot as plt
import cebra
from os.path import join, realpath, dirname, isfile, split, isdir, basename

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'

# Get paths
with open(join(split(split(realpath(__file__))[0])[0], 'paths.json')) as json_file:
    path_dict = json.load(json_file)

# Load in data
with gzip.open(join(path_dict['local_data_path'], 'CEBRA',
                    f'{SUBJECT}_{DATE}_{PROBE}.pickle'), 'rb') as handle:
    data_dict = pickle.load(handle)    

# Initialize CEBRA
cebra_pos_model = cebra.CEBRA(model_architecture='offset10-model',
                              batch_size=512,
                              learning_rate=3e-4,
                              temperature=1,
                              output_dimension=32,
                              max_iterations=500,
                              distance='cosine',
                              conditional='time_delta',
                              device='cpu',
                              verbose=True,
                              time_offsets=10)
pos_decoder = cebra.KNNDecoder(n_neighbors=36, metric='cosine')

# Fit model
cebra_pos_model.fit(data_dict['spike_counts'], data_dict['relative_distance'])

# Decode position
embedding = cebra_pos_model.transform(data_dict['spike_counts'])
pos_decoder.fit(embedding, data_dict['relative_distance'])
pos_pred = pos_decoder.predict(embedding)

# Plot result
f, ax1 = plt.subplots()
ax1.plot(data_dict['relative_distance'])
ax1.plot(pos_pred)

