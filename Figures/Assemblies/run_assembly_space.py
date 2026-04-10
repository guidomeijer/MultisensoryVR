# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 10/03/2026
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style, load_objects
path_dict = paths()
colors, dpi = figure_style()

# Region order
REGIONS = ['CA1', 'LEC', 'PERI', 'TEa', 'AUD', 'VIS']

# Get data files
assembly_df = pd.read_csv(path_dict['save_path'] / 'assembly_sig.csv')
assembly_df['subject'] = assembly_df['subject'].astype(str)
assembly_df['date'] = assembly_df['date'].astype(str)
assembly_df['probe'] = assembly_df['probe'].astype(str)
activation_files = (path_dict['google_drive_data_path'] / 'Assemblies').rglob('*amplitudes.npy')
time_files = (path_dict['google_drive_data_path'] / 'Assemblies').rglob('*times.npy')
time_ax = np.load(list(time_files)[0])

# Load in data per region
assembly_act = {'CA1': [], 'LEC': [], 'AUD': [], 'VIS': [], 'TEa': [], 'PERI': []}
for i, file in enumerate(activation_files):

    # Get info
    subject = file.stem.split('.')[0].split('_')[0]
    date = file.stem.split('.')[0].split('_')[1]
    probe = file.stem.split('.')[0].split('_')[2]
    region = file.stem.split('.')[0].split('_')[3]

    # Load in data
    this_act = np.load(file)
    all_obj_df = load_objects(subject, date)




    asd

