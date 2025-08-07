# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 09:46:12 2025 by Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style
colors, dpi = figure_style()

path_dict = paths()

error_df = pd.read_csv(path_dict['save_path'] / 'position_decoding_error.csv')
error_df['error_cm'] = error_df['error_mm'] / 10

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.boxplot(data=error_df, x='region', y='error_cm', ax=ax1)

