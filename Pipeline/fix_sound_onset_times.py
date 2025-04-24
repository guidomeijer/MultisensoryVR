# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:03:17 2025

By Guido Meijer
"""

import os
import pandas as pd

root_folder = 'V:\\imaging1\\guido\\Subjects'

for subject in os.listdir(root_folder):
    subject_path = os.path.join(root_folder, subject)
    if os.path.isdir(subject_path):
        for session in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session)
            trials_csv_path = os.path.join(session_path, "trials.csv")
            if os.path.isfile(trials_csv_path):
                df = pd.read_csv(trials_csv_path)
                ads