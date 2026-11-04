# -*- coding: utf-8 -*-
"""
Copy the files in each session folder on the server to the hard disk when they
don't exist there yet. Folders inside the session folders are ignored.

Created on Wed Nov 04 2026

By Guido Meijer
"""

import os
from os import path
import shutil

# Paths: both should contain subject folders, which contain session folders
server_path = 'V:\\imaging1\\guido\\Subjects'
hd_path = 'D:\\MultisensoryVR\\Subjects'

# Loop over subjects on server
for subject in os.listdir(server_path):
    if not path.isdir(path.join(server_path, subject)):
        continue
    print(f'Starting subject {subject}')

    # Loop over sessions
    for session in os.listdir(path.join(server_path, subject)):
        server_ses_path = path.join(server_path, subject, session)
        hd_ses_path = path.join(hd_path, subject, session)
        if not path.isdir(server_ses_path):
            continue

        # Loop over files, ignore folders and .flag files
        server_files = [f for f in os.listdir(server_ses_path)
                        if path.isfile(path.join(server_ses_path, f))
                        and not f.endswith('.flag')]
        for this_file in server_files:
            if not path.isfile(path.join(hd_ses_path, this_file)):
                print(f'Copying {subject}/{session}/{this_file}')
                os.makedirs(hd_ses_path, exist_ok=True)
                shutil.copy2(path.join(server_ses_path, this_file),
                             path.join(hd_ses_path, this_file))
