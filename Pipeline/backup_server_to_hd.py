# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:14:18 2025

By Guido Meijer
"""

import os
from os import path
import shutil
from msvr_functions import paths

# Get paths
path_dict = paths()
hd_path = 'F:\\Guido'

# Loop over subjects on server
for subject in os.listdir(path.join(path_dict['server_path'], 'Subjects')):
    
    print(f'Starting subject {subject}')
    
    # Loop over sessions
    for session in os.listdir(path.join(path_dict['server_path'], 'Subjects', subject)):
        
        server_path = path.join(path_dict['server_path'], 'Subjects', subject, session)
        local_path = path.join(hd_path, 'Subjects', subject, session)
        if path.isfile(path.join(server_path, 'ephys_session.flag')):
            
            print(f'Backing up session {session}')        
            if not path.isdir(path.join(hd_path, 'Subjects', subject)):
                os.mkdir(path.join(hd_path, 'Subjects', subject))
            if not path.isdir(local_path):
                os.mkdir(local_path)
            
            # Loop over files
            server_files = [f for f in os.listdir(server_path) if path.isfile(path.join(server_path, f))]
            for this_file in server_files:
                if not path.isfile(path.join(local_path, this_file)):
                    shutil.copy(path.join(server_path, this_file), path.join(local_path, this_file))
            
            # Loop over folders
            server_folders = [f for f in os.listdir(server_path) if path.isdir(path.join(server_path, f))]
            for this_dir in server_folders:
                if not path.isdir(path.join(local_path, this_dir)):
                    shutil.copytree(path.join(server_path, this_dir), path.join(local_path, this_dir),
                                    dirs_exist_ok=True)
            
