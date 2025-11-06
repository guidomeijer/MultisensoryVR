# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 10:40:19 2025

By Guido Meijer
"""

import os
import shutil
import subprocess

# --- CONFIGURATION ---
# IMPORTANT: Set these paths before running the script.
# Use raw strings (r'...') on Windows or escape backslashes ('\\').
source_dir = r"V:\imaging1\guido\Subjects"
dest_dir = r"F:\Subjects"
FLAG_FILENAME = "ephys_session.flag"
VIDEO_FOLDER_NAME = "raw_behavior_data"

"""
Moves video data from the source to the destination, preserving the
Subject/Session structure, unless a specific flag file is present.
"""
print(f"Starting script. Source: '{source_dir}', Destination: '{dest_dir}'")

# 1. Iterate through each subject folder
for subject_name in os.listdir(source_dir):
    subject_path = os.path.join(source_dir, subject_name)
    
    if not os.path.isdir(subject_path):
        continue # Skip files, only process directories

    # 2. Iterate through each session folder for the subject
    for session_name in os.listdir(subject_path):
        session_path = os.path.join(subject_path, session_name)
        
        if not os.path.isdir(session_path):
            continue

        # 3. Check for the exclusion flag file in the session folder
        flag_path = os.path.join(session_path, FLAG_FILENAME)
        if os.path.exists(flag_path):
            print(f"-> Skipping session '{session_name}' for subject '{subject_name}': Found '{FLAG_FILENAME}'.")
            continue

        # 4. Locate the source video folder
        video_source_path = os.path.join(session_path, VIDEO_FOLDER_NAME)
        if not os.path.isdir(video_source_path):
            # This session doesn't have the target video folder, so we skip it.
            continue

        # 5. Prepare the destination directory structure
        video_dest_path = os.path.join(dest_dir, subject_name, session_name, VIDEO_FOLDER_NAME)
        try:
            os.makedirs(video_dest_path, exist_ok=True)
        except OSError as e:
            print(f"ERROR: Could not create destination directory '{video_dest_path}'. Reason: {e}")
            continue # Skip this session if we can't create the destination

        # 6. Iterate through and move each file in the video folder
        if not os.listdir(video_source_path):
            print(f"-> Source folder '{video_source_path}' is empty. Nothing to move.")
            continue

        print(f"Processing: {video_source_path}")
        for filename in os.listdir(video_source_path):
            full_source_path = os.path.join(video_source_path, filename)

            # Move the file
            try:
                shutil.move(full_source_path, video_dest_path)
                print(f"   -> Moved: {filename}")
            except Exception as e:
                print(f"   [X] ERROR: Failed to move '{filename}'. Reason: {e}")
