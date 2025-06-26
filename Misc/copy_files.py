# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:19:08 2025

By Gemini 2.5 Pro
"""

import os
import shutil

def selective_copy_sessions(source_dir, dest_dir):
    """
    Copies session folders from a source directory to a destination if they
    contain 'probe00' or 'probe01' folders. It maintains the subject-session
    directory structure.

    - Copies files directly within the session folder.
    - Copies files within 'probe00'/'probe01', skipping subdirectories and any
      file starting with '_phy_spikes'.
    - Checks if a file already exists in the destination before copying.

    Args:
        source_dir (str): The absolute path to the main 'Subjects' folder.
        dest_dir (str): The absolute path to the destination directory.
    """
    print(f"Starting updated selective copy from '{source_dir}' to '{dest_dir}'...")

    # --- 1. Basic Path Validation ---
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return
    if not os.path.isdir(dest_dir):
        print(f"Error: Destination directory not found at '{dest_dir}'. Please ensure it exists.")
        return

    copied_session_count = 0
    # --- 2. Iterate Through Subjects ---
    for subject_name in os.listdir(source_dir):
        subject_path = os.path.join(source_dir, subject_name)
        if not os.path.isdir(subject_path):
            continue

        # --- 3. Iterate Through Sessions ---
        for session_name in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session_name)
            if not os.path.isdir(session_path):
                continue

            # --- 4. Check for 'probe00' or 'probe01' Folders (Condition to copy) ---
            probes_found_in_session = []
            for probe_folder_name in ["probe00", "probe01"]:
                if os.path.isdir(os.path.join(session_path, probe_folder_name)):
                    probes_found_in_session.append(probe_folder_name)
            
            # If the condition (a probe folder exists) is not met, skip this session
            if not probes_found_in_session:
                continue

            print(f"\nProcessing: {subject_name}/{session_name}")
            copied_session_count += 1
            
            # --- 5. Create Destination Session Folder ---
            dest_session_path = os.path.join(dest_dir, subject_name, session_name)
            os.makedirs(dest_session_path, exist_ok=True)
            
            # --- 6. (NEW) Copy Session-Level Files ---
            print("  -> Checking for session-level files...")
            for item in os.listdir(session_path):
                source_item_path = os.path.join(session_path, item)
                if os.path.isfile(source_item_path):
                    dest_item_path = os.path.join(dest_session_path, item)
                    
                    # (NEW) Check for existence before copying
                    if not os.path.exists(dest_item_path):
                        print(f"    - Copying file: '{item}'")
                        shutil.copy2(source_item_path, dest_item_path)
                    else:
                        print(f"    - Skipping existing file: '{item}'")

            # --- 7. Handle File Copying for Each Found Probe ---
            for probe_folder_name in probes_found_in_session:
                source_probe_path = os.path.join(session_path, probe_folder_name)
                dest_probe_path = os.path.join(dest_session_path, probe_folder_name)
                os.makedirs(dest_probe_path, exist_ok=True)
                
                print(f"  -> Processing contents of '{probe_folder_name}'...")

                for item in os.listdir(source_probe_path):
                    source_item_path = os.path.join(source_probe_path, item)
                    dest_item_path = os.path.join(dest_probe_path, item)
                    
                    # Rule 1: Must be a file
                    if not os.path.isfile(source_item_path):
                        print(f"    - Skipping folder: '{item}'")
                        continue
                        
                    # Rule 2: Filename must NOT start with '_phy_spikes'
                    if item.startswith("_phy_spikes"):
                        print(f"    - Skipping excluded file: '{item}'")
                        continue
                        
                    # (NEW) Rule 3: Copy only if it doesn't already exist
                    if not os.path.exists(dest_item_path):
                        print(f"    - Copying file: '{item}'")
                        shutil.copy2(source_item_path, dest_item_path)
                    else:
                        print(f"    - Skipping existing file: '{item}'")

    print(f"\n--------------------------------------------------")
    print(f"Script finished. Processed {copied_session_count} session(s) meeting the criteria.")
    print(f"Data is located in: {dest_dir}")
    print(f"--------------------------------------------------")


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # IMPORTANT: You MUST change these two paths.
    
    # --- Set the path to your source 'Subjects' folder ---
    # Example for Windows: SOURCE_SUBJECTS_FOLDER = r"C:\data\project_x\Subjects"
    # Example for macOS/Linux: SOURCE_SUBJECTS_FOLDER = "/home/user/data/project_x/Subjects"
    SOURCE_SUBJECTS_FOLDER = r'D:\MultisensoryVR\Subjects'

    # --- Set the path to your destination external hard drive ---
    # Example for Windows: DESTINATION_DRIVE = r"E:\backup"
    # Example for macOS/Linux: DESTINATION_DRIVE = "/Volumes/MyExternalDrive/backup"
    DESTINATION_DRIVE = r'F:\Data\MultisensoryVR\Subjects'
    # =======================================================


    # --- Run the function ---
    selective_copy_sessions(SOURCE_SUBJECTS_FOLDER, DESTINATION_DRIVE)