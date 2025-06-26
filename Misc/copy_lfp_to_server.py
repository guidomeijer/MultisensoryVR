# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:43:48 2025

By Gemini
"""

import shutil
from pathlib import Path
import os

# --- Configuration ---
# IMPORTANT: Replace these placeholder paths with the actual paths on your system.

# 1. Set the path to the source "Subjects" folder on your computer.
source_subjects_dir = Path('D:\MultisensoryVR\Subjects')

# 2. Set the path to the "Subjects" folder on your external hard drive.
dest_subjects_dir = Path("V:\imaging1\guido\Subjects")

# 3. Set to False to perform the actual copy operation.
#    Keeping it True will only print what the script intends to do.
DRY_RUN = False

# --- End of Configuration ---


def copy_lfp_data(source_base: Path, dest_base: Path, dry_run: bool = True):
    """
    Finds and copies 'lfp_raw_binary' folders from a source to a destination directory.

    Args:
        source_base (Path): The root directory of the source 'Subjects' folder.
        dest_base (Path): The root directory of the destination 'Subjects' folder.
        dry_run (bool): If True, prints actions without copying files.
                        If False, performs the copy.
    """
    # Check if the source directory exists
    if not source_base.is_dir():
        print(f"Error: Source directory not found at '{source_base}'")
        return

    # Create the base destination directory if it doesn't exist
    if not dest_base.is_dir():
        print(f"Destination directory '{dest_base}' not found.")
        if not dry_run:
            print("Creating it...")
            dest_base.mkdir(parents=True, exist_ok=True)
        else:
            print("Would create it in a real run.")


    print(f"Scanning for 'lfp_raw_binary' folders in: {source_base}")
    print("-" * 30)

    # Use glob to find all folders named "lfp_raw_binary" inside a "probe01" folder.
    # The pattern '*/*/' corresponds to the <subject> and <session> folders.
    glob_pattern = "*/*/probe01/lfp_raw_binary"
    found_count = 0
    copied_count = 0

    for source_path in source_base.glob(glob_pattern):
        if source_path.is_dir():
            found_count += 1
            # Determine the relative path from the source 'Subjects' folder
            relative_path = source_path.relative_to(source_base)

            # Construct the full destination path
            dest_path = dest_base / relative_path

            print(f"Found: {source_path}")
            print(f"  --> Destination: {dest_path}\n")

            if dest_path.exists():
                print("  ! Already exists at destination. Skipping.\n")
                continue

            if not dry_run:
                try:
                    # Create the parent directories in the destination if they don't exist
                    # e.g., .../Subjects/SubjectX/SessionY/probe01/
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy the entire directory tree
                    shutil.copytree(source_path, dest_path)
                    print(f"  ✓ Successfully copied.\n")
                    copied_count += 1
                except Exception as e:
                    print(f"  ✗ ERROR copying {source_path}: {e}\n")

    print("-" * 30)
    print("Scan complete.")
    print(f"Total folders found: {found_count}")
    if dry_run:
        print(f"This was a DRY RUN. No files were copied.")
        print("To copy files, set DRY_RUN = False at the top of the script.")
    else:
        print(f"Total new folders copied: {copied_count}")


if __name__ == "__main__":
    copy_lfp_data(source_subjects_dir, dest_subjects_dir, dry_run=DRY_RUN)