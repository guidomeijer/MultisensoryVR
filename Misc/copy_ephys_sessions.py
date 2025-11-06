import shutil
from pathlib import Path
import functools

# --- Configuration ---

# 1. The root folder to scan. Assumes the script is in the SAME directory
#    as the 'Subjects' folder.
SOURCE_ROOT = Path(r"D:\MultisensoryVR\Subjects") 

# 2. The destination drive or folder.
DESTINATION_ROOT = Path(r"G:\My Drive\Work\Data\Subjects")

# 3. List of exact subfolder names to copy.
ALLOWED_SUBFOLDERS = {"probe00", "probe01"}

# 4. File prefix to ignore
IGNORE_FILE_PREFIX = "_phy"

# ---------------------

def custom_copy_ignore(session_root_path, allowed_folders, ignore_prefix, src, names):
    """
    This function is called by shutil.copytree to decide what to ignore.
    
    - `session_root_path`: The top-level session (e.g., .../Session_A)
    - `allowed_folders`: The set {"probe00", "probe01"}
    - `ignore_prefix`: The string "_phy"
    - `src`: The directory *currently* being scanned (could be Session_A or Session_A/probe00)
    - `names`: A list of all files/folders in `src`.
    """
    src_path = Path(src)
    ignored_names = set()
    
    # ---
    # Rule 1: We are in the root session folder (e.g., .../Session_A)
    # ---
    if src_path == session_root_path:
        for name in names:
            item_path = src_path / name
            # Ignore files starting with the prefix
            if item_path.is_file() and name.startswith(ignore_prefix):
                ignored_names.add(name)
            # Ignore directories NOT in our allowed list
            elif item_path.is_dir() and name not in allowed_folders:
                ignored_names.add(name)
    
    # ---
    # Rule 2: We are *inside* one of the allowed folders (e.g., .../Session_A/probe00)
    # ---
    elif src_path.parent == session_root_path and src_path.name in allowed_folders:
        for name in names:
            item_path = src_path / name
            # Ignore files starting with the prefix
            if item_path.is_file() and name.startswith(ignore_prefix):
                ignored_names.add(name)
            # Ignore *any* sub-directory inside probe00/probe01
            elif item_path.is_dir():
                ignored_names.add(name)
            
    return ignored_names


def main():
    """
    Finds and copies session folders.
    - Only copies sessions that contain 'probe00' or 'probe01'.
    - In the session folder, only copies files (not starting with _phy) 
      and the 'probe00'/'probe01' dirs.
    - In the 'probe00'/'probe01' dirs, only copies files (not starting with _phy).
    """
    try:
        source_base = SOURCE_ROOT.resolve(strict=True)
    except FileNotFoundError:
        print(f"Error: Source directory '{SOURCE_ROOT}' not found.")
        return

    dest_base = DESTINATION_ROOT.resolve()
    print(f"Scanning: {source_base}")
    print(f"Destination: {dest_base}")
    
    session_count = 0

    # Iterate through Subject_Name folders
    for subject_path in SOURCE_ROOT.iterdir():
        if not subject_path.is_dir():
            continue 

        # Iterate through Session_Name folders
        for session_path in subject_path.iterdir():
            if not session_path.is_dir():
                continue
            
            # --- Check: Does 'probe00' or 'probe01' exist? ---
            found_matching_probe = False
            try:
                for folder_name in ALLOWED_SUBFOLDERS:
                    if (session_path / folder_name).is_dir():
                        found_matching_probe = True
                        break 
            
            except PermissionError:
                print(f"    *** Skipping {session_path} (PermissionError)")
                continue
            except FileNotFoundError:
                print(f"    *** Skipping {session_path} (File not found during scan)")
                continue

            # If we found at least one matching probe folder, copy
            if found_matching_probe:
                session_count += 1
                print(f"\n[+] Found matching subfolder in: {session_path}")
                
                relative_session_path = session_path.relative_to(SOURCE_ROOT)
                dest_path = DESTINATION_ROOT / relative_session_path
                
                print(f"    -> Copying to: {dest_path} (filtering files and folders)")
                
                # --- MODIFIED COPY ACTION ---
                # Pass the new IGNORE_FILE_PREFIX to the ignore function
                ignore_func = functools.partial(
                    custom_copy_ignore, 
                    session_path,       # arg 1: session_root_path
                    ALLOWED_SUBFOLDERS, # arg 2: allowed_folders
                    IGNORE_FILE_PREFIX  # arg 3: ignore_prefix
                )
                
                try:
                    shutil.copytree(
                        session_path, 
                        dest_path, 
                        dirs_exist_ok=True, 
                        ignore=ignore_func
                    )
                    print(f"    ... Success.")
                except Exception as e:
                    print(f"    *** ERROR copying {session_path}: {e}")

    print(f"\n--- Scan Complete ---")
    print(f"Found and copied {session_count} session folders.")

if __name__ == "__main__":
    if not DESTINATION_ROOT.exists():
        print(f"Destination folder '{DESTINATION_ROOT}' does not exist.")
        print("Creating it now...")
        try:
            DESTINATION_ROOT.mkdir(parents=True, exist_ok=True)
            print("... Destination created.")
            main()
        except Exception as e:
            print(f"*** ERROR: Could not create destination folder: {e}")
    else:
        main()