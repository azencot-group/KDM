import os
import shutil


def delete_unwanted_subdirs(target_folder, allowed_subdirs):
    # Get a list of all subdirectories in the target folder
    all_subdirs = [d for d in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, d))]

    # Loop through all subdirectories and delete those not in allowed_subdirs list
    for subdir in all_subdirs:
        if subdir not in allowed_subdirs:
            subdir_path = os.path.join(target_folder, subdir)
            shutil.rmtree(subdir_path)
            print(f"Deleted: {subdir_path}")


# Example usage
target_folder = '/path/to/target/folder'  # Replace with the path to your folder
allowed_subdirs = ['keep_this_dir', 'and_this_one']  # Replace with the list of directories to keep

delete_unwanted_subdirs(target_folder, allowed_subdirs)