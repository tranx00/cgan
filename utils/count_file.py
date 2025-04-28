import os

def count_files_in_folder(folder_path, include_subfolders=True):
    total_files = 0
    if include_subfolders:
        for root, _, files in os.walk(folder_path):
            total_files += len(files)
    else:
        total_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    return total_files


folder_path = "./dataset/prepos_dataset/val/images"
file_count = count_files_in_folder(folder_path, False)
print(f"Total number of files: {file_count}")