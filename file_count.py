import os

path = "Data_sets/"

for folder_name in os.listdir(path):
    folder_path = os.path.join(path, folder_name)

    if os.path.isdir(folder_path):
        file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

        print(f"{folder_name}: {file_count} files")