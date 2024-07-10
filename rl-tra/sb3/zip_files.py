import os
import zipfile
                        
def zip_files(dir_path: str, file_prefix: str, zip_file_name: str):
    with zipfile.ZipFile(f'{dir_path}{zip_file_name}.zip', 'w') as zipf:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.startswith(file_prefix) and file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    print(f'Adding {os.path.relpath(full_path, dir_path)} ({full_path}) to {dir}{zip_file_name}.zip')
                    #if os.path.getsize(full_path) >= 1024 * 1024:
                    zipf.write(full_path,
                        arcname=os.path.relpath(full_path, dir_path))
                    # Delete the file after adding it to the ZIP
                    os.remove(full_path)
