import os
import shutil
def consolidate_images(source_dir, dest_dir, extensions=('.png')):
    total = 0
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for subdir, _, files in os.walk(source_dir, topdown=False):
        for file in files:
            if file.lower().endswith(extensions):
                total += 1
                source_path = os.path.join(subdir, file)
                dest_path = os.path.join(dest_dir, file)

                # Handle duplicate filenames
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(file)
                    dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.copy2(source_path, dest_path)
    print(f"Total files copied: {total}")

def rename_files(directory):
    # Get all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith('.jpeg')]

    
    # Sort the files to ensure consistent ordering
    files.sort()

    # Rename files
    for i, filename in enumerate(files):
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, f"{i:02}.jpg")  # formatted to have at least two digits
        os.rename(old_path, new_path)

    print(f"Renamed {len(files)} files in {directory}.")

# Specify the directory containing the files

if __name__ == "__main__":
    SOURCE_DIR = "leapGestRecog"
    DEST_DIR = "database"
    # consolidate_images(SOURCE_DIR, DEST_DIR)
    directory_path = '/Users/haikeyu/Desktop/bones'
    rename_files(directory_path)
