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


if __name__ == "__main__":
    SOURCE_DIR = "leapGestRecog"
    DEST_DIR = "database"
    consolidate_images(SOURCE_DIR, DEST_DIR)
