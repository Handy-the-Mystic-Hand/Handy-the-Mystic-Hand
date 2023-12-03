import os
import shutil
import cv2
import mediapipe as mp
import pickle
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

def find_hand_from_image(image_path):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    frame = cv2.imread(image_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                            frame,  # image to draw
                            hand_landmarks,  # model output
                            mp_hands.HAND_CONNECTIONS,  # hand connections
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow('Hand Detection', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No hand detected")
    # Close the hand detection object
    hands.close()

def read_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
        # Assuming the data is a list or other iterable
        for item in data:
            print(item)

if __name__ == "__main__":
    # SOURCE_DIR = "leapGestRecog"
    # DEST_DIR = "database"
    # consolidate_images(SOURCE_DIR, DEST_DIR)
    # directory_path = '/Users/haikeyu/Desktop/bones'
    # rename_files(directory_path)
    # find_hand_from_image("/Users/haikeyu/Desktop/CSC490/Handy-the-Mystic-Hand/backend/bones/2.png")
    read_pickle("/Users/haikeyu/Desktop/CSC490/Handy-the-Mystic-Hand/backend/pickles/bonesPickle.pickle")