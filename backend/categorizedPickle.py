import mediapipe as mp
import cv2
import os
import numpy as np
import pickle
import shutil
from tqdm import tqdm
import imghdr
from datetime import datetime

def is_image(file_path):
    #Return True if the file is an image, otherwise return False.
    image_types = ['rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff', 'rast', 'xbm', 'jpeg', 'bmp', 'png', 'webp', 'exr']
    return imghdr.what(file_path) in image_types

def load_from_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Function to save data to a pickle file
def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def normalize_landmarks(landmarks):
    # Convert landmarks to numpy array
    landmarks_np = np.array([(l.x, l.y) for l in landmarks])

    # Compute mean
    mean = np.mean(landmarks_np, axis=0)

    # Center landmarks around the mean
    centered = landmarks_np - mean

    # Scale landmarks
    max_distance = np.max(np.linalg.norm(centered, axis=1))
    if max_distance == 0:
        max_distance = 1  # Prevent division by zero
    
    normalized = centered / max_distance

    # Convert back to the original format (list of dicts)
    normalized_landmarks = [{'x': x, 'y': y} for x, y in normalized]
    
    return normalized_landmarks

def get_landmarks(image_path, hands_processor):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_processor.process(img_rgb)
    if results.multi_hand_landmarks:
        return normalize_landmarks(results.multi_hand_landmarks[0].landmark)
    return None

class Category:
    def __init__(self, dir_path, boneImg, pickle_path):
        self.pickle_path = pickle_path
        self.dir_path = dir_path
        self.boneImg = boneImg


class HandDatabase:
    def __init__(self, bones_path, database_path, pickle_path, hands_processor):
        self.bones_path = bones_path
        self.database_path = database_path
        self.hands_processor = hands_processor
        self.catrgory_list = []
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        self.pickle_path = os.path.join(pickle_path, "./bonesPickle.pickle")

        if os.path.exists(self.pickle_path):
            self.processed_images = self.load_from_pickle(pickle_path)
        else:
            self.processed_images = []
            
    def boneStructure(self):
        new_images = False
        for filename in tqdm(os.listdir(self.bones_path), desc="Creating Bones Tuple", unit="bone"):
            image_path = os.path.join(self.bones_path, filename)
            if is_image(image_path):  # Add other image extensions if necessary
                # Check if the image has already been processed
                landmarks = get_landmarks(image_path, self.hands_processor)
                if not landmarks:
                    continue
                timestamp = datetime.now()
                for index, image in enumerate(self.processed_images):
                    if image[0] == image_path:
                        if self.image_was_modified(image[0], image[2]):
                            # Image was modified since the last time
                            # Update the tuple in processed_images with new landmarks and timestamp
                            updated_image_data = (image_path, landmarks, timestamp, image[3])  # Assuming you want to keep the old order
                            self.processed_images[index] = updated_image_data
                            # Set a flag if you want to track modifications
                            new_images = True
                        else:
                            continue
                else:
                    self.processed_images.append((image_path, landmarks, timestamp))
                    new_images = True
        # If there are new images, update the bones pickle file
        if new_images:
            for index, image in enumerate(self.processed_images):
                updated_image_data = (image[0], image[1], image[2], index)  # Add order
                self.processed_images[index] = updated_image_data
            save_to_pickle(self.processed_images, self.pickle_path)
        
    def image_was_modified(self, image_path, processed_time):
        """Check if the image at image_path was modified since last processed."""
        current_timestamp = os.path.getmtime(image_path)
        mod_time_datetime = datetime.fromtimestamp(current_timestamp)
        return processed_time < mod_time_datetime

    def update_processed_images(self, image_path, processed_images):
        """Update the timestamp for the image at image_path in processed_images."""
        processed_images[image_path] = os.path.getmtime(image_path)

    def generateDatabase(self):
        if not os.path.exists(self.database_path):
            os.makedirs(database_path)
        # Process each tuple and create directories, then copy the image into that directory
        for data in tqdm(self.processed_images, desc="Creating database", unit="bonetuple"):
            img_path = data[0]
            filename = os.path.basename(img_path)  # e.g., "image1.jpg"
            dir_name = os.path.splitext(filename)[0]  # Remove the extension, e.g., "image1"
            # Create the directory under database directory
            target_dir = os.path.join(self.database_path, dir_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            category_path = self.pickle_path + "/" + str(data[3])
            shutil.copy2(img_path, target_dir)
            category = Category(target_dir, data, category_path)
            self.catrgory_list.append(category)
    
if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands_processor = mp_hands.Hands(static_image_mode=True)
    database_path = "backend/database"
    download_path = "backend/download"
    pickle_path = "backend/pickles"
    bones_path = "backend/bones"
    newHandPath = "backend/newHand"
    database = HandDatabase(bones_path, database_path, pickle_path, hands_processor)
    database.boneStructure()
    database.generateDatabase()
