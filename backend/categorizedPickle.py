import mediapipe as mp
import cv2
import os
import numpy as np
import pickle
import shutil
from tqdm import tqdm
import imghdr

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

def get_landmarks(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        return normalize_landmarks(results.multi_hand_landmarks[0].landmark)
    return None
    
def boneStructure(bonesPath, processed_images):
    landmarks_tuple = []
    new_images = False
    
    for filename in tqdm(os.listdir(bonesPath), desc="Creating Bones Tuple", unit="bone"):
        image_path = os.path.join(bonesPath, filename)
        if is_image(image_path):  # Add other image extensions if necessary
            
            # Check if the image has already been processed
            if image_path not in processed_images:
                # Process the image
                landmarks = get_landmarks(image_path)
                if landmarks is not None:
                    # Append the landmarks to the list
                    landmarks_tuple.append((image_path, landmarks))
                    # Mark the image as processed
                    processed_images.append(image_path)
                    new_images = True

    # If there are new images, update the bones pickle file
    if new_images:
        save_to_pickle(landmarks_tuple, processed_images_pickle)
        
    return landmarks_tuple

def generateDatabase(landmarks_tuple):
    if not os.path.exists(database_path):
        os.makedirs(database_path)
    # Process each tuple and create directories, then copy the image into that directory
    for data in tqdm(landmarks_tuple, desc="Creating database", unit="bonetuple"):
        img_path = data[0]
        filename = os.path.basename(img_path)  # e.g., "image1.jpg"
        dir_name = os.path.splitext(filename)[0]  # Remove the extension, e.g., "image1"
        
        # Create the directory under database directory
        target_dir = os.path.join(database_path, dir_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Copy the image into this directory
        shutil.copy2(img_path, target_dir)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True)
bonesPath = "backend/bones"
database_path = "backend/database"
processed_images_pickle = "backend/bonesPickle.pickle"
if os.path.exists(processed_images_pickle):
    processed_images = load_from_pickle(processed_images_pickle)
else:
    processed_images = []
bones = boneStructure(bonesPath, processed_images)
generateDatabase(bones)
