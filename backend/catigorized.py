import categorizedPickle as cpkl
import mediapipe as mp
import cv2
import os
import numpy as np
import pickle
import shutil
from tqdm import tqdm
import imghdr
def compute_similarity(landmarks1, landmarks2):
    landmarks1_np = np.array([(l['x'], l['y']) for l in landmarks1])
    landmarks2_np = np.array([(l['x'], l['y']) for l in landmarks2])
    distances = np.linalg.norm(landmarks1_np - landmarks2_np, axis=1)
    return np.mean(distances)

def distrubteDownload():
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(download_path):
        for filename in tqdm(filenames, desc="Reading Downlaod"):
            # Join the directory path and file name to get the full path
            full_path = os.path.join(dirpath, filename)
            if cpkl.is_image(full_path):
                addToDB(full_path)

def addToDB(full_path):
    normalized_img = cpkl.get_landmarks(full_path)
    if normalized_img == None:
        print("\nCould not find hand in image: " + full_path)
        return
    maxSimilar = -1
    maxSimilarName = None
    for bone in bones:
        nextSimilar =  compute_similarity(normalized_img, bone[1])
        if maxSimilar < nextSimilar:
            maxSimilar = nextSimilar
            maxSimilarName = bone[0]
    if maxSimilar < 0.6:
        shutil.copy2(full_path, newHandPath)
        return
    # add to bone[0] dir
    filename = os.path.basename(maxSimilarName)
    dir_name = os.path.splitext(filename)[0]
    target_dir = os.path.join(database_path, dir_name)
    shutil.copy2(full_path, target_dir)

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True)
    database_path = "backend/database"
    download_path = "backend/download"
    processed_images_pickle = "backend/bonesPickle.pickle"
    bonesPath = "backend/bones"
    newHandPath = "backend/newHand"
    if os.path.exists(processed_images_pickle):
        processed_images = cpkl.load_from_pickle(processed_images_pickle)
    else:
        processed_images = []
    bones = cpkl.boneStructure(bonesPath, processed_images)
    if not os.path.exists(database_path):
        os.makedirs(database_path)
    distrubteDownload()