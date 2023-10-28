import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import shutil
from tqdm import tqdm
import imghdr
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class EmbeddingExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device)
        # Remove the final classification layer to get embeddings
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image_path):
        # Load the image in RGB format
        image = Image.open(image_path).convert("RGB")
        
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image)
        return embedding.squeeze().cpu().numpy()

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)

# --- Utility Functions ---
def is_image(file_path):
    return imghdr.what(file_path) in ['rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff', 'rast', 'xbm', 'jpeg', 'bmp', 'png', 'webp', 'exr']

def load_from_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def get_landmarks(image_path, hands_processor):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Forces the image to be read in RGB format.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_processor.process(img_rgb)
    if results.multi_hand_landmarks:
        return normalize_landmarks(results.multi_hand_landmarks[0].landmark)
    return None

# --- Similarity and Normalization Functions ---
# def compute_similarity(landmarks1, landmarks2):
#     landmarks1_np = np.array([(l['x'], l['y']) for l in landmarks1])
#     landmarks2_np = np.array([(l['x'], l['y']) for l in landmarks2])
#     # Compute mean of each set of landmarks
#     mean1 = np.mean(landmarks1_np, axis=0)
#     mean2 = np.mean(landmarks2_np, axis=0)
    
#     # Use the distance between the mean and the wrist as the normalization factor
#     normalization_factor1 = np.linalg.norm(mean1 - landmarks1_np[0])
#     normalization_factor2 = np.linalg.norm(mean2 - landmarks2_np[0])

#     landmarks1_np /= normalization_factor1
#     landmarks2_np /= normalization_factor2
#     distances = np.linalg.norm(landmarks1_np - landmarks2_np, axis=1)
#     return np.mean(distances)

def normalize_landmarks(landmarks):
    landmarks_np = np.array([(l.x, l.y) for l in landmarks])
    centered = landmarks_np - np.mean(landmarks_np, axis=0)
    normalized = centered / np.max(np.linalg.norm(centered, axis=1) + 1e-8)
    return [{'x': x, 'y': y} for x, y in normalized]

# --- Main HandDatabase Class ---
class HandDatabase:
    def __init__(self, bones_path, database_path, pickle_path, download_path, hands_processor):
        self.bones_path = bones_path
        self.database_path = database_path
        self.download_path = download_path
        self.hands_processor = hands_processor
        self.embedding_extractor = EmbeddingExtractor()
        self.pickle_path = os.path.join(pickle_path, "bonesPickle.pickle")
        if os.path.exists(self.pickle_path):
            self.processed_images = load_from_pickle(self.pickle_path)
        else:
            os.makedirs(pickle_path, exist_ok=True)
            self.processed_images = []

    def bone_structure(self):
        new_images = False
        for filename in tqdm(os.listdir(self.bones_path), desc="Creating Bones Tuple", unit="bone"):
            image_path = os.path.join(self.bones_path, filename)
            if is_image(image_path):
                landmarks = get_landmarks(image_path, self.hands_processor)
                if not landmarks:
                    continue
                self._update_or_append_image(image_path, landmarks)
                new_images = True

        if new_images:
            for index, image in enumerate(self.processed_images):
                self.processed_images[index] = (image[0], image[1], image[2], index)
            save_to_pickle(self.processed_images, self.pickle_path)

    def _update_or_append_image(self, image_path, landmarks):
        timestamp = datetime.now()
        for index, image in enumerate(self.processed_images):
            if image[0] == image_path:
                if self._image_was_modified(image[0], image[2]):
                    self.processed_images[index] = (image_path, landmarks, timestamp, image[3])
                return
        self.processed_images.append((image_path, landmarks, timestamp))

    def _image_was_modified(self, image_path, processed_time):
        current_timestamp = os.path.getmtime(image_path)
        return processed_time.timestamp() < current_timestamp

    def distribute_downloads(self):
        for dirpath, _, filenames in tqdm(os.walk(self.download_path), desc="Processing directories", unit="dir"):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                if is_image(full_path):
                    self._add_to_db(full_path)

    def _add_to_db(self, full_path):
        embedding_img = self.embedding_extractor.get_embedding(full_path)
        closest_image, similarity = self._find_most_similar(embedding_img)
        print(similarity)
        if similarity < 0.70:  # Adjust threshold if needed
            os.makedirs("backend/newHand", exist_ok=True)
            shutil.copy2(full_path, "backend/newHand")
        else:
            dir_name = os.path.splitext(os.path.basename(closest_image))[0]
            target_dir = os.path.join(self.database_path, dir_name)
            shutil.copy2(full_path, target_dir)

    def _find_most_similar(self, embedding):
        max_similarity = float('-inf')
        closest_image = None
        for bone in self.processed_images:
            bone_embedding = self.embedding_extractor.get_embedding(bone[0])
            similarity = self.embedding_extractor.cosine_similarity(embedding, bone_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                closest_image = bone[0]
        return closest_image, max_similarity

    def generate_database(self):
        os.makedirs(self.database_path, exist_ok=True)
        for data in tqdm(self.processed_images, desc="Creating database", unit="bonetuple"):
            img_path, _, _, _ = data
            filename = os.path.basename(img_path)
            target_dir = os.path.join(self.database_path, os.path.splitext(filename)[0])
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(img_path, target_dir)

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands_processor = mp_hands.Hands(static_image_mode=True)

    database = HandDatabase(
        bones_path="backend/bones",
        database_path="backend/database",
        pickle_path="backend/pickles",
        download_path="backend/download",
        hands_processor=hands_processor)
    database.bone_structure()
    database.generate_database()
    database.distribute_downloads()
