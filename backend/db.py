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

def is_image(file_path):
    return imghdr.what(file_path) in ['rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff', 'rast', 'xbm', 'jpeg', 'bmp', 'png', 'webp', 'exr']

def load_from_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

class HandDatabase:
    def __init__(self, bones_path, database_path, pickle_dir, download_path, newHand_path, hands_processor):
        self.bones_path = bones_path
        self.database_path = database_path
        self.download_path = download_path
        self.hands_processor = hands_processor
        self.embedding_extractor = EmbeddingExtractor()
        os.makedirs(pickle_dir, exist_ok=True)
        self.pickle_path = os.path.join(pickle_dir, "/bonesPickle.pickle")

        # Ensure directories exist
        os.makedirs(self.bones_path, exist_ok=True)  
        os.makedirs(self.database_path, exist_ok=True)
        os.makedirs(self.download_path, exist_ok=True)
        
        if os.path.exists(self.pickle_path):
            self.processed_images = load_from_pickle(self.pickle_path)
        else:
            self.processed_images = []

    def _save_embedding(self, image_path):
        """Utility function to save embedding."""
        embedding = self.embedding_extractor.get_embedding(image_path)
        timestamp = datetime.now()
        self.processed_images.append((image_path, embedding, timestamp))

    def bone_structure(self):
        for filename in tqdm(os.listdir(self.bones_path), desc="Processing Bones", unit="file"):
            image_path = os.path.join(self.bones_path, filename)
            if is_image(image_path):
                self._save_embedding(image_path)

    def _find_most_similar(self, embedding):
        max_similarity = float('-inf')
        closest_image = None
        for bone in self.processed_images:
            bone_embedding = bone[1]
            similarity = self.embedding_extractor.cosine_similarity(embedding, bone_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                closest_image = bone[0]
        return closest_image, max_similarity

    def generate_database(self):
        for data in tqdm(self.processed_images, desc="Creating database", unit="bonetuple"):
            img_path, _, _, _ = data
            filename = os.path.basename(img_path)
            target_dir = os.path.join(self.database_path, os.path.splitext(filename)[0])
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(img_path, target_dir)
    
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
            shutil.copy2(full_path, "backend/newHand")
        else:
            dir_name = os.path.splitext(os.path.basename(closest_image))[0]
            target_dir = os.path.join(self.database_path, dir_name)
            shutil.copy2(full_path, target_dir)


    def collect_hand_gesture_data(self, capture_delay=1):
        # Determine the starting class name based on existing filenames.
        existing_files = os.listdir(self.bones_path)
        start_class = len(existing_files) + 1 if existing_files else 0

        number_of_classes = int(input("Enter the number of new classes to collect: "))
        number_of_classes += start_class

        while start_class < number_of_classes:
            accepted = False  # Flag to determine if the captured image is accepted or not
            cap = cv2.VideoCapture(0)
            while not accepted:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                if not ret:
                    print("Failed to capture frame!")
                    continue

                if frame is None:
                    print("Frame is None!")
                    continue

                if frame.size == 0:
                    print("Frame has zero size!")
                    continue
                cv2.putText(frame, 'Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                
                if cv2.waitKey(25) == ord('q'):
                    print("Detected 'Q' key press.")
                    print(f'Collecting data for class: {start_class}')
                    print(f'Position your hand and press "Q" to start collecting 1 samples...')
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

                    # Check if hand landmarks can be found
                    results = self.hands_processor.process(frame_rgb)
                    if not results.multi_hand_landmarks:
                        print("No hand landmarks detected.")
                        cv2.putText(frame, 'Couldn\'t find hands, try again', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow('frame', frame)
                        cv2.waitKey(1000)  # Pause for a bit before trying again
                        continue

                    # If hand landmarks are found, check its similarity using embeddings
                    saved_image_path = self._frame_to_image_path(frame_rgb)
                    new_embedding = self.embedding_extractor.get_embedding(saved_image_path)
                    current_similarity = self._find_most_similar(new_embedding)[1]
                    print(f"Calculated similarity: {current_similarity}")

                    # If the similarity is below a certain threshold, accept the image
                    if current_similarity < 0.90:
                        accepted = True  # Update the flag
                        start_class += 1
                        self._save_embedding(saved_image_path)
                        cv2.putText(frame, f'Accepted: {start_class}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow('frame', frame)
                        cv2.waitKey(1000)
                    else:
                        cv2.putText(frame, 'Too similar, try again', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow('frame', frame)
                        cv2.waitKey(1000)  # Pause for a bit before trying again

            cap.release()
            cv2.destroyAllWindows()
        rename_files(self.bones_path)
        save_to_pickle(self.processed_images, self.pickle_path)
        

    
    def _frame_to_image_path(self, frame, filename_prefix="frame_"):
        """ Saves a frame in the 'bones' directory and returns the image path. """
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        saved_image_path = os.path.join(self.bones_path, f"{filename_prefix}{timestamp}.png")
        cv2.imwrite(saved_image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return saved_image_path

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands_processor = mp_hands.Hands(static_image_mode=True)

    database = HandDatabase(
        bones_path="backend/bones",
        database_path="backend/database",
        pickle_dir="backend/pickles",
        download_path="backend/download",
        newHand_path="backend/newHand",
        hands_processor=hands_processor)
    database.collect_hand_gesture_data()
    # database.bone_structure()
    # database.generate_database()
    # database.distribute_downloads()