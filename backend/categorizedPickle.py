import os
import cv2
import mediapipe as mp
import numpy as np
import dill
import shutil
from tqdm import tqdm
import imghdr
from datetime import datetime
import pickle
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

class HandDatabase:
    def __init__(self, config):
        self.bones_path = config['bones_path']
        self.database_path = config['database_path']
        self.download_path = config['download_path']
        self.newHand_path = config['newHand_path']
        self.pickle_dir = config['pickle_dir']
        self.hands_processor = hands_processor
        self.embedding_extractor = EmbeddingExtractor()
        self._ensure_directories_exist()
        self._initialize_bones()
        self.all_landmarks = []

    def _save_embedding(self, image_path):
        """Utility function to save embedding."""
        embedding = self.embedding_extractor.get_embedding(image_path)
        timestamp = datetime.now()
        self.processed_images.append((image_path, embedding, timestamp))

    def _ensure_directories_exist(self):
        """Ensure that required directories exist."""
        directories = [self.bones_path, self.database_path, self.download_path, self.newHand_path, self.pickle_dir]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _initialize_bones(self):
        self.pickle_path = os.path.join(self.pickle_dir, "bonesPickle.pickle")
        if os.path.exists(self.pickle_path):
            self.processed_images = load_from_pickle(self.pickle_path)
            self.last_image_id = len(self.processed_images)
        else:
            self.processed_images = []
            self.last_image_id = 0
    
    def _capture_frame(self, cap):
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if not ret or frame is None or frame.size == 0:
                self._handle_frame_error(frame)
                return None

            cv2.putText(frame, 'Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(25) == ord('q'):
                return frame
    
    def _process_and_check_landmarks(self, frame_rgb):
        results = self.hands_processor.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return False
        return results

    def _crop_hand_from_frame(self, frame_rgb, landmarks, pixel_margin=100):
        x_coordinates = [landmark.x for landmark in landmarks]
        y_coordinates = [landmark.y for landmark in landmarks]
        xmin, xmax = min(x_coordinates), max(x_coordinates)
        ymin, ymax = min(y_coordinates), max(y_coordinates)
        height, width, _ = frame_rgb.shape

        # Add fixed pixel margin to the bounding box
        xmin_pixel = int(xmin * width) - pixel_margin
        xmax_pixel = int(xmax * width) + pixel_margin
        ymin_pixel = int(ymin * height) - pixel_margin
        ymax_pixel = int(ymax * height) + pixel_margin

        # Ensure values are within frame boundaries
        xmin_pixel = max(0, xmin_pixel)
        ymin_pixel = max(0, ymin_pixel)
        xmax_pixel = min(width, xmax_pixel)
        ymax_pixel = min(height, ymax_pixel)

        return frame_rgb[ymin_pixel:ymax_pixel, xmin_pixel:xmax_pixel]

    def _frame_to_image_path(self, frame, filename_prefix="frame_"):
        """ Saves a frame in the 'bones' directory and returns the image path. """
        saved_image_path = os.path.join(self.bones_path, f"{self.last_image_id}.png")
        cv2.imwrite(saved_image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return saved_image_path

    def _process_and_check_similarity(self, frame_rgb):
        saved_image_path = self._frame_to_image_path(frame_rgb)
        new_embedding = self.embedding_extractor.get_embedding(saved_image_path)
        current_similarity = self._find_most_similar(new_embedding)[1]
        print(f"Calculated similarity: {current_similarity}")
        return saved_image_path, current_similarity
    
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
    
    def _display_message(self, frame, message, position, color):
        cv2.putText(frame, message, position, cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)
    
    def _add_to_db(self, full_path):
        frame_rgb = cv2.imread(full_path, cv2.IMREAD_COLOR)
        detected = self._process_and_check_landmarks(frame_rgb)
        if detected:
            crop_frame_rgb = self._crop_hand_from_frame(frame_rgb, detected.multi_hand_landmarks[0].landmark)
            crop_detected = self._process_and_check_landmarks(crop_frame_rgb)
            if crop_detected:
                crop_resize_rgb = cv2.resize(crop_frame_rgb, (256, 256))
                tmp_path = os.path.join("./backend", "temp_hand.png")
                cv2.imwrite(tmp_path, crop_resize_rgb)
                new_embedding = self.embedding_extractor.get_embedding(tmp_path)
                closest_image, current_similarity = self._find_most_similar(new_embedding)
                if current_similarity < 0.70:  # Adjust threshold if needed
                    shutil.copy2(full_path, self.newHand_path)
                else:
                    dir_name = os.path.splitext(os.path.basename(closest_image))[0]
                    target_dir = os.path.join(self.database_path, dir_name)
                    shutil.copy2(full_path, target_dir)
                    return (target_dir, crop_detected.multi_hand_landmarks[0])
        return None

    def collect_hand_gesture_data(self, capture_delay=1):
        start_class = len(self.processed_images)
        number_of_classes = int(input("Enter the number of new classes to collect: ")) + start_class
        cap = cv2.VideoCapture(0)
        while start_class < number_of_classes:
            frame = self._capture_frame(cap)
            if frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected = self._process_and_check_landmarks(frame_rgb)
            if detected:
                crop_frame_rgb = self._crop_hand_from_frame(frame_rgb, detected.multi_hand_landmarks[0].landmark)
                crop_detected = self._process_and_check_landmarks(crop_frame_rgb)
                if crop_detected:
                    crop_resize_rgb = cv2.resize(crop_frame_rgb, (256, 256))
                    copy = crop_resize_rgb.copy()
                    display_media(copy, crop_detected)
                    cv2.imshow('frame', copy)
                    cv2.waitKey(1000)
                    saved_image_path = self._frame_to_image_path(crop_resize_rgb)
                    new_embedding = self.embedding_extractor.get_embedding(saved_image_path)
                    current_similarity = self._find_most_similar(new_embedding)[1]

                    if current_similarity < 0.90:
                        start_class += 1
                        self.last_image_id += 1
                        self._save_embedding(saved_image_path)
                        self._display_message(frame, f'Accepted: {start_class}', (100, 100), (0, 255, 0))
                    else:
                        os.remove(saved_image_path)
                        self._display_message(frame, 'Too similar, try again', (100, 100), (0, 255, 0))
                else:
                    self._display_message(frame, 'Couldn\'t find hands, try again', (100, 100), (0, 255, 0))
            else:
                self._display_message(frame, 'Couldn\'t find hands, try again', (100, 100), (0, 255, 0))
        cap.release()
        cv2.destroyAllWindows()
        save_to_pickle(self.processed_images, self.pickle_path)

    def generate_database(self):
        for data in tqdm(self.processed_images, desc="Creating database", unit="bonetuple"):
            img_path, _, _,= data
            filename = os.path.basename(img_path)
            target_dir = os.path.join(self.database_path, os.path.splitext(filename)[0])
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(img_path, target_dir)
            pickle_path = os.path.join(target_dir, f"{os.path.splitext(filename)[0]}.pickle")

    def distribute_downloads(self):
        # First, gather all the image file paths.
        image_paths = []
        for dirpath, _, filenames in os.walk(self.download_path):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                if is_image(full_path):
                    image_paths.append(full_path)

        # Dictionary to store landmarks per directory
        dir_landmarks = {}
        # Process each image and categorize landmarks by directory
        for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
            result = self._add_to_db(image_path)

            if result:
                target_dir, landmarks = result
                if target_dir not in dir_landmarks:
                    dir_landmarks[target_dir] = []
                dir_landmarks[target_dir].append(landmarks)

        for dir_path, landmarks_list in tqdm(dir_landmarks.items(), desc="Processing directories", unit="directory"):
            data = []      
            for img_path in os.listdir(os.path.join(dir_path)):
            # Create a filename for the pickle file based on the directory name
                pickle_filename = os.path.basename(dir_path) + "_landmarks.pickle"
                pickle_path = os.path.join(self.pickle_dir, pickle_filename)
                data_aux = []
                x_ = []
                y_ = []
                # print(landmarks_list)
                for hand_landmarks in landmarks_list:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                data.append(data_aux)
            save_to_pickle(data, pickle_path)

            
def display_media(frame, hands):
    for hand_landmarks in hands.multi_hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_drawing.draw_landmarks(
        frame,  # image to draw
        hand_landmarks,  # model output
        mp_hands.HAND_CONNECTIONS,  # hand connections
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

def load_from_pickle(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def is_image(file_path):
    return imghdr.what(file_path) in ['rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff', 'rast', 'xbm', 'jpeg', 'bmp', 'png', 'webp', 'exr']

def delete_directories(*dirs):
    """
    Deletes the specified directories.

    Args:
    - *dirs: Directories to be deleted.
    """
    for directory in dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Deleted {directory}")
        else:
            print(f"{directory} does not exist.")


if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands_processor = mp_hands.Hands(static_image_mode=True)

    config = {
        "bones_path": "backend/bones",
        "database_path": "backend/database",
        "pickle_dir": "backend/pickles",
        "download_path": "backend/download",
        "newHand_path": "backend/newHand",
        "hands_processor": hands_processor
    }
    delete_directories(config["bones_path"], config["database_path"], config["pickle_dir"], config["newHand_path"])
    database = HandDatabase(config)
    database.collect_hand_gesture_data()
    database.generate_database()
    database.distribute_downloads()