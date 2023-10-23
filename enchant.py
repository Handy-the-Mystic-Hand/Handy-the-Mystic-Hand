import mediapipe as mp
import cv2
import os
import numpy as np
import pickle
import shutil
from tqdm import tqdm

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

def compute_similarity(landmarks1, landmarks2):
    landmarks1_np = np.array([(l['x'], l['y']) for l in landmarks1])
    landmarks2_np = np.array([(l['x'], l['y']) for l in landmarks2])
    distances = np.linalg.norm(landmarks1_np - landmarks2_np, axis=1)
    return np.mean(distances)
    
def process_new_images(dataset_path, landmarks_data):
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png')]
    
    # Only process images that are not in landmarks_data
    new_image_files = [image_path for image_path in image_files if image_path not in landmarks_data]
    
    for image_path in new_image_files:
        landmarks = get_landmarks(image_path)
        if landmarks:
            landmarks_data[image_path] = landmarks

    return landmarks_data

def display(results):
    for hand_number, hand_landmarks in enumerate(results.multi_hand_landmarks):
        original_landmarks = [{'x': l.x, 'y': l.y} for l in hand_landmarks.landmark]
        normalized = normalize_landmarks(hand_landmarks.landmark)

        # Check Mean and Scale
        normalized_np = np.array([(l['x'], l['y']) for l in normalized])
        print("Mean of normalized landmarks:", np.mean(normalized_np, axis=0))
        max_distance = np.max(np.linalg.norm(normalized_np, axis=1))
        print(f"Max distance from center: {max_distance:.4f}")
        
        # Print each landmark's coordinates
        for id, landmark in enumerate(hand_landmarks.landmark):
            print(f"Landmark {id}: x={landmark.x}, y={landmark.y}, z={landmark.z}")
        
        # Draw landmarks on the image
        landmarks_np = np.array([(l.x * img_rgb.shape[1], l.y * img_rgb.shape[0]) for l in hand_landmarks.landmark])
        mean = np.mean(landmarks_np, axis=0).astype(int)
        # cv2.circle(img_rgb, tuple(mean), 5, (0, 255, 0), -1)  # Draws a green circle at the mean

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(
            img_rgb,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    # Display the result with landmarks
    cv2.imshow('Hand landmarks', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    PICKLE_FILENAME = "normalized_landmarks_data.pkl"
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True)
    ranking = 100
    img_path = "data/1/0.jpg"
    given_image = get_landmarks(img_path)
    if not given_image:
        print("No hands detected in the image.")
        exit()
    dataset_path = "database"  # Replace with your dataset directory path
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png')]
    
    # Check if pickle file with normalized landmarks data exists
    added = 0
    if os.path.exists(PICKLE_FILENAME):
        landmarks_data = load_from_pickle(PICKLE_FILENAME)
        
        # Process any new images and update landmarks data
        landmarks_data = process_new_images(dataset_path, landmarks_data)
        added += 1
    else:
        landmarks_data = {}
        # Added tqdm around the loop to display a progress bar
        for image_path in tqdm(image_files, desc="Processing Images"):
            landmarks = get_landmarks(image_path)
            if landmarks:
                landmarks_data[image_path] = landmarks

    # Save normalized landmarks data to the initial pickle file
    save_to_pickle(landmarks_data, PICKLE_FILENAME)
    print(f"Added {added} new images to the dataset.")
    similarities = []
    for image_path in tqdm(landmarks_data.keys(), desc="Computing Similarities"):
        similarity = compute_similarity(given_image, landmarks_data[image_path])
        similarities.append((image_path, similarity))

    top_100 = sorted(similarities, key=lambda x: x[1])[:100]
    
    simi100_dir = "simi100"
    if not os.path.exists(simi100_dir):
        os.makedirs(simi100_dir)

    for image, similarity in top_100:
        if similarity == -1:
            break
        shutil.copy(image, simi100_dir)
        print(f"Image: {image}, Similarity: {similarity}")
