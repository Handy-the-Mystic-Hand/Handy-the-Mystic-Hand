import os
import pickle
from math import sqrt
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create a Hands object with specific parameters
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'  # Directory containing the hand images

data = []  # List to store the processed data
labels = []  # List to store the labels (categories) of the hand poses
ignore = [0, 1, 5, 9, 13, 17]  # Landmarks to ignore (like wrist points)

# Iterate over each directory (representing a label) in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Iterate over each image file in the directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store data for each image

        # Read the image and convert it to RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to find hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_ = []
            y_ = []
            # Extract X and Y coordinates of the landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Compute bounding box coordinates for the hand
            x1 = int(min(x_) * 640) - 10
            y1 = int(min(y_) * 480) - 10
            x2 = int(max(x_) * 640) + 10
            y2 = int(max(y_) * 480) + 10
            dist_weight = max(abs(x2 - x1), abs(y2 - y1))

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            # Calculate the distance of each landmark from the wrist, normalized by dist_weight
            for i in range(len(hand_landmarks.landmark)):
                if i not in ignore:
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    dist = sqrt((x - wrist_x)**2 + (y - wrist_y)**2) * 640 / dist_weight
                    data_aux.append(dist)
            data.append(data_aux)
            labels.append(int(dir_))  # The directory name is used as the label

# Save the processed data and labels to a pickle file for future use
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
