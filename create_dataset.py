import os
import pickle
from math import sqrt

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
ignore = [0, 1, 5, 9, 13, 17]
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_ = []
            y_ = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * 640) - 10
            y1 = int(min(y_) * 480) - 10

            x2 = int(max(x_) * 640) + 10
            y2 = int(max(y_) * 480) + 10
            dist_weight = max(abs(x2 - x1), abs(y2 - y1))
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            for i in range(len(hand_landmarks.landmark)):
                if not i in ignore:
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    dist = sqrt((x - wrist_x)**2 + (y - wrist_y)**2)*640/dist_weight
                    data_aux.append(dist)
            data.append(data_aux)
            labels.append(int(dir_))

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()