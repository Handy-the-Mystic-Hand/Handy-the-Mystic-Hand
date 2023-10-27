import os
import pickle
import random

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
num_label = 0
true_label = []
false_label = []
for i in range(100):
    true_label.append(1)
    false_label.append(0)
for dir_ in os.listdir(DATA_DIR):
    num_label += 1
    sub_data = []
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

            sub_data.append(data_aux)
    data.append(sub_data)

for i in range(num_label):
    all_datas = []
    all_labels = []
    pickle_name = "data" + str(i) + ".pickle"
    f = open(pickle_name, 'wb')
    all_datas.extend(data[i])
    all_labels.extend(true_label)
    other_data = []
    for j in range(num_label):
        if i != j:
            other_data.extend(data[j])
    false_data = random.sample(other_data, 100)
    all_datas.extend(false_data)
    all_labels.extend(false_label)
    pickle.dump({'data': all_datas, 'labels': all_labels}, f)
    f.close()
