import os
import pickle
from math import sqrt

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


def clip_hand(hand_landmarks, W, H):
    x_ = []
    y_ = []
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y

        x_.append(x)
        y_.append(y)

    x1 = int(min(x_) * W) - 20
    y1 = int(min(y_) * H) - 20

    x2 = int(max(x_) * W) + 20
    y2 = int(max(y_) * H) + 20

    if (x2 - x1) > (y2 - y1):
        y2 = y1 + (x2 - x1)
    else:
        x2 = x1 + (y2 - y1)

    result = [x1, x2, y1, y2]
    return result

def alt_tab():
    # Hold down the 'alt' key
    pyautogui.keyDown('alt')

    # Press the 'tab' key while 'alt' is still being held down
    pyautogui.press('tab')

    # Release the 'alt' key
    pyautogui.keyUp('alt')

def swipe_left():
    pyautogui.keyDown('win')
    pyautogui.press('left')
    pyautogui.keyUp('win')

def swipe_right():
    pyautogui.keyDown('win')
    pyautogui.press('right')
    pyautogui.keyUp('win')

def ctrl_c():
    pyautogui.keyDown('ctrl')
    pyautogui.press('c')
    pyautogui.keyUp('ctrl')

def lock():
    pyautogui.keyDown('win')
    pyautogui.press('l')
    pyautogui.keyUp('win')

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


screen_width, screen_height = pyautogui.size()
in_action = False
mouse_on = False
ignore = [0, 1, 5, 9, 13, 17]

def clip_hand(hand_landmarks, W, H):
    x_ = []
    y_ = []
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y

        x_.append(x)
        y_.append(y)

    x1 = int(min(x_) * W) - 20
    y1 = int(min(y_) * H) - 20

    x2 = int(max(x_) * W) + 20
    y2 = int(max(y_) * H) + 20

    if (x2 - x1) > (y2 - y1):
        y2 = y1 + (x2 - x1)
    else:
        x2 = x1 + (y2 - y1)

    result = [x1, x2, y1, y2]
    return result

i = 0
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    first_hand = True
    if results.multi_hand_landmarks:
        # if len(results.multi_hand_landmarks)==1:
            hand_landmark = results.multi_hand_landmarks[0]
            for i in range(len(hand_landmark.landmark)):
                x = hand_landmark.landmark[i].x
                y = hand_landmark.landmark[i].y

                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            dist_weight = max(abs(x2 - x1), abs(y2 - y1))

            wrist_x = hand_landmark.landmark[0].x
            wrist_y = hand_landmark.landmark[0].y
            for i in range(len(hand_landmark.landmark)):
                if not i in ignore:
                    x = hand_landmark.landmark[i].x
                    y = hand_landmark.landmark[i].y
                    dist = sqrt((x - wrist_x) ** 2 + (y - wrist_y) ** 2) * W / dist_weight
                    data_aux.append(dist)

            prediction = model.predict([np.asarray(data_aux)])[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, str(prediction), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (0, 0, 0), 3, cv2.LINE_AA)
            if not in_action and prediction == 2:
                in_action = True
                mouse_on = False
            if in_action and prediction == 1:
                mouse_on = True
            if mouse_on:
                index_x = hand_landmark.landmark[8].x * screen_width
                index_y = hand_landmark.landmark[8].y * screen_height
                pyautogui.moveTo(index_x, index_y)
            if (in_action or  mouse_on) and prediction == 0:
                pyautogui.click()
                in_action = False
            if in_action and prediction == 3:
                alt_tab()
                in_action = False
            # if prediction == 7:
            #     pyautogui.press('esc')
            #     in_action = False
            if in_action and prediction == 5:
                swipe_right()
                in_action = False
            if in_action and prediction == 6:
                swipe_left()
                in_action = False
            if in_action and prediction == 4:
                ctrl_c()
                in_action = False
            if in_action and prediction == 8:
                lock()
                in_action = False
            if in_action and prediction == 9:
                print('9')
                in_action = False
            if in_action and prediction == 10:
                print('10')
                in_action = False
            if in_action and prediction == 11:
                print('11')
                in_action = False
        # else:
        #     hand_landmark = results.multi_hand_landmarks[0]
        #     for i in range(len(hand_landmark.landmark)):
        #         x = hand_landmark.landmark[i].x
        #         y = hand_landmark.landmark[i].y
        #
        #         x_.append(x)
        #         y_.append(y)
        #
        #     x1 = int(min(x_) * W) - 10
        #     y1 = int(min(y_) * H) - 10
        #
        #     x2 = int(max(x_) * W) + 10
        #     y2 = int(max(y_) * H) + 10
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        #     cv2.putText(frame, "right", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
        #                 (0, 0, 0), 3, cv2.LINE_AA)



    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

