import pickle

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = dict([(i, str(i)) for i in range(model.n_classes_)])


screen_width, screen_height = pyautogui.size()
in_action = False
mouse_on = False

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
        for hand_landmarks in results.multi_hand_landmarks:
            if first_hand:
                first_hand = False
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

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

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])[0]

                if not in_action and prediction == 2:
                    in_action = True
                    mouse_on = False
                if in_action and prediction == 1:
                    mouse_on = True
                if mouse_on:
                    index_x = hand_landmarks.landmark[8].x * screen_width
                    index_y = hand_landmarks.landmark[8].y * screen_height
                    pyautogui.moveTo(index_x, index_y)
                if in_action and prediction == 0:
                    pyautogui.click()
                    in_action = False
                if in_action and prediction == 6:
                    alt_tab()
                    in_action = False
                if prediction == 3:
                    pyautogui.press('esc')
                    in_action = False
                if in_action and prediction == 5:
                    swipe_right()
                    in_action = False
                if in_action and prediction == 4:
                    swipe_left()
                    in_action = False


                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, str(prediction), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

