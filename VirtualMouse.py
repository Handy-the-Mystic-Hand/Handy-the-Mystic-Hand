import cv2
import mediapipe
import pyautogui

hand_captured = mediapipe.solutions.hands.Hands()
drawing = mediapipe.solutions.drawing_utils
capture = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output_hand = hand_captured.process(rgb_frame)

    hands = output_hand.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            tip_x = int(landmarks[8].x*frame_width)
            tip_y = int(landmarks[8].y*frame_height)
            cv2.circle(frame, (tip_x, tip_y), 10, (0, 0, 0, 128))
            pyautogui.moveTo(tip_x*screen_width/frame_width, tip_y*screen_height/frame_height)
            middle_x = int(landmarks[12].x*frame_width)
            middle_y = int(landmarks[12].y*frame_height)
            cv2.circle(frame, (middle_x, middle_y), 10, (0, 0, 0, 128))
            if abs(tip_x - middle_x) < 20 and abs(tip_y - middle_y) < 20:
                pyautogui.click()
                pyautogui.sleep(1)
    cv2.imshow('current', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


capture.release()

cv2.destroyAllWindows()


