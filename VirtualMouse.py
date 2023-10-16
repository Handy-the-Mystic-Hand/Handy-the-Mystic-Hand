import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui

hand_detector = HandDetector()
capture = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    hands, img = hand_detector.findHands(frame)
    for hand in hands:
        if hand["type"] == "Left":
            landmarks = hand["lmList"]
            index_tip_x = int(landmarks[8][0])
            index_tip_y = int(landmarks[8][1])
            pyautogui.moveTo(index_tip_x*screen_width/frame_width, index_tip_y*screen_height/frame_height)
            middle_tip_x = int(landmarks[12][0])
            middle_tip_y = int(landmarks[12][1])
            if abs(index_tip_x - middle_tip_x) < 20 and abs(index_tip_y - middle_tip_y) < 20:
                pyautogui.click()
                pyautogui.sleep(1)

    cv2.imshow('current', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


capture.release()

cv2.destroyAllWindows()


