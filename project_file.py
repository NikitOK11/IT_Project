import cv2
import mediapipe as mp
import numpy as np
from Settings import *


def check_only_finger(image):
    hands = handsDetector.process(image)

    if hands.multi_hand_landmarks is not None:
        y_thumb_tip = int(hands.multi_hand_landmarks[0].landmark[4].y * image.shape[0])
        y_thumb_mcp = int(hands.multi_hand_landmarks[0].landmark[2].y * image.shape[0])
        x_thumb_tip = int(hands.multi_hand_landmarks[0].landmark[4].x * image.shape[1])
        x_thumb_mcp = int(hands.multi_hand_landmarks[0].landmark[2].x * image.shape[1])

        y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * image.shape[0])
        y_index_mcp = int(hands.multi_hand_landmarks[0].landmark[5].y * image.shape[0])

        y_middle_finger_tip = int(hands.multi_hand_landmarks[0].landmark[12].y * image.shape[0])
        y_middle_finger_mcp = int(hands.multi_hand_landmarks[0].landmark[9].y * image.shape[0])

        y_ring_finger_tip = int(hands.multi_hand_landmarks[0].landmark[16].y * image.shape[0])
        y_ring_finger_mcp = int(hands.multi_hand_landmarks[0].landmark[13].y * image.shape[0])

        y_pinky_tip = int(hands.multi_hand_landmarks[0].landmark[20].y * image.shape[0])
        y_pinky_mcp = int(hands.multi_hand_landmarks[0].landmark[17].y * image.shape[0])

        if abs(y_thumb_tip - y_thumb_mcp) <= 110 and abs(y_middle_finger_tip - y_middle_finger_mcp) <= 110 and abs(
                y_ring_finger_tip - y_ring_finger_mcp) <= 110 and abs(
                y_pinky_tip - y_pinky_mcp) <= 110 and y_index_tip <= y_index_mcp and abs(
                x_thumb_tip - x_thumb_mcp) <= 85:
            return True
        return False
    return False


with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as handsDetector:
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

        flipped = np.fliplr(frame)
        flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        if PHASE == 1:
            if only_index_finger:
                cv2.putText(flippedRGB, "Captured your finger!", (160, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0))
                if START_PHASE_TWO == 25:
                    PHASE = 2
                else:
                    START_PHASE_TWO += 1
            else:
                only_index_finger = check_only_finger(flippedRGB)
                cv2.putText(flippedRGB, "You should be drawing with only one finger!", (35, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))

        elif PHASE == 2:
            cv2.circle(flippedRGB, (flippedRGB.shape[1] // 2, flippedRGB.shape[0] // 2), 10, (84, 59, 59), -1)
            if not DRAWING:
                cv2.putText(flippedRGB, f"Start drawing in {SECONDS_UNTIL_DRAWING} seconds", (140, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0))
                if REMOVE_SECOND_UNTIL_DRAWING == 30:
                    SECONDS_UNTIL_DRAWING -= 1
                    REMOVE_SECOND_UNTIL_DRAWING = 0
                else:
                    REMOVE_SECOND_UNTIL_DRAWING += 1

                if SECONDS_UNTIL_DRAWING == 0:
                    DRAWING = True

        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hands", res_image)

    handsDetector.close()
