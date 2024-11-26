import math

import cv2
import mediapipe as mp
import numpy as np
import Settings


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


# noinspection PyGlobalUndefined
def phaseCapturingFinger(frm):
    cv2.circle(frm, (frm.shape[1] // 2, frm.shape[0] // 2), 13, (44, 62, 80), -1)
    if Settings.only_index_finger:
        cv2.putText(frm, "Captured your finger!", (160, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0))
        if Settings.START_PHASE_TWO == 25:
            Settings.CURRENT_PHASE = phaseDrawingCircle
        else:
            Settings.START_PHASE_TWO += 1
    else:
        Settings.only_index_finger = check_only_finger(flippedRGB)
        cv2.putText(flippedRGB, "Can't see your index finger...", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))


# noinspection PyGlobalUndefined
def phaseDrawingCircle(frm):
    hands = handsDetector.process(frm)

    cv2.circle(frm, (frm.shape[1] // 2, frm.shape[0] // 2), 13, (44, 62, 80), -1)
    if not Settings.DRAWING:
        cv2.putText(flippedRGB, "Place your finger to the point from which you start!", (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
        cv2.putText(frm, f"Start drawing in {Settings.SECONDS_UNTIL_DRAWING} seconds", (140, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0))

        if Settings.REMOVE_SECOND_UNTIL_DRAWING == 30:
            Settings.SECONDS_UNTIL_DRAWING -= 1
            Settings.REMOVE_SECOND_UNTIL_DRAWING = 0
        else:
            Settings.REMOVE_SECOND_UNTIL_DRAWING += 1

        if Settings.SECONDS_UNTIL_DRAWING == 0:
            x_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].x * frm.shape[0])
            y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * frm.shape[0])
            Settings.circle_radius = math.hypot(abs(frm.shape[1] // 2 - x_index_tip), abs(frm.shape[0] // 2 - y_index_tip))

            Settings.index_frame_circles.append((x_index_tip, y_index_tip))
            Settings.DRAWING = True
    else:
        cv2.putText(frm, f"Draw the circle around the dot", (105, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0))

        for circle_point in Settings.index_frame_circles:
            ...


Settings.CURRENT_PHASE = phaseCapturingFinger
with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as handsDetector:
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

        flippedRGB = cv2.cvtColor(np.fliplr(frame), cv2.COLOR_BGR2RGB)

        Settings.CURRENT_PHASE(flippedRGB)

        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hands", res_image)

    handsDetector.close()
