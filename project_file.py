import cv2
import mediapipe as mp
import numpy as np


with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as handsDetector:
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

        only_index_finger = False

        flipped = np.fliplr(frame)
        flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        hands = handsDetector.process(flippedRGB)
        if hands.multi_hand_landmarks is not None:
            x_thumb_tip = int(hands.multi_hand_landmarks[0].landmark[4].y * flipped.shape[0])
            x_thumb_mcp = int(hands.multi_hand_landmarks[0].landmark[2].y * flipped.shape[0])

            x_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * flipped.shape[0])
            x_index_mcp = int(hands.multi_hand_landmarks[0].landmark[5].y * flipped.shape[0])

            x_middle_finger_tip = int(hands.multi_hand_landmarks[0].landmark[12].y * flipped.shape[0])
            x_middle_finger_mcp = int(hands.multi_hand_landmarks[0].landmark[9].y * flipped.shape[0])

            x_ring_finger_tip = int(hands.multi_hand_landmarks[0].landmark[16].y * flipped.shape[0])
            x_ring_finger_mcp = int(hands.multi_hand_landmarks[0].landmark[13].y * flipped.shape[0])

            x_pinky_tip = int(hands.multi_hand_landmarks[0].landmark[20].y * flipped.shape[0])
            x_pinky_mcp = int(hands.multi_hand_landmarks[0].landmark[17].y * flipped.shape[0])

            print(f"Thumb tip: {x_thumb_tip}, thumb mcp: {x_thumb_mcp}")
            print(f"index tip: {x_index_tip}, index mcp: {x_index_mcp}")
            print(f"middle tip: {x_middle_finger_tip}, middle mcp: {x_middle_finger_mcp}")
            print(f"ring tip: {x_ring_finger_tip}, ring mcp: {x_ring_finger_mcp}")
            print(f"pinky tip: {x_index_tip}, pinky mcp: {x_index_mcp}")
            print("")

            if abs(x_thumb_tip - x_thumb_mcp) <= 110 and abs(x_middle_finger_tip - x_middle_finger_mcp) <= 110 and abs(x_ring_finger_tip - x_ring_finger_mcp) <= 110 and abs(x_pinky_tip - x_pinky_mcp) <= 110 and x_index_tip <= x_index_mcp:
                only_index_finger = True

            if only_index_finger:
                cv2.circle(flippedRGB, (flipped.shape[0] // 2, flipped.shape[1] // 2), 10, (0, 255, 0), -1)
            else:
                cv2.circle(flippedRGB, (flipped.shape[0] // 2, flipped.shape[1] // 2), 10, (255, 0, 0), -1)

        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hands", res_image)

    handsDetector.close()
