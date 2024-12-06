import math
import cv2
import mediapipe as mp
import numpy as np
import Settings
import os


def checkOnlyFinger(frame) -> bool:
    hands = handsDetector.process(frame)

    if hands.multi_hand_landmarks is not None:
        y_thumb_tip = int(hands.multi_hand_landmarks[0].landmark[4].y * frame.shape[0])
        y_thumb_mcp = int(hands.multi_hand_landmarks[0].landmark[2].y * frame.shape[0])
        x_thumb_tip = int(hands.multi_hand_landmarks[0].landmark[4].x * frame.shape[1])
        x_thumb_mcp = int(hands.multi_hand_landmarks[0].landmark[2].x * frame.shape[1])

        y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * frame.shape[0])
        y_index_mcp = int(hands.multi_hand_landmarks[0].landmark[5].y * frame.shape[0])

        y_middle_finger_tip = int(hands.multi_hand_landmarks[0].landmark[12].y * frame.shape[0])
        y_middle_finger_mcp = int(hands.multi_hand_landmarks[0].landmark[9].y * frame.shape[0])

        y_ring_finger_tip = int(hands.multi_hand_landmarks[0].landmark[16].y * frame.shape[0])
        y_ring_finger_mcp = int(hands.multi_hand_landmarks[0].landmark[13].y * frame.shape[0])

        y_pinky_tip = int(hands.multi_hand_landmarks[0].landmark[20].y * frame.shape[0])
        y_pinky_mcp = int(hands.multi_hand_landmarks[0].landmark[17].y * frame.shape[0])

        if abs(y_thumb_tip - y_thumb_mcp) <= 110 and abs(y_middle_finger_tip - y_middle_finger_mcp) <= 110 and abs(
                y_ring_finger_tip - y_ring_finger_mcp) <= 110 and abs(
                y_pinky_tip - y_pinky_mcp) <= 110 and y_index_tip <= y_index_mcp and abs(
                x_thumb_tip - x_thumb_mcp) <= 85:
            return True
        return False
    return False


def checkHands(frame):
    hands = handsDetector.process(frame)
    if hands.multi_hand_landmarks is None:
        return False
    return True


def isCircleClosed(circles, radius, threshold=0.5, min_close_points=5):
    if len(circles) < 10:
        return False

    start_point = circles[0]
    close_to_start_count = 0

    for point in circles:
        distance = math.hypot(start_point[0] - point[0], start_point[1] - point[1])
        if distance < radius * threshold:
            close_to_start_count += 1

    return close_to_start_count >= min_close_points


def loadBestScore():
    if os.path.exists(Settings.RESULTS_FILE):
        with open(Settings.RESULTS_FILE, "r") as file:
            try:
                return float(file.read().strip())
            except ValueError:
                return 0.0
    return 0.0


def savBestScore(score):
    with open(Settings.RESULTS_FILE, "w") as file:
        file.write(f"{score:.1f}")


def resetSettings() -> None:
    Settings.DRAWING = False
    Settings.only_index_finger = False

    Settings.index_frame_circles = []
    Settings.circle_radius = 0

    Settings.SECONDS_UNTIL_DRAWING = 5
    Settings.REMOVE_SECOND_UNTIL_DRAWING = 20
    Settings.START_PHASE_TWO = 0

    Settings.ACCURACY_DRAWING = 1

    Settings.CURRENT_PHASE = checkOnlyFinger


def drawCircleOnFrame(frame) -> None:
    for i in range(len(Settings.index_frame_circles)):
        circle_point = Settings.index_frame_circles[i]
        x_tip, y_tip = circle_point[0], circle_point[1]

        dist_from_center = int(math.hypot(abs(frame.shape[1] // 2 - x_tip), abs(frame.shape[0] // 2 - y_tip)))
        mistake = abs(Settings.circle_radius - dist_from_center) // 4
        color_mistake = Settings.colors_error_from_radius[int(min(5, mistake))]
        if i > 0:
            cv2.line(frame, (x_tip, y_tip),
                     (Settings.index_frame_circles[i - 1][0],
                      Settings.index_frame_circles[i - 1][1]), color_mistake, 4)


def phaseNoHandsFound(frame) -> None:
    cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (44, 62, 80), -1)
    cv2.putText(frame, "Please, make sure your hands are fully in the frame", (25, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0))

    if len(Settings.index_frame_circles) > 0:
        drawCircleOnFrame(frame)


# noinspection PyGlobalUndefined
def phaseCapturingFinger(frame) -> None:
    cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (44, 62, 80), -1)
    if Settings.only_index_finger:
        cv2.putText(frame, "Captured your finger!", (120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0))
        if Settings.START_PHASE_TWO == 25:
            Settings.CURRENT_PHASE = phaseDrawingCircle
        else:
            Settings.START_PHASE_TWO += 1
    else:
        Settings.only_index_finger = checkOnlyFinger(flippedRGB)
        cv2.putText(flippedRGB, "Can't see your index finger...", (85, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))


# noinspection PyGlobalUndefined,PyTypeChecker
def phaseDrawingCircle(frame) -> None:
    hands = handsDetector.process(frame)

    cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (44, 62, 80), -1)
    if not Settings.DRAWING:
        cv2.putText(flippedRGB, "Place your finger to the point from which you start!", (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
        cv2.putText(frame, f"Start drawing in {Settings.SECONDS_UNTIL_DRAWING} seconds", (140, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0))

        if Settings.REMOVE_SECOND_UNTIL_DRAWING == 20:
            Settings.SECONDS_UNTIL_DRAWING -= 1
            Settings.REMOVE_SECOND_UNTIL_DRAWING = 0
        else:
            Settings.REMOVE_SECOND_UNTIL_DRAWING += 1

        if Settings.SECONDS_UNTIL_DRAWING == 0:
            if hands.multi_hand_landmarks is not None:
                x_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].x * frame.shape[1])
                y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * frame.shape[0])
                Settings.circle_radius = int(math.hypot(abs(frame.shape[1] // 2 - x_index_tip), abs(frame.shape[0] // 2 - y_index_tip)))

                Settings.index_frame_circles.append((x_index_tip, y_index_tip))
            Settings.DRAWING = True
    else:
        x_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].x * frame.shape[1])
        y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * frame.shape[0])
        if math.hypot(x_index_tip, y_index_tip, frame.shape[0] // 2, frame.shape[1] // 2) < 300:
            cv2.putText(frame, f"Your finger is too close to center!", (105, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0))
        else:
            cv2.putText(frame, f"Draw the circle around the dot", (130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0))

            accuracy_sum = 0
            for i in range(len(Settings.index_frame_circles)):
                circle_point = Settings.index_frame_circles[i]
                x_tip, y_tip = circle_point[0], circle_point[1]

                dist_from_center = math.hypot(abs(frame.shape[1] // 2 - x_tip), abs(frame.shape[0] // 2 - y_tip))
                mistake = abs(Settings.circle_radius - dist_from_center) // 4

                difference = 1 - (abs(Settings.circle_radius - dist_from_center) / Settings.circle_radius) * 2.5
                difference = max(0, difference)
                accuracy_sum += difference

                color_mistake = Settings.colors_error_from_radius[int(min(5, mistake))]
                if i > 0:
                    cv2.line(frame, (x_tip, y_tip),
                             (Settings.index_frame_circles[i - 1][0],
                              Settings.index_frame_circles[i - 1][1]), color_mistake, 4)

            if len(Settings.index_frame_circles) > 0:
                Settings.ACCURACY_DRAWING = accuracy_sum / len(Settings.index_frame_circles)

            cv2.putText(frame, f"Accuracy: {round(Settings.ACCURACY_DRAWING * 100, 1)}%",
                        (210, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))

            if hands.multi_hand_landmarks is not None:
                x_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].x * frame.shape[1])
                y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * frame.shape[0])
                Settings.index_frame_circles.append((x_index_tip, y_index_tip))

                if hands.multi_hand_landmarks is not None:
                    x_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].x * frame.shape[1])
                    y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * frame.shape[0])
                    Settings.index_frame_circles.append((x_index_tip, y_index_tip))

                    if len(Settings.index_frame_circles) > 150:
                        circles = Settings.index_frame_circles[-50:]
                        if isCircleClosed(circles, Settings.circle_radius):
                            Settings.CURRENT_PHASE = phaseEndGame


def phaseEndGame(frame) -> None:
    cv2.putText(frame, f"Accuracy: {round(Settings.ACCURACY_DRAWING * 100, 1)}%",
                (230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))

    best_score: float = loadBestScore()
    if Settings.ACCURACY_DRAWING > best_score:
        savBestScore(Settings.ACCURACY_DRAWING)
        cv2.putText(frame, "Congratulations! New Record!", (140, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

    drawCircleOnFrame(frame)

    cv2.putText(frame, "Press Enter to play again", (180, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))


Settings.CURRENT_PHASE = phaseCapturingFinger
with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as handsDetector:
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame_recorded = capture.read()
        if not ret:
            break

        flippedRGB = cv2.cvtColor(np.fliplr(frame_recorded), cv2.COLOR_BGR2RGB)
        hands_in_frame = checkHands(flippedRGB)

        if hands_in_frame or Settings.CURRENT_PHASE != phaseDrawingCircle:
            Settings.CURRENT_PHASE(flippedRGB)
        elif not hands_in_frame and Settings.CURRENT_PHASE == phaseDrawingCircle:
            phaseNoHandsFound(flippedRGB)

        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hands", res_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 13:
            resetSettings()
            Settings.CURRENT_PHASE = phaseCapturingFinger

    handsDetector.close()
    capture.release()
