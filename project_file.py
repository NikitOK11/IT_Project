import math
import cv2
import mediapipe as mp
import numpy as np
import Settings
import os


def check_only_finger(image) -> bool:
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


def load_best_score():
    if os.path.exists(Settings.RESULTS_FILE):
        with open(Settings.RESULTS_FILE, "r") as file:
            try:
                return float(file.read().strip())
            except ValueError:
                return 0.0
    return 0.0


def save_best_score(score):
    with open(Settings.RESULTS_FILE, "w") as file:
        file.write(f"{score:.1f}")


def reset_settings() -> None:
    Settings.DRAWING = False
    Settings.only_index_finger = False

    Settings.index_frame_circles = []
    Settings.circle_radius = 0

    Settings.SECONDS_UNTIL_DRAWING = 5
    Settings.REMOVE_SECOND_UNTIL_DRAWING = 20
    Settings.START_PHASE_TWO = 0

    Settings.ACCURACY_DRAWING = 1

    Settings.CURRENT_PHASE = check_only_finger


# noinspection PyGlobalUndefined
def phaseCapturingFinger(frm) -> None:
    cv2.circle(frm, (frm.shape[1] // 2, frm.shape[0] // 2), 5, (44, 62, 80), -1)
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
        cv2.putText(flippedRGB, "Can't see your index finger...", (120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0))


# noinspection PyGlobalUndefined
def phaseDrawingCircle(frm) -> None:
    hands = handsDetector.process(frm)

    cv2.circle(frm, (frm.shape[1] // 2, frm.shape[0] // 2), 5, (44, 62, 80), -1)
    if not Settings.DRAWING:
        cv2.putText(flippedRGB, "Place your finger to the point from which you start!", (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
        cv2.putText(frm, f"Start drawing in {Settings.SECONDS_UNTIL_DRAWING} seconds", (140, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0))

        if Settings.REMOVE_SECOND_UNTIL_DRAWING == 30:
            Settings.SECONDS_UNTIL_DRAWING -= 1
            Settings.REMOVE_SECOND_UNTIL_DRAWING = 0
        else:
            Settings.REMOVE_SECOND_UNTIL_DRAWING += 1

        if Settings.SECONDS_UNTIL_DRAWING == 0:
            if hands.multi_hand_landmarks is not None:
                x_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].x * frm.shape[1])
                y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * frm.shape[0])
                Settings.circle_radius = math.hypot(abs(frm.shape[1] // 2 - x_index_tip), abs(frm.shape[0] // 2 - y_index_tip))

                Settings.index_frame_circles.append((x_index_tip, y_index_tip))
            Settings.DRAWING = True
    else:
        x_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].x * frm.shape[1])
        y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * frm.shape[0])
        if math.hypot(x_index_tip, y_index_tip, frm.shape[1] // 2, frm.shape[0] // 2) < 10:
            cv2.putText(frm, f"Your finger is too close to center!", (105, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0))
        else:
            cv2.putText(frm, f"Draw the circle around the dot", (105, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0))

            accuracy_sum = 0
            for i in range(len(Settings.index_frame_circles)):
                circle_point = Settings.index_frame_circles[i]
                x_tip, y_tip = circle_point[0], circle_point[1]

                dist_from_center = math.hypot(abs(frm.shape[1] // 2 - x_tip), abs(frm.shape[0] // 2 - y_tip))
                mistake = abs(Settings.circle_radius - dist_from_center) // 4

                difference = 1 - (abs(Settings.circle_radius - dist_from_center) / Settings.circle_radius) * 2.5
                difference = max(0, difference)
                accuracy_sum += difference

                color_mistake = Settings.colors_error_from_radius[int(min(5, mistake))]
                if i > 0:
                    cv2.line(frm, (x_tip, y_tip),
                             (Settings.index_frame_circles[i - 1][0],
                              Settings.index_frame_circles[i - 1][1]), color_mistake, 4)

            if len(Settings.index_frame_circles) > 0:
                Settings.ACCURACY_DRAWING = accuracy_sum / len(Settings.index_frame_circles)

            cv2.putText(frm, f"Accuracy: {round(Settings.ACCURACY_DRAWING * 100, 1)}%",
                        (230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))

            if hands.multi_hand_landmarks is not None:
                x_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].x * frm.shape[1])
                y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * frm.shape[0])
                Settings.index_frame_circles.append((x_index_tip, y_index_tip))

                if hands.multi_hand_landmarks is not None:
                    x_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].x * frm.shape[1])
                    y_index_tip = int(hands.multi_hand_landmarks[0].landmark[8].y * frm.shape[0])
                    Settings.index_frame_circles.append((x_index_tip, y_index_tip))

                    if len(Settings.index_frame_circles) > 150:
                        start_point = Settings.index_frame_circles[0]
                        recent_points = Settings.index_frame_circles[-10:]
                        close_to_start_count = 0

                        for point in recent_points:
                            distance = math.hypot(start_point[0] - point[0], start_point[1] - point[1])
                            if distance < Settings.circle_radius * 0.1:
                                close_to_start_count += 1

                        if close_to_start_count >= 5:
                            Settings.CURRENT_PHASE = phaseEndGame


def phaseEndGame(frm) -> None:
    cv2.putText(frm, f"Accuracy: {round(Settings.ACCURACY_DRAWING * 100, 1)}%",
                (230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))

    best_score = load_best_score()
    if Settings.ACCURACY_DRAWING > best_score:
        save_best_score(Settings.ACCURACY_DRAWING)
        cv2.putText(frm, "Congratulations! New Record!", (140, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

    for i in range(len(Settings.index_frame_circles)):
        circle_point = Settings.index_frame_circles[i]
        x_tip, y_tip = circle_point[0], circle_point[1]

        dist_from_center = math.hypot(abs(frm.shape[1] // 2 - x_tip), abs(frm.shape[0] // 2 - y_tip))
        mistake = abs(Settings.circle_radius - dist_from_center) // 4
        color_mistake = Settings.colors_error_from_radius[int(min(5, mistake))]
        if i > 0:
            cv2.line(frm, (x_tip, y_tip),
                     (Settings.index_frame_circles[i - 1][0],
                      Settings.index_frame_circles[i - 1][1]), color_mistake, 4)
        else:
            cv2.line(frm, (x_tip, y_tip),
                     (Settings.index_frame_circles[-1][0],
                      Settings.index_frame_circles[-1][1]), color_mistake, 4)

    cv2.putText(frm, "Press Enter to play again", (180, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

    if cv2.waitKey(1) & 0xFF == 13:
        reset_settings()
        Settings.CURRENT_PHASE = phaseCapturingFinger


Settings.CURRENT_PHASE = phaseCapturingFinger
with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as handsDetector:
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        flippedRGB = cv2.cvtColor(np.fliplr(frame), cv2.COLOR_BGR2RGB)
        Settings.CURRENT_PHASE(flippedRGB)
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)

        cv2.imshow("Hands", res_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 13:
            reset_settings()
            Settings.CURRENT_PHASE = phaseCapturingFinger

    handsDetector.close()
    capture.release()
