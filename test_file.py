import pygame
import random
import cv2
import mediapipe as mp

pygame.init()
screen = pygame.display.set_mode((300, 600))
pygame.display.set_caption('Gesture-Controlled Tetris')

WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLACK = (0, 0, 0)
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (128, 0, 128), (255, 165, 0)]

BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20

SHAPES = [
    [[[1, 1, 1, 1]], [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]],
    [[[1, 1, 1], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 1]], [[0, 0, 1], [0, 1, 1], [0, 0, 1]]],
    [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]],
    [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
    [[[1, 1], [1, 1]]],
    [[[1, 1, 1], [1, 0, 0]], [[1, 0], [1, 0], [1, 1]], [[0, 0, 1], [1, 1, 1]], [[1, 1], [0, 1], [0, 1]]],
    [[[1, 1, 1], [0, 0, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 0, 0], [1, 1, 1]], [[0, 1], [0, 1], [1, 1]]],
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
current_piece = None
piece_x, piece_y = 0, 0
clock = pygame.time.Clock()
last_finger_position = None
score = 0
previous_positions = None
current_positions = None


class Tetrimino:
    def __init__(self, shape):
        self.shape = shape
        self.color = random.choice(COLORS)
        self.rotation = 0

    def rotate(self, direction=1):
        self.rotation = (self.rotation + direction) % len(self.shape)

    def get_current(self):
        return self.shape[self.rotation]


def create_piece():
    global score
    shape = random.choice(SHAPES)
    score += 1
    return Tetrimino(shape)


def is_valid_position(piece, offset_x, offset_y):
    for y, row in enumerate(piece.get_current()):
        for x, cell in enumerate(row):
            if cell:
                grid_x = x + offset_x
                grid_y = y + offset_y
                if grid_x < 0 or grid_x >= GRID_WIDTH or grid_y >= GRID_HEIGHT or grid[grid_y][grid_x]:
                    return False
    return True


def place_piece(piece, offset_x, offset_y):
    for y, row in enumerate(piece.get_current()):
        for x, cell in enumerate(row):
            if cell:
                grid[offset_y + y][offset_x + x] = piece.color


def clear_lines():
    global grid, score
    lines_to_clear = [y for y in range(GRID_HEIGHT) if all(grid[y])]
    for y in lines_to_clear:
        del grid[y]
        grid.insert(0, [0] * GRID_WIDTH)
        score += 10


def draw_grid():
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, GRAY if grid[y][x] == 0 else grid[y][x], rect)
            pygame.draw.rect(screen, WHITE, rect, 1)


def draw_piece(piece, offset_x, offset_y):
    for y, row in enumerate(piece.get_current()):
        for x, cell in enumerate(row):
            if cell:
                rect = pygame.Rect((offset_x + x) * BLOCK_SIZE, (offset_y + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(screen, piece.color, rect)
                pygame.draw.rect(screen, WHITE, rect, 1)


def get_finger_positions(hand_landmarks):
    return {
        'thumb': (hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x,
                  hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y),
        'index': (hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x,
                  hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y),
        'middle': (hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                 hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y)
    }


def check_finger_swap(previous_positions, current_positions):
    if not previous_positions or not current_positions:
        return False


    return previous_positions['thumb'][1] < previous_positions['middle'][1] and current_positions['thumb'][1] > current_positions['middle'][1] and current_positions['thumb'][1] - previous_positions['thumb'][1] > 0.02 and current_positions['middle'][1] - previous_positions['middle'][1] < -0.03


def detect_gesture(frame, previous_positions, current_positions):
    global last_finger_position
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    gesture = None
    annotated_frame = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            positions = get_finger_positions(hand_landmarks)
            h, w, _ = frame.shape

            for name, pos in positions.items():
                cx, cy = int(pos[0] * w), int(pos[1] * h)
                if name == 'index':
                    cv2.circle(annotated_frame, (cx, cy), 10, (0, 0, 255), -1)
                else:
                    cv2.circle(annotated_frame, (cx, cy), 10, (0, 255, 0), -1)

            if last_finger_position:
                dx = positions['index'][0] - last_finger_position[0]
                dy = positions['index'][1] - last_finger_position[1]

                if check_finger_swap(previous_positions, current_positions):
                    gesture = "rotate"
                elif abs(dx) > abs(dy):
                    if dx > 0.02:
                        gesture = "left"
                    elif dx < -0.02:
                        gesture = "right"
                else:
                    if dy > 0.06 and abs(dx) < 0.02:
                        gesture = "down"

            last_finger_position = positions['index']

    return gesture, annotated_frame


def is_valid_rotation(piece, offset_x, offset_y, direction):
    next_rotation = (piece.rotation + direction) % len(piece.shape)
    next_shape = piece.shape[next_rotation]

    for y, row in enumerate(next_shape):
        for x, cell in enumerate(row):
            if cell:
                grid_x = x + offset_x
                grid_y = y + offset_y
                if grid_x < 0 or grid_x >= GRID_WIDTH or grid_y >= GRID_HEIGHT or grid[grid_y][grid_x]:
                    return False
    return True


current_piece = create_piece()
piece_x, piece_y = GRID_WIDTH // 2, 0
fall_time = 0

running = True
while running:
    screen.fill(BLACK)

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_h = hands.process(frame_rgb)

    if results_h.multi_hand_landmarks:
        hand_landmarks = results_h.multi_hand_landmarks[0]
        current_positions = get_finger_positions(hand_landmarks)

    gesture, annotated_frame = detect_gesture(frame, previous_positions, current_positions)
    previous_positions = current_positions
    if gesture == "left" and is_valid_position(current_piece, piece_x - 1, piece_y):
        piece_x -= 1
    elif gesture == "right" and is_valid_position(current_piece, piece_x + 1, piece_y):
        piece_x += 1
    elif gesture == "down":
        while is_valid_position(current_piece, piece_x, piece_y + 1):
            piece_y += 1
    elif gesture == "rotate" and is_valid_rotation(current_piece, piece_x, piece_y, 1):
        current_piece.rotate()

    fall_time += clock.get_rawtime()
    if fall_time > 500:
        if is_valid_position(current_piece, piece_x, piece_y + 1):
            piece_y += 1
        else:
            place_piece(current_piece, piece_x, piece_y)
            clear_lines()
            current_piece = create_piece()
            piece_x, piece_y = GRID_WIDTH // 2, 0
            if not is_valid_position(current_piece, piece_x, piece_y):
                running = False
        fall_time = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_grid()
    draw_piece(current_piece, piece_x, piece_y)
    pygame.display.flip()

    cv2.imshow("Camera", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

    clock.tick(30)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
print(f"Game over! Your result: {score} points.")
