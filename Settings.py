DRAWING: bool = False
only_index_finger: bool = False

CURRENT_PHASE: callable = None

index_frame_circles: list[tuple[int, int]] = []  # Координаты точек, рисующих круг
circle_radius: float = 0  # Радиус круга, который нужно нарисовать

colors_error_from_radius: list[tuple[int, int, int]] = [
    (61, 255, 24),
    (148, 255, 24),
    (218, 255, 24),
    (255, 222, 24),
    (255, 176, 24),
    (255, 97, 24)
]

SECONDS_UNTIL_DRAWING: int = 5
REMOVE_SECOND_UNTIL_DRAWING: int = 20

START_PHASE_TWO: int = 0

ACCURACY_DRAWING: float = 1

RESULTS_FILE: str = "results.txt"
