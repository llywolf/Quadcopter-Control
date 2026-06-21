import numpy as np


INITIAL_OBSTACLE_SHAPES = [
    [[5, 5], [9, 5], [7, 10]],
    [[12, 2], [17, 3], [16, 7], [11, 8]],
    [[10, 14], [12, 12], [14, 12], [16, 16], [12, 16]],
    [[3, 14], [6.5, 14], [6.5, 19], [3, 19]],
]


def get_initial_obstacle_shapes():
    return [
        np.array(shape, dtype=float)
        for shape in INITIAL_OBSTACLE_SHAPES
    ]


def load_initial_base_obstacles():
    return [
        {"vertices": np.array(shape, dtype=float)}
        for shape in INITIAL_OBSTACLE_SHAPES
    ]