import numpy as np

__G_matixes__ = {
    0: np.array(
        [
            [0, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ],
        dtype=np.float32
    ),
    1: np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        dtype=np.float32
    ),
    2: np.array(
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        dtype=np.float32
    ),
    3: np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        dtype=np.float32
    ),
    4: np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        dtype=np.float32
    ),
    5: np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ],
        dtype=np.float32
    ),
}

def G_matrix(key:int) -> np.ndarray:
    return __G_matixes__[key]
