import numpy as np


def hom_coords(coords: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        coords (ndarray): n x m coords

    Returns:
        np.ndarray: n x (m + 1) extended coordinates
    """

    coords_extended = np.copy(np.hstack(
        [
            coords,
            np.ones((coords.shape[0], 1) , dtype=coords.dtype)
        ]
    ))
    return coords_extended
