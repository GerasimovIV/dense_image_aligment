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



def theta_matrix(thetas: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        thetas (np.ndarray): [θ_1, θ_2, θ_3]

    Returns:
        np.ndarray: G_1 * θ_1 + G_2 * θ_2 + G_3 * θ_3
    """

    G_1 = np.zeros((3, 3), dtype=np.float32)
    G_2 = np.zeros((3, 3), dtype=np.float32)
    G_3 = np.zeros((3, 3), dtype=np.float32)

    G_1[1, 2] = -1.
    G_1[2, 1] = 1.

    G_2[0, 2] = 1.
    G_2[2, 0] = -1.

    G_3[0, 1] = -1.
    G_3[1, 0] = 1.

    return G_1 * thetas[0] + G_2 * thetas[1] + G_3 * thetas[2]

def rotation_matrix(thetas: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        thetas (np.ndarray): [θ_1, θ_2, θ_3]

    Returns:
        np.ndarray: exp(θ^)
    """

    theta_m = theta_matrix(thetas)

    theta_norm = np.linalg.norm(thetas, ord=2)

    R = np.eye(3)

    if theta_norm != 0:
        R = R \
            + np.sin(theta_norm) / theta_norm * theta_m  \
                + (1 - np.cos(theta_norm)) / theta_norm ** 2 * theta_m @ theta_m

    return R
