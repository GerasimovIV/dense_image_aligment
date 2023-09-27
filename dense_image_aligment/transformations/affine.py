from typing import Optional

import numpy as np
from numpy import ndarray

from .basic_transformation import BaseTransform


class AffineTransformation(BaseTransform):
    n: int = 6
    p = np.eye(2, 3, dtype=np.float32).reshape(-1)

    def __init__(self, p_init: Optional[ndarray] = None) -> None:

        if p_init is not None:
            assert p_init.shape == (6,), f'Wrong parameters shape, given: {p_init.shape}'
            self.p = p_init


    def jacobian(self, x: ndarray, p_c: ndarray) -> ndarray:

        N = x.shape[0]

        jacobian = np.zeros((N, 2, self.n))

        jacobian[:, 0, 0] = x[:, 0]
        jacobian[:, 0, 1] = x[:, 1]
        jacobian[:, 0, 2] = 1.

        jacobian[:, 1, 3] = x[:, 0]
        jacobian[:, 1, 4] = x[:, 1]
        jacobian[:, 1, 5] = 1.

        return jacobian

    def apply_transformation_to_coordinates(self, coords: ndarray) -> ndarray:
        coords_extended = np.copy(np.hstack(
            [
                coords,
                np.ones((coords.shape[0], 1) , dtype=coords.dtype)
            ]
        ))

        warp_matrix = np.zeros((2, 3))
        warp_matrix[0, :] = self.p[:3]
        warp_matrix[1, :] = self.p[3:]

        return (warp_matrix @ coords_extended.T).T
