from typing import Optional

import numpy as np
from numpy import ndarray

from .basic_transformation import BaseTransform


class TranslationTransformation(BaseTransform):
    n: int = 2
    p = np.array([0., 0.], dtype=np.float32)

    def __init__(self, p_init: Optional[ndarray] = None) -> None:

        if p_init is not None:
            assert p_init.shape == (2,), f'Wrong parameters shape, given: {p_init.shape}'
            self.p = p_init


    def jacobian(self, x: ndarray, p_c: ndarray) -> ndarray:

        N = x.shape[0]

        jacobian = np.zeros((N, 2, self.n), dtype=np.float32)

        jacobian[:, 0, 0] = 1.
        jacobian[:, 0, 1] = 0.

        jacobian[:, 1, 0] = 0.
        jacobian[:, 1, 1] = 1.

        return jacobian

    def apply_transformation_to_coordinates(self, coords: ndarray) -> ndarray:
        coords_new = np.copy(coords)
        coords_new[:, 0] += self.p[0]
        coords_new[:, 1] += self.p[1]
        return coords_new
