from typing import Optional

import numpy as np
from numpy import ndarray

from .basic_transformation import BaseTransform
from .coords_utils import hom_coords


class HomographyTransformation(BaseTransform):
    n: int = 8
    p = np.eye(3, 3, dtype=np.float32).reshape(-1)[:8]

    def __init__(self, p_init: Optional[ndarray] = None) -> None:

        if p_init is not None:
            assert p_init.shape == (self.n,), f'Wrong parameters shape, given: {p_init.shape}'
            self.p = p_init

    def transformed_hom_coords(self, coords: ndarray, p_c: Optional[np.ndarray] = None) -> np.ndarray:
        coords_extended = hom_coords(coords)

        p = p_c if p_c is not None else self.p

        warp_matrix = np.ones(9)
        warp_matrix[:8] = p
        warp_matrix = warp_matrix.reshape(3, 3)

        coords_new = (warp_matrix @ coords_extended.T).T # n x 3
        return coords_new


    def jacobian(self, x: ndarray, p_c: ndarray) -> ndarray:

        N = x.shape[0]

        jacobian = np.zeros((N, 2, self.n), dtype=np.float32)

        trasformed_coords = self.transformed_hom_coords(x, p_c)

        scale = trasformed_coords[:, 2]

        jacobian[:, 0, 0] = x[:, 0]
        jacobian[:, 0, 1] = x[:, 1]
        jacobian[:, 0, 2] = 1.

        jacobian[:, 0, 6] = - trasformed_coords[:, 0] * x[:, 0] / (scale) ** 2
        jacobian[:, 0, 7] = - trasformed_coords[:, 0] * x[:, 1] / (scale) ** 2

        jacobian[:, 1, 3] = x[:, 0]
        jacobian[:, 1, 4] = x[:, 1]
        jacobian[:, 1, 5] = 1.

        jacobian[:, 1, 6] = - trasformed_coords[:, 1] * x[:, 0] / (scale) ** 2
        jacobian[:, 1, 7] = - trasformed_coords[:, 1] * x[:, 1] / (scale) ** 2

        return jacobian

    def apply_transformation_to_coordinates(self, coords: ndarray) -> ndarray:
        coords_new = self.transformed_hom_coords(coords)
        coords_new[:, 0] /= coords_new[:, 2]
        coords_new[:, 1] /= coords_new[:, 2]

        return coords_new[:, :2]
