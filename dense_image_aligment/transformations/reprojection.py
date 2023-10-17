from typing import Optional

import numpy as np
from numpy import ndarray

from .basic_transformation import BaseTransform
from .coords_utils import hom_coords


class ProjectionTransformation(BaseTransform):
    n: int = 9
    p = np.eye(3, 3, dtype=np.float32).reshape(-1)
    def __init__(self, p_init: Optional[ndarray] = None) -> None:
        if p_init is not None:
            assert p_init.shape == (self.n,), f'Wrong parameters shape, given: {p_init.shape}'
            self.p = p_init


    def apply_inverse_transformation_to_coordinates(self, coords: ndarray, depths: ndarray) -> ndarray:
        """_summary_

        Args:
            coords (ndarray): n x 2, points coordinates in image coordinates system
            depths (ndarray): n, depths for each point (along point ray, not exactly z coordinate)

        Returns:
            ndarray: n x 3, X coordinates in scene Space
        """
        raise NotImplementedError


    def apply_transformation_to_coordinates(self, coords: ndarray) -> ndarray:
        """_summary_

        Args:
            coords (ndarray): n x 3, X coordinates in scene Space

        Returns:
            ndarray: n x 2, points coordinates in image coordinates system
        """
        projection_matrix = self.p.reshape(3, 3)

        coords_new = np.copy(coords)
        coords_new = (projection_matrix @ coords_new.T).T
        coords_new[:, 0] /= coords_new[:, 2]
        coords_new[:, 1] /= coords_new[:, 2]
        coords_new = coords_new[:, :2]

        return coords_new





class ReprojectionTransformation(BaseTransform):
    # n: int = 8
    # p = np.eye(3, 3, dtype=np.float32).reshape(-1)[:8]

    def __init__(self, p_init: Optional[ndarray] = None) -> None:
        raise NotImplementedError
        # if p_init is not None:
        #     assert p_init.shape == (self.n,), f'Wrong parameters shape, given: {p_init.shape}'
        #     self.p = p_init

    def transformed_hom_coords(self, coords: ndarray, p_c: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

        # coords_extended = np.copy(np.hstack(
        #     [
        #         coords,
        #         np.ones((coords.shape[0], 1) , dtype=coords.dtype)
        #     ]
        # ))

        # p = p_c if p_c is not None else self.p

        # warp_matrix = np.ones(9)
        # warp_matrix[:8] = p
        # warp_matrix = warp_matrix.reshape(3, 3)

        # coords_new = (warp_matrix @ coords_extended.T).T # n x 3
        # return coords_new


    def jacobian(self, x: ndarray, p_c: ndarray) -> ndarray:
        raise NotImplementedError

        # N = x.shape[0]

        # jacobian = np.zeros((N, 2, self.n), dtype=np.float32)

        # trasformed_coords = self.transformed_hom_coords(x, p_c)

        # scale = trasformed_coords[:, 2]

        # jacobian[:, 0, 0] = x[:, 0]
        # jacobian[:, 0, 1] = x[:, 1]
        # jacobian[:, 0, 2] = 1.

        # jacobian[:, 0, 6] = - trasformed_coords[:, 0] * x[:, 0] / (scale) ** 2
        # jacobian[:, 0, 7] = - trasformed_coords[:, 0] * x[:, 1] / (scale) ** 2

        # jacobian[:, 1, 3] = x[:, 0]
        # jacobian[:, 1, 4] = x[:, 1]
        # jacobian[:, 1, 5] = 1.

        # jacobian[:, 1, 6] = - trasformed_coords[:, 1] * x[:, 0] / (scale) ** 2
        # jacobian[:, 1, 7] = - trasformed_coords[:, 1] * x[:, 1] / (scale) ** 2

        # return jacobian

    def apply_transformation_to_coordinates(self, coords: ndarray) -> ndarray:
        raise NotImplementedError

        # coords_new = self.transformed_hom_coords(coords)
        # coords_new[:, 0] /= coords_new[:, 2]
        # coords_new[:, 1] /= coords_new[:, 2]

        # return coords_new[:, :2]
