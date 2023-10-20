from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.interpolate import LinearNDInterpolator

from .basic_transformation import BaseTransform
from .coords_utils import hom_coords, rotation_matrix


class ProjectionTransformation(BaseTransform):
    n: int = 9
    p = np.eye(3, 3, dtype=np.float32).reshape(-1)
    def __init__(self, p_init: Optional[ndarray] = None) -> None:
        if p_init is not None:
            assert p_init.shape == (self.n,), f'Wrong parameters shape, given: {p_init.shape}'
            self.p = p_init

    def apply_transformation_to_coordinates(self, coords: ndarray) -> Tuple[ndarray]:
        """_summary_

        Args:
            coords (ndarray): n x 3, X coordinates in camera Space

        Returns:
            ndarray: n x 2, points coordinates in image coordinates system
        """
        projection_matrix = self.p.reshape(3, 3)

        coords_new = np.copy(coords)
        coords_new = (projection_matrix @ coords_new.T).T
        coords_new[:, 0] /= coords_new[:, 2]
        coords_new[:, 1] /= coords_new[:, 2]

        # mask_visibility = np.zeros(coords_new.shape[0], dtype=np.bool_)
        # indexes = np.copy(coords_new[:, :2]).round().astype(int)
        # indexes[:, 0] += indexes[:, 0].min()
        # indexes[:, 1] += indexes[:, 1].min()

        # mask = np.zeros((indexes[:, 0].max(), indexes[:, 1].max()), dtype=np.bool_)
        # mask

        coords_new = coords_new[:, :2]

        return coords_new


class ProjectionPseudoInvTransformation(BaseTransform):
    n: int = 9
    p = np.eye(3, 3, dtype=np.float32).reshape(-1)
    def __init__(self, p_init: Optional[ndarray] = None) -> None:
        if p_init is not None:
            assert p_init.shape == (self.n,), f'Wrong parameters shape, given: {p_init.shape}'
            self.p = p_init

    def apply_transformation_to_coordinates(self, coords: ndarray, depth: ndarray) -> ndarray:
        """_summary_

        Args:
            coords (ndarray): n x 2,  coordinates in image coord system
            depth (ndarray): n, depth along ray for each pixel

        Returns:
            ndarray: n x 3, coordinates in camera coord system
        """
        projection_matrix = self.p.reshape(3, 3)
        projection_matrix_inv = np.linalg.inv(projection_matrix)


        coords_new = np.copy(coords)
        coords_new = hom_coords(coords_new)

        coords_new = coords_new * depth[..., None]

        coords_new = (projection_matrix_inv @ coords_new.T).T
        return coords_new


class RT_Transformation(BaseTransform):
    n: int = 6
    p: np.ndarray = np.zeros(6, dtype=np.float32)

    def apply_transformation_to_coordinates(self, coords: ndarray) -> ndarray:
        """_summary_

        Args:
            coords (ndarray): n x 3,  coordinates in camera coord system

        Returns:
            ndarray: n x 3, coordinates in camera coord system
        """
        R = rotation_matrix(self.p[:3])

        RT_matrix = np.zeros((4, 4), dtype=np.float32)
        RT_matrix[3, 3] = 1.
        RT_matrix[:3, :3] = R
        RT_matrix[:3, 3] = self.p[3:]


        coords_new = np.copy(coords)
        coords_new = hom_coords(coords_new)

        coords_new = (RT_matrix @ coords_new.T).T
        coords_new = coords_new[:, :3]

        return coords_new


class ReprojectionTransformation(BaseTransform):
    n: 6
    p: np.ndarray = np.zeros(6, dtype=np.float32)

    def __init__(self, p_init: np.ndarray, intrinsic: np.ndarray) -> None:
        super().__init__(p_init)

        self.camera_projection = ProjectionTransformation(intrinsic)
        self.camera_projection_inv = ProjectionPseudoInvTransformation(intrinsic)
        self.RT = RT_Transformation(p_init=p_init)


    def apply_transformation_to_coordinates(self, coords: np.ndarray, depth: Optional[np.array] = None) -> np.ndarray:
        # apply inverse projection

        X = self.camera_projection_inv.apply_transformation_to_coordinates(coords=coords, depth=depth)
        X = self.RT.apply_transformation_to_coordinates(coords=X)
        x = self.camera_projection.apply_transformation_to_coordinates(coords=X)

        return x

    def apply_transformation(self, image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """return transformed image

        Args:
            image (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            np.array: _description_
        """

        assert len(image.shape) == 3 or len(image.shape) == 3, f'image shape = {image.shape}'

        x_coord = np.arange(image.shape[1], dtype=np.float32) - float(image.shape[1]) / 2
        y_coord = np.arange(image.shape[0], dtype=np.float32) - float(image.shape[0]) / 2
        x_coord, y_coord = np.meshgrid(x_coord, y_coord, indexing='xy')

        image_pixels_coordinates = np.vstack(
            [
                x_coord.reshape(-1),
                y_coord.reshape(-1)
            ]
        ).T # n x 2

        transformed_coordinates = self.apply_transformation_to_coordinates(
            image_pixels_coordinates,
            depth=image[:, :, 1].reshape(-1)
        )

        image_values = image[:, :, 0].astype(np.float32).reshape(-1)

        inter_func = LinearNDInterpolator(
            points=transformed_coordinates,
            values=image_values,
            fill_value=0.
        )

        x_coord = np.arange(shape[1], dtype=np.float32) - float(shape[1]) / 2
        y_coord = np.arange(shape[0], dtype=np.float32) - float(shape[0]) / 2

        x_coord, y_coord = np.meshgrid(x_coord, y_coord, indexing='xy')  # 2D grid for interpolation

        transformed_image_values = inter_func(
            x_coord,
            y_coord,
        )

        indexes = np.copy(transformed_coordinates)
        indexes[:, 0] += float(image.shape[1]) / 2
        indexes[:, 1] += float(image.shape[0]) / 2
        indexes = indexes.round().astype(int)
        indexes = indexes[indexes[:, 0] >= 0]
        indexes = indexes[indexes[:, 0] < shape[1]]
        indexes = indexes[indexes[:, 1] >= 0]
        indexes = indexes[indexes[:, 1] < shape[0]]

        values_mask = np.zeros(shape, dtype=np.bool_)
        values_mask[indexes[:, 1], indexes[:, 0]] = 1

        transformed_image = np.zeros(shape, dtype=transformed_image_values.dtype)
        transformed_image[values_mask] = transformed_image_values.reshape(*shape)[values_mask]

        return transformed_image
