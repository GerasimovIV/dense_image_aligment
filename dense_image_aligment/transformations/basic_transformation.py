from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class BaseTransform(object):
    n: int
    p: np.ndarray
    def __init__(self, p_init: np.ndarray) -> None:
        self.p = p_init

    def jacobian(self, x: np.array, p_c: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x (ndarray): N x 2 matrix of coordinates
            p_c (ndarray): n vector of parameters

        Returns:
            ndarray: N x 2 x n
        """
        raise NotImplementedError

    def apply_transformation(self, image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """return transformed image

        Args:
            image (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            np.array: _description_
        """

        assert len(image.shape) == 2 or len(image.shape) == 3, f'image shape = {image.shape}'

        x_coord = np.arange(image.shape[1], dtype=np.float32) - float(image.shape[1]) / 2
        y_coord = np.arange(image.shape[0], dtype=np.float32) - float(image.shape[0]) / 2
        x_coord, y_coord = np.meshgrid(x_coord, y_coord, indexing='xy')

        image_pixels_coordinates = np.vstack(
            [
                x_coord.reshape(-1),
                y_coord.reshape(-1)
            ]
        ).T # n x 2

        if len(image.shape) == 2:
            transformed_coordinates = self.apply_transformation_to_coordinates(
                image_pixels_coordinates
            )
        elif len(image.shape) == 3:
            transformed_coordinates = self.apply_transformation_to_coordinates(
                image_pixels_coordinates,
                depth=image[:, :, 1].reshape(-1)
            )

        if isinstance(transformed_coordinates, Tuple):
            transformed_coordinates, inter_mask = transformed_coordinates
            transformed_coordinates = transformed_coordinates[inter_mask]
            image_values = image_values[inter_mask]

        if len(image.shape) == 2:
            image_values = image.astype(np.float32).reshape(-1)
        elif len(image.shape) == 3:
            image_values = image[:, :, 0].astype(np.float32).reshape(-1)
        else:
            raise NotImplementedError



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

        transformed_image = transformed_image_values.reshape(*shape)

        return transformed_image



    def apply_transformation_to_coordinates(self, coords: np.ndarray, depth: Optional[np.array] = None) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """apply transformation to image coordinates

        Args:
            coords (np.ndarray): N x 2 matrix (x, y coordinates)

        Returns:
            np.ndarray: transformed coordinates N x 2 matrix
        """
        raise NotImplementedError

    def inverse_transform(self) -> 'BaseTransform':
        """return inverse transformation to given

        Raises:
            NotImplementedError: _description_

        Returns:
            BaseTransform: _description_
        """
        raise NotImplementedError

    def apply_inverse_transformation_to_coordinates(self, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __mul__(self, coords: np.ndarray) -> np.ndarray:
        return self.apply_transformation_to_coordinates(coords=coords)
