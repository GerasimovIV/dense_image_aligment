import numpy as np 
from typing import Tuple 
from scipy.interpolate import LinearNDInterpolator


class BaseTransform(object):
    
    def __init__(self, p_init: np.ndarray) -> None:
        self.p = p_init
    
    def jacobian(self, p_c: np.ndarray) -> np.ndarray:
        """return $\frac{\partial W}{\partial p}(p_c) $

        Raises:
            NotImplementedError: _description_

        Returns:
            np.ndarray: 2 x n matrix (derivative for x coordinate and y)
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
        
        assert len(image.shape) == 2, f'image shape = {image.shape}'
        
        x_coord = np.arange(image.shape[1], dtype=np.float32)
        y_coord = np.arange(image.shape[0], dtype=np.float32)
        x_coord, y_coord = np.meshgrid(x_coord, y_coord, indexing='xy')
        
        image_pixels_coordinates = np.vstack(
            [
                x_coord.reshape(-1), 
                y_coord.reshape(-1)
            ]
        ).T # n x 2
        
        transformed_coordinates = self.apply_transformation_to_coordinates(
            image_pixels_coordinates
        )
        
        image_values = image.astype(np.float32).reshape(-1)
        
        inter_func = LinearNDInterpolator(
            points=transformed_coordinates,
            values=image_values,
            fill_value=0.
        )
        
        x_coord = np.arange(shape[1], dtype=np.float32)
        y_coord = np.arange(shape[0], dtype=np.float32)
        
        x_coord, y_coord = np.meshgrid(x_coord, y_coord, indexing='xy')  # 2D grid for interpolation
        
        transformed_image_values = inter_func(
            x_coord,
            y_coord,
        )
        
        return transformed_image_values.reshape(*shape)
        
        
    def apply_transformation_to_coordinates(self, coords: np.ndarray) -> np.ndarray:
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