from .base_transformation import BaseTransform
import numpy as np
from typing import Optional
from numpy import ndarray


class AffineTransformation(BaseTransform):
    n: int = 6
    p = np.eye(2, 3, dtype=np.float32).reshape(-1)
    
    def __init__(self, p_init: Optional[ndarray] = None) -> None:
        
        if p_init is not None:
            assert p_init.shape == (6,), f'Wrong parameters shape, given: {p_init.shape}'
            self.p = p_init
        
        
    def jacobian(self, x: ndarray, p_c: ndarray) -> ndarray:
        """_summary_

        Args:
            x (ndarray): N x 2 metrix of coordinates
            p_c (ndarray): n vector of parameters

        Returns:
            ndarray: _description_
        """
        N = x.shape[0]
        
        jacobian = np.zeros((N, self.n, 2))
        
        jacobian[:, 0, 0] = x[:, 0]
        jacobian[:, 1, 0] = x[:, 1]
        jacobian[:, 2, 0] = 1.
        
        jacobian[:, 3, 1] = x[:, 0]
        jacobian[:, 4, 1] = x[:, 1]
        jacobian[:, 5, 1] = 1.
        
        
        return jacobian
    
    def apply_transformation_to_coordinates(self, coords: ndarray) -> ndarray:
        coords_extended = np.hstack(
            [
                coords,
                np.ones((coords.shape[0], 1) , dtype=coords.dtype)
            ]
        )
        
        warp_matrix = np.zeros((2, 3))
        warp_matrix[0, :] = self.p[:3]
        warp_matrix[1, :] = self.p[3:]
        
        return coords_extended @ warp_matrix.T