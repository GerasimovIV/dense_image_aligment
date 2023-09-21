from typing import Tuple

import numpy as np

from .transformations.base_transformation import BaseTransform


def compute_image_grad(image: np.ndarray) -> np.ndarray:
    """compute ∇I(x) for each pixel

    Args:
        image (np.ndarray)

    Returns:
        np.ndarray: matrix with shape N x 2 where N is a number of pixels in image
    """

    gigx = np.zeros_like(image, dtype=image.dtype)
    gigx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
    gigy = np.zeros_like(image, dtype=image.dtype)
    gigy[1:-1] = (image[2:] - image[:-2]) / 2

    nabla_I = np.vstack(
        [
            gigx.reshape(-1),
            gigy.reshape(-1),
        ]
    ).T

    x_coord = np.arange(image.shape[1], dtype=np.float32)
    y_coord = np.arange(image.shape[0], dtype=np.float32)
    x_coord, y_coord = np.meshgrid(x_coord, y_coord, indexing='xy')

    image_pixels_coordinates = np.vstack(
        [
            x_coord.reshape(-1),
            y_coord.reshape(-1)
        ]
    ).T # n x 2


    return nabla_I, image_pixels_coordinates



def compute_J(
    image: np.ndarray,
    coord_transform: BaseTransform,
    p_c: np.ndarray,
) -> np.ndarray:
    """compute J(x, p_c) = ∇I(x) × ∂W(x, p) / ∂p |_{p=pc}, J has a shape N x n,
        where N is a number of pixels and n is the nuber of warp parameters

    Args:
        image (np.ndarray): image (I)
        coord_transform (BaseTransform): warp coordinates (W)
        p_c (np.ndarray): set of parameters for W

    Returns:
        np.ndarray: computed for each pixel J(x, p_c) matrix with shape N x n
        where N is the number of pixels and n is the number of warp parameters
    """
    nabla_I, x = compute_image_grad(image) # N x 2
    warp_jacobian = coord_transform.jacobian(x=x, p_c=p_c) # N x 2 x n
    J = np.einsum('ij,ijk->ik', nabla_I, warp_jacobian) # N x n
    return J


def compute_H(
    J: np.ndarray
) -> float:
    """compute H(x, p_c) = Σ J.T × J for given J(x, p_c)

    Args:
        J (np.ndarray): see compute_J function in dense_image_alignment.utils.compute_J

    Returns:
        float: exact value
    """

    H = np.einsum('ij,ij->i', J, J).sum()
    return H
