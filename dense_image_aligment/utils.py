from typing import Tuple

import numpy as np


def grad_image(
    image: np.ndarray,
    border: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    grad_image
    function to compute gradient of image over coordinate space
    by central difference method
    Let W, H := width and height of image. Then this function returns gigx and
    gigy, where
    gigx[x, y] := (image[x + 1, y] - image[x - 1, y]) / 2
    for x in (1, ..., W - 2) and y in (0, ..., H - 1),
    gigy[x, y] := (image[x, y + 1] - image[x, y - 1]) / 2
    for x in (0, ..., W - 1) and y in (1, ..., H - 2).
    If border is True, then the left and right columns of gigx and the top and
    bottom rows of gigy are computed by forward or backward difference method.
    Otherwise, those border values are set to 0.

    Parameters
    ----------
    image : np.ndarray
        input image
    border : bool, optional
        the left and right columns of gigx and the top and
        bottom rows of gigy are computed by forward or
        backward difference method, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        gigx and gigy
    """

    gigx = np.zeros_like(image)
    gigx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
    gigy = np.zeros_like(image)
    gigy[1:-1] = (image[2:] - image[:-2]) / 2

    if border:
        gigx[:, 0] = image[:, 1] - image[:, 0]
        gigx[:, -1] = image[:, -1] - image[:, -2]

        gigy[0] = image[1] - image[0]
        gigy[-1] = image[-1] - image[-2]

    return gigx, gigy
