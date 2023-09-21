from copy import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .transformations.base_transformation import BaseTransform
from .utils import compute_H, compute_image_grad, compute_J


def forward_additive(
    template: np.ndarray,
    image: np.ndarray,
    coord_transform: BaseTransform,
    p_init: Optional[np.ndarray] = None,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-4,
    alpha: float = 1.,
    verbose: bool = True,
) -> List[np.ndarray]:
    '''Lucas-Kanade forward additive method (Fig. 1) for affine warp model.

    Args:
        template (numpy.ndarray):
            A grayscale image of shape (height, width).
        image (numpy.ndarray):
            A template image (height_t, width_t).
        coord_transform (BaseTransform):
            image warp
        p_init (Optional[numpy.ndarray]):
            Initial parameters for image warp,
            if it is not given internal parameters in coord_transform will be used
        max_iterations (int):
            number of iteration
        convergence_threshold (float):
            stop iterations if ||∇p||_2 < convergence_threshold
        alpha (float):
            p = p_prev + ∇p * alpha

    Returns:
        ps (list):
            Estimates of parameters (p) for each iteration.
    '''
    progression_bar = tqdm(range(max_iterations), disable=not verbose)

    p_c = copy(p_init) if p_init is not None else copy(coord_transform.p)

    ps = [p_c]

    for it in progression_bar:
        coord_transform.p = p_c

        I_W = coord_transform.apply_transformation(image=image, shape=template.shape)

        J = compute_J(
            image=I_W,
            coord_transform=coord_transform,
            p_c=p_c
        )

        H = compute_H(J)

        diff = template.reshape(-1) - I_W.reshape(-1)

        delta_p = 1 / H * (np.einsum('Nn,N->n', J, diff))

        p_c = p_c + alpha * delta_p
        ps.append(p_c)

        delta_p_norm = np.linalg.norm(delta_p)

        progression_bar.set_description(f'iteration: {it}, |∇p|={delta_p_norm:.5f}')

        if delta_p_norm <= convergence_threshold:
            if verbose: print('Converged')
            break
    return ps


__methods__ = {
    'forward_additive': forward_additive,
}

__def_params__ = {
    'forward_additive': {
        'max_iterations': 200,
        'convergence_threshold': 1e-4,
        'verbose': True,
        'alpha': 1.0,
        },
}


def image_aligment_method(key: str) -> Tuple[Callable, Dict[str, Any]]:
    """
    image_aligment_method define method and set of its default parameters

    Parameters
    ----------
    key : str
        name of the method, one of:
            forward_compositional
            forward_additive
            inverse_compositional

    Returns
    -------
    Tuple[Callable, Dict[str, Any]]
        method and its default parameters
    """
    return __methods__[key], __def_params__[key]
