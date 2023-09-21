from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .utils import grad_image


def forward_additive(
    template: np.ndarray,
    image: np.ndarray,
    p_init: Optional[np.ndarray] = None,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-4,
    alpha: float = 1.,
    verbose: bool = True
) -> List[np.ndarray]:
    '''Lucas-Kanade forward additive method (Fig. 1) for affine warp model.

    Args:
        template (numpy.ndarray):
            A grayscale image of shape (height, width).
        image (numpy.ndarray):
            A template image (height_t, width_t).
        p_init (numpy.ndarray):
            Initial parameter of affine transform which shape is (2, 3).
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

    # Warp function W(x; p) is a mapping from x_template to x_image
    height, width = template.shape
    gx_template, gy_template = grad_image(template)

    # initialize p
    if p_init is not None:
        p = p_init.copy()
    else:
        p = np.array([[1.2, 0.0, 50],
                      [0.0, 1.2, 70]], dtype='d')
    ps = [p.copy()]

    progression_bar = tqdm(range(max_iterations), disable=not verbose)

    for it in progression_bar:
        p_inv = cv2.invertAffineTransform(p)
        # step (1)
        template_w = cv2.warpAffine(template, p_inv, image.shape[::-1])

        # step (2)
        error = image - template_w

        # step (3)
        gx_template_w = cv2.warpAffine(gx_template, p_inv, image.shape[::-1])
        gy_template_w = cv2.warpAffine(gy_template, p_inv, image.shape[::-1])

        # step (4)
        g_template_w_height, g_template_w_width = gx_template_w.shape
        x, y = np.meshgrid(np.arange(g_template_w_width), np.arange(g_template_w_height))
        zero = np.zeros_like(x)
        one = np.ones_like(x)
        gp_warp = np.array([[x, y, one, zero, zero, zero],
                            [zero, zero, zero, x, y, one]], dtype='d')

        # step (5)
        g_template_w = np.stack((gx_template_w, gy_template_w), axis=0)
        # compute the steepest descent images
        sd_templates = np.einsum('jhw,jihw->ihw', g_template_w, gp_warp)

        # step (6)
        sd_templates_flat = sd_templates.reshape(6, -1)
        hessian = sd_templates_flat.dot(sd_templates_flat.T)

        # step (7)
        # steepest descent parameter update
        sd_updates = sd_templates_flat.dot(error.ravel())

        # step (8)
        p_update = sd_updates.dot(np.linalg.inv(hessian)).reshape(p.shape)

        # step (9)
        p += p_update * alpha

        ps.append(p.copy())

        delpa_p_norm = np.linalg.norm(p_update)

        progression_bar.set_description(f'iteration: {it}, |∇p|={delpa_p_norm:.5f}')

        # converged?
        if delpa_p_norm < convergence_threshold:
            if verbose: print('Converged.')
            break
    return ps


__p_init_base__ = np.array(
    [
        [1., 0., 0.],
        [0., 1., 0.],
    ]
)


__methods__ = {
    'forward_additive': forward_additive,
}

__def_params__ = {
    'forward_compositional': {
        'p_init': __p_init_base__,
        'max_iterations': 200,
        'convergence_threshold': 1e-4,
        'verbose': True,
        },
    'forward_additive': {
        'p_init': __p_init_base__,
        'max_iterations': 200,
        'convergence_threshold': 1e-4,
        'verbose': True,
        'alpha': 1.0,
        },
    'inverse_compositional': {
        'p_init': __p_init_base__,
        'max_iterations': 200,
        'convergence_threshold': 1e-4,
        'verbose': True,
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
