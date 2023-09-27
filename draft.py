import cv2
import matplotlib.pyplot as plt
import numpy as np

from dense_image_aligment import (
    image_aligment_method,
    read_as_colored,
    read_as_grayscale,
    save_aligment_progress,
    show_data,
)
from dense_image_aligment.transformations import AffineTransformation


def create_simple_gauss(mu, sigma, shape):
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])

    xx, yy = np.meshgrid(x, y, indexing='xy')

    z = np.exp(-( (xx - mu[0])**2 / (sigma[0]**2) +  (yy - mu[1])**2 / (sigma[1]**2)) / 2) / (np.sqrt(sigma[0]**2 + sigma[1]**2) * np.sqrt(2 * np.pi))
    return z


def create_simple_L(shape):
    mask = np.zeros(shape, dtype=np.float32)
    mask[shape[0] // 5 : shape[0] * 4 // 5, shape[1] // 5 : shape[0] * 2 // 5] = 1.
    mask[shape[0] // 2 : shape[0] * 4 // 5, shape[1] // 5 : shape[0] * 4 // 5] = 1.

    return mask


# template = create_simple_gauss([0.5, 0.5], [0.1, 0.1], [100, 100])
# image = create_simple_gauss([0.5, 0.5], [0.1, 0.1], [100, 100])
template = create_simple_L([100, 90])
image = create_simple_L([100, 90])


method, params = image_aligment_method(key='forward_additive')
params['alpha'] = 1.0
params['max_iterations'] = 500
params['convergence_threshold'] = 1e-8
params['p_init'] = np.array(
    [
        [1, 0.0, 0.],
        [0.0, 1., -20.],
    ]
).reshape(-1)

transform = AffineTransformation(params['p_init'])


ps = method(
    image=image,
    template=template,
    coord_transform=transform,
    **params
)

transform.p = ps[-1]

show_data(
    image=image,
    template=template,
    coords_transform=transform
)
