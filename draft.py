import matplotlib.pyplot as plt
import numpy as np

from dense_image_aligment import (
    image_aligment_method,
    read_as_colored,
    read_as_grayscale,
    save_aligment_progress,
    show_data,
)
from dense_image_aligment.transformations.translation import TranslationTransformation


def create_simple_gauss(mu, sigma, shape):
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])

    xx, yy = np.meshgrid(x, y, indexing='xy')

    z = np.exp(-( (xx - mu[0])**2 +  (yy - mu[1])**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return z


template = create_simple_gauss([0.5, 0.5], 0.1, [100, 100])
image = create_simple_gauss([0.3, 0.3], 0.1, [100, 100])


method, params = image_aligment_method(key='forward_additive')
params['alpha'] = 1.0
params['max_iterations'] = 100
params['p_init'] = np.array([1., 1.])

affine_transform = TranslationTransformation(params['p_init'])


show_data(
    image=image,
    template=template,
    coords_transform=affine_transform
)
