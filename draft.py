import numpy as np

from dense_image_aligment import (
    image_aligment_method,
    read_as_colored,
    read_as_grayscale,
    save_aligment_progress,
    show_data,
)
from dense_image_aligment.transformations.affine import AffineTransformation

image = read_as_grayscale('./media/kanade_image.jpg')
templ = read_as_grayscale('./media/kanade.jpg')

image.shape, templ.shape


method, params = image_aligment_method(key='forward_additive')
params['alpha'] = 1.
params['max_iterations'] = 100
params['p_init'] = np.array(
    [
        [1, 0.1, 40],
        [0.2, 1., 70.],
    ]
).reshape(-1)

affine_transform = AffineTransformation(params['p_init'].reshape(-1))

ps = method(
    image=image,
    template=templ,
    coord_transform=affine_transform,
    **params
)
