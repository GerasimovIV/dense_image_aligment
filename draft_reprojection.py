
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dense_image_aligment import image_aligment_method, read_as_grayscale, show_data
from dense_image_aligment.transformations.reprojection import ReprojectionTransformation

root_data = Path('./datasets/Multi-FoV/data')
info_intensity_images = pd.read_csv('./datasets/Multi-FoV/info/images.txt', names=['image_id', 'timestamp' , 'path_to_img'], sep=" ")
info_depth_images = pd.read_csv('./datasets/Multi-FoV/info/depthmaps.txt', names=['image_id', 'path_to_img'], sep=" ")

image_id = 196

intensity_fname = info_intensity_images[info_intensity_images['image_id'] == image_id]['path_to_img'].values[0]
image_intensity = read_as_grayscale(str(root_data / intensity_fname))

depth_fname = info_depth_images[info_depth_images['image_id'] == image_id]['path_to_img'].values[0]
image_depth = np.loadtxt(str(root_data / depth_fname)).reshape(image_intensity.shape[:2])

print(intensity_fname)
image_intensity.shape, image_depth.shape


image_id = 200

intensity_fname = info_intensity_images[info_intensity_images['image_id'] == image_id]['path_to_img'].values[0]
image_intensity_template = read_as_grayscale(str(root_data / intensity_fname))


camera_params_path = './datasets/Multi-FoV/info/intrinsics copy.txt'
with open(camera_params_path) as f:
    data = f.read()
    data = data.split('=')[-1]

    K = np.array(eval(data))


method, params = image_aligment_method(key='forward_additive')
params['alpha'] = 1.0
params['max_iterations'] = 500
params['convergence_threshold'] = 0.000001
params['p_init'] = np.array(
    [0., 0., 0., 0., 0., 0.],
    dtype=np.float32
)


transform = ReprojectionTransformation(p_init=params['p_init'], intrinsic=K.reshape(-1))

ps = method(
    image=(image_intensity, image_depth),
    template=image_intensity_template,
    coord_transform=transform,
    **params
)
