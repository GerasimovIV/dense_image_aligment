from enum import Enum
from typing import List, Literal

import cv2
import numpy as np
from PIL import Image


def build_intristic_camera_matrix(intrinsics: List[float]) -> np.ndarray:
    f1, f2, p1, p2 = intrinsics
    return np.array(
        [
            [f1, 0., p1],
            [0., f2, p2],
            [0., 0., 1.],
        ],
        dtype=np.float32
    )


def undistort_image(
    image: np.ndarray,
    intrinsics: List[float],
    distortion_coeffs: List[float],
    distortion_model: Literal['equidistant'] = 'equidistant',
) -> np.ndarray:

    K = build_intristic_camera_matrix(intrinsics)

    undistorted_image: np.array

    if distortion_model == 'equidistant':
        D = np.array(distortion_coeffs, dtype=np.float32)

        undistorted_image = cv2.fisheye.undistortImage(
            distorted=image,
            K=K,
            D=D,
            new_size=image.shape[::-1],
            Knew=K,
        )
    else:
        raise NotImplementedError(f'No implementation for {distortion_model} distortion model')

    return undistorted_image
