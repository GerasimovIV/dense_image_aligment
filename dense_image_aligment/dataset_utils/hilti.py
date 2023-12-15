from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def read_camera_params(path: str, cam_id: int = 0) -> Dict[str, List[float] | float | str]:

    assert Path(path).exists() and Path(path).is_file()

    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    camera_key = f'cam{cam_id}'
    camera_params= data.get(camera_key)

    intrinsics = camera_params.get('intrinsics')
    distortion_model = camera_params.get('distortion_model')
    distortion_coeffs = camera_params.get('distortion_coeffs')

    return {
        'intrinsics': intrinsics,
        'distortion_model': distortion_model,
        'distortion_coeffs': distortion_coeffs
    }
