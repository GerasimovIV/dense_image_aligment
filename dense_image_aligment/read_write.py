from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def read_as_grayscale(p: str) -> np.ndarray:
    """
    read_as_grayscale
    read image by its path
    and transform to grayscale

    Parameters
    ----------
    p : str
        path to image

    Returns
    -------
    np.ndarray
        image (n x m) with type float32
        image has values in [0, 1]
    """
    image = np.array(
        cv2.imread(p, cv2.IMREAD_GRAYSCALE),
        dtype=np.float32
    )
    image /= 255.0

    return image

def read_as_colored(p: str) -> np.ndarray:
    """
    read_as_colored
    read image by its path

    Parameters
    ----------
    p : str
        path to image

    Returns
    -------
    np.ndarray
        image (n x m x 3) with type float32
        image has values in [0, 1]
    """
    image = np.array(
        cv2.imread(p),
        dtype=np.float32
    )
    image /= 255.0

    return image



def show_data(
    image: np.ndarray,
    template: np.ndarray,
    mat_affine: np.ndarray
) -> None:
    """
    show_data helper function for visualization

    Parameters
    ----------
    image : np.ndarray
        input image (warped)
    template : np.ndarray
        template image
    mat_affine : np.ndarray
        matrix of affine transformation
        that was used for input image
    """
    p = mat_affine
    image_wrapped = cv2.warpAffine(
        image,
        p,
        template.shape[::-1]
    )

    image_with_template = np.mean(
        np.concatenate(
            [
                template[None],
                image_wrapped[None],
            ],
            axis=0
        ),
        axis=0
    )

    h, w = image.shape

    a = np.array([[0, 0, w, w],
                  [0, h, h, 0],
                  [1, 1, 1, 1]])
    b = p.dot(a)

    fig, axs = plt.subplots()
    axs.matshow(image_with_template, cmap=plt.cm.gray)
    axs.plot(b[0, [0, 1, 2, 3, 0]], b[1, [0, 1, 2, 3, 0]], '-or', linewidth=1)


def save_aligment_progress(
    filename: str,
    image: np.ndarray,
    template: np.ndarray,
    mat_affine_list: List[np.ndarray],
    duration: float = 200
) -> None:
    """
    save_aligment_progress
    function for creating .gif
    with propagationg steps during aligment

    Parameters
    ----------
    filename : str
        path to ".gif" file, where to save
    image : np.ndarray
        source image, which is not changed during aligment
    template : np.ndarray
        image that should be matched with "image"
    mat_affine_list : List[np.ndarray]
        list of affine transformations (their matrixes 2x3)
    duration : float, optional
        time in ms between frames in .gif file, by default 200
    """

    h, w = image.shape

    process_bar = tqdm(mat_affine_list)

    fig, axs = plt.subplots()
    images = []

    for p in process_bar:
        fig.tight_layout()

        a = np.array([[0, 0, w, w],
                  [0, h, h, 0],
                  [1, 1, 1, 1]])
        b = p.dot(a)

        image_wrapped = cv2.warpAffine(
            image,
            p,
            template.shape[::-1]
        )

        image_with_template = np.mean(
            np.concatenate(
                [
                    template[None],
                    image_wrapped[None],
                ],
                axis=0
            ),
            axis=0)

        axs.matshow(image_with_template, cmap=plt.cm.gray)
        axs.plot(b[0, [0, 1, 2, 3, 0]], b[1, [0, 1, 2, 3, 0]], '-or')

        img_pill = Image.new('RGB', fig.canvas.get_width_height())
        fig.canvas.draw()
        img_pill.paste(
            Image.frombytes(
                'RGB',
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb()
            ),
            (0,0)
        )

        images.append(img_pill)

        axs.cla()

    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
