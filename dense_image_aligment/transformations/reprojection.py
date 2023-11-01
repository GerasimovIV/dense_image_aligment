from tabnanny import verbose
from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm

from .basic_transformation import BaseTransform
from .coords_utils import hom_coords, rotation_matrix
from .differential_utils import G_matrix


class ProjectionTransformation(BaseTransform):
    n: int = 9
    p = np.eye(3, 3, dtype=np.float32).reshape(-1)
    def __init__(self, p_init: Optional[ndarray] = None) -> None:
        if p_init is not None:
            assert p_init.shape == (self.n,), f'Wrong parameters shape, given: {p_init.shape}'
            self.p = p_init

    def apply_transformation_to_coordinates(self, coords: ndarray) -> Tuple[ndarray, ndarray]:

        # """_summary_

        # Args:
        #     coords (ndarray): n x 3, X coordinates in camera Space

        # Returns:
        #     ndarray: n x 2, points coordinates in image coordinates system
        #     ndarray:
        # """
        projection_matrix = self.p.reshape(3, 3)

        coords_new = np.copy(coords)
        coords_new = (projection_matrix @ coords_new.T).T
        coords_new[:, 0] /= coords_new[:, 2]
        coords_new[:, 1] /= coords_new[:, 2]

        h_max = coords_new[:, 0].max()
        h_min = coords_new[:, 0].min()
        w_max = coords_new[:, 1].max()
        w_min = coords_new[:, 1].min()


        h = round(h_max - h_min)
        w = round(w_max - w_min)

        mask = ~np.zeros(coords_new.shape[0], dtype=np.bool_)
        points_om_image = np.zeros((h, w, 2), dtype=np.float32)

        for i, c in tqdm(enumerate(coords_new), desc='Projection Z checking', disable=True):
            x, y, z = c
            xi = round(x)
            yi = round(y)

            current_z, current_i = points_om_image[xi, yi]

            if current_z == 0.:
                points_om_image[xi, yi, 0] = z
                points_om_image[xi, yi, 1] = i
            else:
                if np.linalg.norm(z - current_z) >= 0.03:
                    if current_z <= z:
                        mask[i] = False
                    else:
                        mask[int(current_i)] = False
                        mask[i] = True
                        points_om_image[xi, yi] = np.array([z, i], dtype=np.float32)

        coords_new = coords_new[:, :2]

        return coords_new, mask

    def jacobian_over_input(self, x: np.array, z: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x (np.array): N x 2

        Raises:
            NotImplementedError: _description_

        Returns:
            np.ndarray: N x 2 x 3
        """
        N = x.shape[0]
        J1 = np.zeros((N, 2, 3))

        z_non_zero_mask = ~np.isclose(z, 0.)

        J1[z_non_zero_mask, 0, 0] = 1 / z[z_non_zero_mask]
        J1[z_non_zero_mask, 1, 1] = 1 / z[z_non_zero_mask]
        J1[z_non_zero_mask, 0, 2] = - x[z_non_zero_mask, 0] / z[z_non_zero_mask]**2
        J1[z_non_zero_mask, 1, 2] = - x[z_non_zero_mask, 1] / z[z_non_zero_mask]**2

        projection_matrix = self.p.reshape(3, 3)

        jacobian = np.einsum('Nkl,lp->Nkp', J1, projection_matrix)
        return jacobian


class ProjectionPseudoInvTransformation(BaseTransform):
    n: int = 9
    p = np.eye(3, 3, dtype=np.float32).reshape(-1)
    def __init__(self, p_init: Optional[ndarray] = None) -> None:
        if p_init is not None:
            assert p_init.shape == (self.n,), f'Wrong parameters shape, given: {p_init.shape}'
            self.p = p_init

    def apply_transformation_to_coordinates(self, coords: ndarray, depth: ndarray) -> ndarray:
        """_summary_

        Args:
            coords (ndarray): n x 2,  coordinates in image coord system
            depth (ndarray): n, depth along ray for each pixel

        Returns:
            ndarray: n x 3, coordinates in camera coord system
        """
        projection_matrix = self.p.reshape(3, 3)
        projection_matrix_inv = np.linalg.inv(projection_matrix)


        coords_new = np.copy(coords)
        coords_new = hom_coords(coords_new)

        coords_new = coords_new * depth[..., None]

        coords_new = (projection_matrix_inv @ coords_new.T).T
        return coords_new


    def jacobian(self, x: np.array, p_c: ndarray) -> ndarray:
        raise NotImplementedError


class RT_Transformation(BaseTransform):
    n: int = 6
    p: np.ndarray = np.zeros(6, dtype=np.float32)

    def apply_transformation_to_coordinates(self, coords: ndarray) -> ndarray:
        """_summary_

        Args:
            coords (ndarray): n x 3,  coordinates in camera coord system

        Returns:
            ndarray: n x 3, coordinates in camera coord system
        """
        R = rotation_matrix(self.p[:3])

        RT_matrix = np.zeros((4, 4), dtype=np.float32)
        RT_matrix[3, 3] = 1.
        RT_matrix[:3, :3] = R
        RT_matrix[:3, 3] = self.p[3:]


        coords_new = np.copy(coords)
        coords_new = hom_coords(coords_new)

        coords_new = (RT_matrix @ coords_new.T).T
        coords_new = coords_new[:, :3]

        return coords_new


    def jacobian(self, x: np.array, p_c: ndarray) -> ndarray:
        previous_values = np.copy(self.p)
        self.p = p_c
        transformed_coordinates = self.apply_transformation_to_coordinates(coords=x)
        transformed_coordinates = hom_coords(transformed_coordinates) # N x 4

        jacobian = []

        for i in range(self.n):
            G_i = G_matrix(i)
            jacobian.append(np.einsum('kl,nl->nk', G_i, transformed_coordinates)[:, :3]) # N x 3

        jacobian = np.stack(jacobian, axis=-1) # N x 3 x 6

        self.p = previous_values

        return jacobian # N x 3 x 6


class ReprojectionTransformation(BaseTransform):
    n: 6
    __p__: np.ndarray = np.zeros(6, dtype=np.float32)

    def __init__(self, p_init: np.ndarray, intrinsic: np.ndarray) -> None:
        self.camera_projection = ProjectionTransformation(intrinsic)
        self.camera_projection_inv = ProjectionPseudoInvTransformation(intrinsic)
        self.RT = RT_Transformation(p_init=p_init)
        super().__init__(p_init)

    @property
    def p(self) -> np.ndarray:
        return self.__p__

    @p.setter
    def p(self, v: np.ndarray) -> None:
        self.__p__ = v
        self.RT.p = v


    def apply_transformation_to_coordinates(self, coords: np.ndarray, depth: Optional[np.array] = None) -> np.ndarray:
        # apply inverse projection

        X = self.camera_projection_inv.apply_transformation_to_coordinates(coords=coords, depth=depth)
        X = self.RT.apply_transformation_to_coordinates(coords=X)
        depth = np.copy(X[:, 2:3])
        x, mask = self.camera_projection.apply_transformation_to_coordinates(coords=X)
        x = np.concatenate([x, depth], axis=-1)
        return x, mask

    def apply_transformation(
        self,
        image: Tuple[np.ndarray, np.ndarray],
        shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """return transformed image

        Args:
            image (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            np.array: _description_
        """

        assert isinstance(image, Tuple)

        intensity_image_origin = image[0]
        depth_image_origin = image[1]

        x_coord = np.arange(intensity_image_origin.shape[1], dtype=np.float32) - float(intensity_image_origin.shape[1]) / 2
        y_coord = np.arange(intensity_image_origin.shape[0], dtype=np.float32) - float(intensity_image_origin.shape[0]) / 2
        x_coord, y_coord = np.meshgrid(x_coord, y_coord, indexing='xy')

        image_pixels_coordinates = np.vstack(
            [
                x_coord.reshape(-1),
                y_coord.reshape(-1)
            ]
        ).T # n x 2

        transformed_coordinates, vizibility_mask = self.apply_transformation_to_coordinates(
            image_pixels_coordinates,
            depth_image_origin.reshape(-1)
        )

        image_values = intensity_image_origin.astype(np.float32).reshape(-1)

        inter_func = LinearNDInterpolator(
            points=transformed_coordinates[vizibility_mask, :2],
            values=image_values[vizibility_mask],
            fill_value=0.
        )

        x_coord = np.arange(shape[1], dtype=np.float32) - float(shape[1]) / 2
        y_coord = np.arange(shape[0], dtype=np.float32) - float(shape[0]) / 2

        x_coord, y_coord = np.meshgrid(x_coord, y_coord, indexing='xy')  # 2D grid for interpolation

        transformed_image_values = inter_func(
            x_coord,
            y_coord,
        )

        indexes = np.copy(transformed_coordinates[:, :2])
        indexes[:, 0] += float(intensity_image_origin.shape[1]) / 2
        indexes[:, 1] += float(intensity_image_origin.shape[0]) / 2
        indexes = indexes.round().astype(int)
        indexes = indexes[indexes[:, 0] >= 0]
        indexes = indexes[indexes[:, 0] < shape[1]]
        indexes = indexes[indexes[:, 1] >= 0]
        indexes = indexes[indexes[:, 1] < shape[0]]

        values_mask = np.zeros(shape, dtype=np.bool_)
        values_mask[indexes[:, 1], indexes[:, 0]] = 1


        transformed_image = np.zeros(shape, dtype=transformed_image_values.dtype)
        transformed_image[values_mask] = transformed_image_values.reshape(*shape)[values_mask]

        indexes = np.copy(transformed_coordinates[vizibility_mask, :2])
        indexes[:, 0] += float(intensity_image_origin.shape[1]) / 2
        indexes[:, 1] += float(intensity_image_origin.shape[0]) / 2
        indexes = indexes.round().astype(int)

        depth_values = np.copy(transformed_coordinates[vizibility_mask, 2:3])
        depth_image = np.zeros(shape, dtype=transformed_image_values.dtype)

        depth_values = depth_values[indexes[:, 0] >= 0]
        indexes = indexes[indexes[:, 0] >= 0]

        depth_values = depth_values[indexes[:, 0] < shape[1]]
        indexes = indexes[indexes[:, 0] < shape[1]]

        depth_values = depth_values[indexes[:, 1] >= 0]
        indexes = indexes[indexes[:, 1] >= 0]

        depth_values = depth_values[indexes[:, 1] < shape[0]]
        indexes = indexes[indexes[:, 1] < shape[0]]

        depth_image[indexes[:, 1], indexes[:, 0]] = depth_values[:, 0]

        return transformed_image, depth_image


    def jacobian(self, x: np.array, p_c: ndarray, z: np.ndarray) -> ndarray:
        jacobian_proj_inv = self.camera_projection.jacobian_over_input(x=x, z=z) # N x 2 x 3
        X = self.camera_projection_inv.apply_transformation_to_coordinates(coords=x, depth=z)
        jacobian_RT = self.RT.jacobian(x=X, p_c=p_c) # N x 3 x 6

        jacobian = np.einsum('NxX,NXp->Nxp', jacobian_proj_inv, jacobian_RT)
        return jacobian
