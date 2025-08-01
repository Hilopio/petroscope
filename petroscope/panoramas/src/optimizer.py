from classes import StitchingData
import numpy as np
from scipy.optimize import least_squares
from logger import logger, log_time
from abc import ABC, abstractmethod


class OptimizerBase(ABC):

    def __init__(self, data: StitchingData):
        self.data = data
        self.inliers = data.matches
        self.reper_idx = data.reper_idx
        self.homographies = []
        for id in data.tile_set.order:
            img = data.tile_set.images[id]
            self.homographies.append(img.homography)

    @abstractmethod
    def vec_to_homography(self, vec: np.ndarray, i: int) -> np.ndarray:
        pass

    @abstractmethod
    def homography_to_vec(self, Hs: list[np.ndarray]) -> list[float]:
        pass

    @abstractmethod
    def project(self, xy, H):
        pass

    def reprojection_error(self, X: np.ndarray) -> np.ndarray:
        all_errors = []
        for inlier in self.inliers:

            Hi = self.vec_to_homography(X, inlier.i)
            Hj = self.vec_to_homography(X, inlier.j)

            first = self.project(inlier.xy_i, Hi)
            second = self.project(inlier.xy_j, Hj)

            diff = (first - second)
            reproj_error = np.sum(diff ** 2, axis=1) ** 0.5
            all_errors.append(reproj_error)

        return np.concatenate(all_errors)

    @log_time("Bundle adjustment done for", logger)
    def bundle_adjustment(self) -> StitchingData:
        """
        Optimize the transformations using bundle adjustment to minimize reprojection error.

        Args:
            align_data: AlignData object containing image paths, transformations, reference index,
                        and inliers.

        Returns:
            OptimizeData: Data object containing optimized transformations and pivot index.
        """
        vec = self.homography_to_vec(self.homographies)

        init_error = self.reprojection_error(vec)
        initial_error = (init_error**2).mean() ** 0.5
        logger.debug(f"Initial error: {initial_error}")

        res_lm = least_squares(
            self.reprojection_error, vec, method="lm", xtol=1e-6, ftol=1e-6
        )

        optimized_error = (res_lm.fun**2).mean() ** 0.5
        logger.debug(f"Optimized error: {optimized_error}")

        new_vec = res_lm.x
        for i, id in enumerate(self.data.tile_set.order):
            img = self.data.tile_set.images[id]
            img.homography = self.vec_to_homography(new_vec, i)

        return self.data


class ProjectiveOptimizer(OptimizerBase):

    def __init__(self, data: StitchingData):
        super().__init__(data)

    def vec_to_homography(self, vec: np.ndarray, i: int) -> np.ndarray:
        if i == self.reper_idx:
            return np.eye(3)

        elif i > self.reper_idx:
            i -= 1

        H = vec[8 * i: 8 * (i + 1)]
        assert H.size == 8, f"Invalid vector size: i = {i}"
        H = np.array([[H[0], H[1], H[2]], [H[3], H[4], H[5]], [H[6], H[7], 1]])
        return H

    def homography_to_vec(self, Hs: list[np.ndarray]) -> list[float]:
        n = len(Hs)
        vec = np.empty(8 * (n - 1))
        for i in range(n):

            if i == self.reper_idx:
                continue

            elif i < self.reper_idx:
                H = Hs[i].reshape(-1)
                H = H[:-1]
                vec[8 * i: 8 * (i + 1)] = H

            else:
                H = Hs[i].reshape(-1)
                H = H[:-1]
                vec[8 * (i - 1): 8 * i] = H

        return vec

    def project(self, xy, H):
        n_matches = xy.shape[0]  # xy is a n_matches x 2 array
        project_points = np.concatenate([xy, np.ones((n_matches, 1))], axis=1)

        assert project_points.shape == (n_matches, 3)

        new_points = H @ project_points.T
        new_points /= new_points[2]

        return new_points[:2, :].T


class AffineOptimizer(OptimizerBase):

    def __init__(self, data: StitchingData):
        super().__init__(data)

    def vec_to_homography(self, vec: np.ndarray, i: int) -> np.ndarray:
        if i == self.reper_idx:
            return np.eye(3)

        elif i > self.reper_idx:
            i -= 1

        H = vec[6 * i: 6 * (i + 1)]
        assert H.size == 6, f"Invalid vector size: i = {i}"
        H = np.array([[H[0], H[1], H[2]], [H[3], H[4], H[5]], [0, 0, 1]])
        return H

    def homography_to_vec(self, Hs: list[np.ndarray]) -> list[float]:
        n = len(Hs)
        vec = np.empty(6 * (n - 1))
        for i in range(n):

            if i == self.reper_idx:
                continue

            elif i < self.reper_idx:
                H = Hs[i].reshape(-1)
                H = H[:-3]
                vec[6 * i: 6 * (i + 1)] = H

            else:
                H = Hs[i].reshape(-1)
                H = H[:-3]
                vec[6 * (i - 1): 6 * i] = H

        return vec

    def project(self, xy, H):
        n_matches = xy.shape[0]  # xy is a n_matches x 2 array
        project_points = np.concatenate([xy, np.ones((n_matches, 1))], axis=1)
        assert project_points.shape == (n_matches, 3)
        new_points = H @ project_points.T
        return new_points[:2, :].T


def Optimizer(transformation_type, data):
    match transformation_type:
        case "affine":
            return AffineOptimizer(data)
        case "projective":
            return ProjectiveOptimizer(data)
        case _:
            raise ValueError("Unknown transformation type")
