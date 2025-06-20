from classes import StitchingData, MatchStruct
import numpy as np
from scipy.optimize import least_squares


class Optimizer:
    def __init__(self, data: 'StitchingData'):
        self.data = data
        self.inliers = data.inliers
        self.reper_idx = data.reper_idx
        self.homographies = data.homographies
    
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
                H = H[:-1]  # Remove the last element (scale factor)
                vec[8 * (i - 1): 8 * i] = H

        return vec
    
    def project(self, xy, H):
        point = np.concatenate([xy, np.array([1])])
        new_point = np.dot(H, point)
        new_point /= new_point[2]
        return new_point[:2]

    def reprojection_error(self, X: np.ndarray) -> np.ndarray:
        errors = []
        for inlier in self.inliers:

            Hi = self.vec_to_homography(X, inlier.i)
            Hj = self.vec_to_homography(X, inlier.j)

            first = self.project(inlier.xy_i, Hi)
            second = self.project(inlier.xy_j, Hj)

            errors.append(first[0] - second[0])
            errors.append(first[1] - second[1])

        return np.array(errors)


    def bundle_adjustment(self) -> 'StitchingData':
        """
        Optimize the transformations using bundle adjustment to minimize reprojection error.
        
        Args:
            align_data: AlignData object containing image paths, transformations, reference index,
                        and inliers.
        
        Returns:
            OptimizeData: Data object containing optimized transformations and pivot index.
        """
        n = len(self.homographies)
        vec = self.homography_to_vec(self.homographies)
        init_error = self.reprojection_error(vec)
        
        initial_error = (init_error**2).mean() ** 0.5

        res_lm = least_squares(
            self.reprojection_error, vec, method="lm", xtol=1e-6, ftol=1e-6
        )
        optimized_error = (res_lm.fun**2).mean() ** 0.5
        new_vec = res_lm.x
        
        homographies = [
            self.vec_to_homography(new_vec, i) for i in range(n)
        ]
        
        return StitchingData(self.data.image_set, self.inliers, homographies, self.reper_idx, None)
