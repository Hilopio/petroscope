import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple


def vec_to_homography(vec: np.ndarray, i: int, pivot: int) -> np.ndarray:
    # If the index is the pivot, return the identity matrix
    if i == pivot:
        return np.eye(3)
    # Adjust index if it is greater than pivot
    elif i > pivot:
        i -= 1
    # Extract the 3x3 homography matrix from the vector
    H = vec[8 * i: 8 * (i + 1)]
    assert H.size == 8, f"Invalid vector size: i = {i}"
    H = np.array([[H[0], H[1], H[2]], [H[3], H[4], H[5]], [H[6], H[7], 1]])
    return H


def homography_to_vec(Hs: List[np.ndarray], pivot: int) -> List[float]:
    n = len(Hs)
    vec = np.empty(8 * (n - 1))
    for i in range(n):
        if i == pivot:
            # Skip the pivot image
            continue
        elif i < pivot:
            # The homography matrix is placed at the position of the image
            H = Hs[i].reshape(-1)
            H = H[:-1]  # Remove the last element (scale factor)
            vec[8 * i: 8 * (i + 1)] = H
        else:
            # The homography matrix is placed at the position of the image
            # minus one (since the pivot image is skipped)
            H = Hs[i].reshape(-1)
            H = H[:-1]  # Remove the last element (scale factor)
            vec[8 * (i - 1): 8 * i] = H
    return vec


def dist(X: List[float], inliers: List[np.ndarray], pivot: int) -> np.ndarray:
    output = []  # Initialize the output list to store distances
    for i, j, x, y, xx, yy in inliers:
        # Get the homography matrices for images i and j
        Hi = vec_to_homography(X, i, pivot)
        Hj = vec_to_homography(X, j, pivot)

        # Transform the coordinates using the homography matrices
        first = np.dot(Hi, [x, y, 1])
        first /= first[2]  # Normalize to get the final coordinates
        second = np.dot(Hj, [xx, yy, 1])
        second /= second[2]  # Normalize to get the final coordinates
        output.append(first[0] - second[0])
        output.append(first[1] - second[1])

    return np.array(output)


def optimize(
    Hs: List[np.ndarray],
    inliers: List[np.ndarray],
    pivot: int,
) -> Tuple[List[np.ndarray], float, float]:

    n = len(Hs)
    vec = homography_to_vec(Hs, pivot)
    norm = dist(vec, inliers, pivot)

    init_error = (norm**2).mean() ** 0.5
    res_lm = least_squares(
        dist, vec, method="lm", xtol=1e-6, ftol=1e-6, args=(inliers, pivot)
    )
    optim_error = (res_lm.fun**2).mean() ** 0.5
    new_vec = res_lm.x

    final_transforms = []
    for i in range(n):
        final_transforms.append(vec_to_homography(new_vec, i, pivot))
    return final_transforms, init_error, optim_error
