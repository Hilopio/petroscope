from classes import AlignData, OptimizeData
import numpy as np
from scipy.optimize import least_squares

def vec_to_homography(vec: np.ndarray, i: int, reference_idx: int) -> np.ndarray:
    """
    Convert a vector to a homography matrix for a given image index.
    
    Args:
        vec: Vector containing homography parameters.
        i: Index of the image.
        reference_idx: Index of the reference image.
    
    Returns:
        np.ndarray: 3x3 homography matrix.
    """
    # If the index is the reference, return the identity matrix
    if i == reference_idx:
        return np.eye(3)
    # Adjust index if it is greater than reference
    elif i > reference_idx:
        i -= 1
    # Extract the 3x3 homography matrix from the vector
    H = vec[8 * i: 8 * (i + 1)]
    assert H.size == 8, f"Invalid vector size: i = {i}"
    H = np.array([[H[0], H[1], H[2]], [H[3], H[4], H[5]], [H[6], H[7], 1]])
    return H

def homography_to_vec(Hs: list[np.ndarray], reference_idx: int) -> list[float]:
    """
    Convert a list of homography matrices to a vector.
    
    Args:
        Hs: List of homography matrices.
        reference_idx: Index of the reference image.
    
    Returns:
        list[float]: Vector containing homography parameters.
    """
    n = len(Hs)
    vec = np.empty(8 * (n - 1))
    for i in range(n):
        if i == reference_idx:
            # Skip the reference image
            continue
        elif i < reference_idx:
            # The homography matrix is placed at the position of the image
            H = Hs[i].reshape(-1)
            H = H[:-1]  # Remove the last element (scale factor)
            vec[8 * i: 8 * (i + 1)] = H
        else:
            # The homography matrix is placed at the position of the image
            # minus one (since the reference image is skipped)
            H = Hs[i].reshape(-1)
            H = H[:-1]  # Remove the last element (scale factor)
            vec[8 * (i - 1): 8 * i] = H
    return vec

def dist(X: list[float], inliers: list[list], reference_idx: int) -> np.ndarray:
    """
    Calculate the reprojection error for inlier points using homography matrices.
    
    Args:
        X: Vector containing homography parameters.
        inliers: List of inlier points with image indices and coordinates.
        reference_idx: Index of the reference image.
    
    Returns:
        np.ndarray: Array of reprojection errors.
    """
    output = []  # Initialize the output list to store distances
    for i, j, x, y, xx, yy in inliers:
        # Get the homography matrices for images i and j
        Hi = vec_to_homography(X, i, reference_idx)
        Hj = vec_to_homography(X, j, reference_idx)

        # Transform the coordinates using the homography matrices
        first = np.dot(Hi, [x, y, 1])
        first /= first[2]  # Normalize to get the final coordinates
        second = np.dot(Hj, [xx, yy, 1])
        second /= second[2]  # Normalize to get the final coordinates
        output.append(first[0] - second[0])
        output.append(first[1] - second[1])

    return np.array(output)

def bundle_adjustment(align_data: 'AlignData') -> 'OptimizeData':
    """
    Optimize the transformations using bundle adjustment to minimize reprojection error.
    
    Args:
        align_data: AlignData object containing image paths, transformations, reference index,
                    and inliers.
    
    Returns:
        OptimizeData: Data object containing optimized transformations and pivot index.
    """
    Hs = align_data.transforms
    inliers = align_data.inliers
    reference_idx = align_data.reference_idx
    img_paths = align_data.img_paths
    
    n = len(Hs)
    vec = homography_to_vec(Hs, reference_idx)
    norm = dist(vec, inliers, reference_idx)
    
    init_error = (norm**2).mean() ** 0.5
    res_lm = least_squares(
        dist, vec, method="lm", xtol=1e-6, ftol=1e-6, args=(inliers, reference_idx)
    )
    optim_error = (res_lm.fun**2).mean() ** 0.5
    new_vec = res_lm.x
    
    final_transforms = []
    for i in range(n):
        final_transforms.append(vec_to_homography(new_vec, i, reference_idx))
    
    return OptimizeData(img_paths, final_transforms, reference_idx)
