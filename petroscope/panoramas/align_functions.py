
from classes import MatchesData, AlignData
import numpy as np
import cv2


def find_translation_and_panorama_size(sizes, transformations):
    """
    Calculate translation matrix and panorama size based on transformed image corners.
    
    Args:
        sizes: List of tuples representing the original sizes (width, height) of images.
        transformations: List of transformation matrices (homographies) for each image.
    
    Returns:
        tuple: A tuple containing:
            - Translation matrix to shift the panorama to positive coordinates.
            - Tuple representing the panorama size (width, height).
    """
    x_coords, y_coords = [], []
    for size, H in zip(sizes, transformations):
        if H is None:
            continue
        corners = np.array([
            [0, 0, 1],
            [0, size[1], 1],
            [size[0], 0, 1],
            [size[0], size[1], 1]
        ])
        new_corners = H @ corners.T

        new_corners /= new_corners[2]
        x_coords += new_corners[0].tolist()
        y_coords += new_corners[1].tolist()

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]])

    panorama_size = (int(np.ceil(x_max - x_min)), int(np.ceil(y_max - y_min)))
    return T, panorama_size

def recentering(transforms, sizes):
    """
    Recenter the transformations to improve panorama alignment.
    
    Args:
        transforms: List of transformation matrices (homographies) for each image.
        sizes: List of original sizes of the images.
    
    Returns:
        tuple: A tuple containing:
            - List of recentered transformation matrices.
            - Index of the new pivot image.
    """
    N = len(transforms)
    assert len(sizes) == N

    img_centers = [ 
        [size[0] / 2, size[1] / 2] for size in sizes
    ]
    warped_img_centers = []
    for center, H in zip(img_centers, transforms):
        new_center = H @ np.array([center[0], center[1], 1])
        new_center /= new_center[2]
        warped_img_centers.append(new_center[:2])
    warped_img_centers = np.array(warped_img_centers)
    assert warped_img_centers.shape == (N, 2)

    x_min, x_max = np.min(warped_img_centers[:, 0]), np.max(warped_img_centers[:, 0])
    y_min, y_max = np.min(warped_img_centers[:, 1]), np.max(warped_img_centers[:, 1])
    panorama_center = np.array((0.5 * (x_min + x_max), 0.5 * (y_min + y_max)))

    new_pivot = np.argmin(((warped_img_centers - panorama_center) ** 2).mean(axis=1))

    inv_pivot_H = np.linalg.inv(transforms[new_pivot])
    new_transforms = []
    for H in transforms:
        new_H = inv_pivot_H @ H
        new_H /= new_H[2, 2]
        new_transforms.append(new_H)
    return new_transforms, new_pivot

from classes import OptimizeData, AlignmentData

def find_homographies(matches_data: 'MatchesData') -> 'AlignData':
    """
    Find homographies between images based on matching data.
    
    Args:
        matches_data: MatchesData object containing correspondences between images
                      and original sizes.
    
    Returns:
        AlignData: Data object containing image paths, transformations (homographies),
                   reference index for alignment, and inliers.
    """    
    diff_corr = matches_data.matches
    orig_sizes = matches_data.orig_sizes
    img_paths = matches_data.img_paths
    n = len(orig_sizes)
    
    inliers = []
    Hs = [[None] * n for _ in range(n)]
    num_matches = np.zeros((n, n), dtype=int)
    
    # Parameters for filtering matches
    confidence_threshold = 0.95
    min_inliers = 5
    max_inliers = 30
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            corrs = diff_corr.pop(0)
            
            if corrs.shape[0] < min_inliers:
                continue
                
            # Filter by confidence
            corrs = corrs[corrs[:, 4] > confidence_threshold]
            if corrs.shape[0] < min_inliers:
                continue
                
            # Run RANSAC to find homography
            H_ij, mask = cv2.findHomography(
                corrs[:, 0:2], corrs[:, 2:4],
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=1.0,
            )
            if H_ij is None or mask.sum() < min_inliers:
                continue
                
            num_matches[i][j] = int(mask.sum())
            num_matches[j][i] = num_matches[i][j]
            
            # Save forward H, and compute inverse for backward
            Hs[i][j] = H_ij
            try:
                Hs[j][i] = np.linalg.inv(H_ij)
                Hs[j][i] /= Hs[j][i][2, 2]
            except np.linalg.LinAlgError:
                assert "Singular homography matrix"
                
            # Store top-k inliers (sorted by confidence)
            inliers_ij = corrs[mask.ravel().astype(bool)]
            topk = inliers_ij[inliers_ij[:, -1].argsort()[::-1]][:max_inliers]
            topk = topk[:, :-1]  # Remove confidence
            inliers += [[i, j, *inl] for inl in topk]
    
    # Initialize the transformations for each image
    transforms = [np.eye(3) for i in range(n)]
    
    queryIdx = [i for i in range(n)]
    targetIdx = []
    outliersIdx = []
    
    # Choose a pivot image randomly
    pivot = np.random.randint(0, n - 1)
    targetIdx.append(pivot)
    queryIdx.remove(pivot)
    
    while queryIdx:
        a = num_matches[queryIdx, :][:, targetIdx]
        curr = np.argmax(a.sum(axis=1))
        best_neighb = np.argmax(a[curr])
        
        if Hs[queryIdx[curr]][targetIdx[best_neighb]] is None:
            assert num_matches[queryIdx[curr], targetIdx[best_neighb]] == 0, \
                f"None homography, matches = {num_matches[queryIdx[curr], targetIdx[best_neighb]]}"
            transforms[queryIdx[curr]] = None
            outliersIdx.append(queryIdx[curr])
            queryIdx.pop(curr)
            continue
            
        H = (
            transforms[targetIdx[best_neighb]]
            @ Hs[queryIdx[curr]][targetIdx[best_neighb]]
        )
        H /= H[2, 2]
        transforms[queryIdx[curr]] = H
        targetIdx.append(queryIdx[curr])
        queryIdx.pop(curr)
    
    assert (set(targetIdx) | set(outliersIdx)) == set(range(n)), "Something wrong in lists"
    outliers_mask = [1 if i in outliersIdx else 0 for i in range(n)]
    idx_shift = np.cumsum(outliers_mask)
    
    real_transforms = [transforms[i] for i in targetIdx]
    real_inliers = []
    for inl in inliers:
        if inl[0] in outliersIdx or inl[1] in outliersIdx:
            continue
        inl[0], inl[1] = inl[0] - idx_shift[inl[0]], inl[1] - idx_shift[inl[1]]
        real_inliers.append(inl)
    pivot -= idx_shift[pivot]
    
    # Recenter the transformations
    sizes = [orig_sizes[i] for i in targetIdx]
    n_recenterings = 25
    for _ in range(n_recenterings):
        real_transforms, new_pivot = recentering(real_transforms, sizes)
        if new_pivot == pivot:
            break
        pivot = new_pivot
    
    # Reordering
    new_img_paths = [img_paths[i] for i in targetIdx]
    targetIdx = [targetIdx[i] - idx_shift[targetIdx[i]] for i in range(len(targetIdx))]
    real_transforms = [real_transforms[i] for i in targetIdx]
    
    # Use the full list of img_paths from input
    return AlignData(new_img_paths, real_transforms, pivot, real_inliers)

def alignment(optimize_data: 'OptimizeData') -> 'AlignmentData':
    """
    Finalize alignment by calculating panorama size and applying translation to transformations.
    
    Args:
        optimize_data: OptimizeData object containing optimized transformations and pivot index.
    
    Returns:
        AlignmentData: Data object containing final transformations, panorama size, and image paths.
    """
    img_paths = optimize_data.img_paths
    transforms = optimize_data.transforms
    pivot = optimize_data.pivot
    T, panorama_size = find_translation_and_panorama_size(img_paths, transforms)
    final_transforms = [T @ H for H in transforms]
    return AlignmentData(img_paths, final_transforms, panorama_size, pivot)
