
from classes import StitchingData, Match, TileSet
import numpy as np
import cv2


def find_homographies_and_inliers(
    matches: list[Match],
    n: int,
    confidence_threshold: float,
    min_inliers: int,
    max_inliers: int,
    min_inliers_rate: float
) -> tuple[list[list[np.ndarray]], list[Match], np.ndarray]:

    inliers = []
    Hs = [[None] * n for _ in range(n)]
    num_inliers = np.zeros((n, n), dtype=int)

    matches = list(filter(lambda x: x.conf > confidence_threshold, matches))
    for i in range(n - 1):
        for j in range(i + 1, n):
            matches_ij = list(filter(lambda x: x.i == i and x.j == j, matches))

            if len(matches_ij) < min_inliers:
                continue

            H_ij, mask = cv2.findHomography(
                matches_ij.xy_i,
                matches_ij.xy_j,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=1.0,
            )
            if H_ij is None:
                continue

            num = mask.sum()

            if num < min_inliers:
                continue

            if num / len(matches_ij) < min_inliers_rate:
                continue

            num_inliers[i][j] = num
            num_inliers[j][i] = num

            Hs[i][j] = H_ij
            try:
                Hs[j][i] = np.linalg.inv(H_ij)
                Hs[j][i] /= Hs[j][i][2, 2]
            except np.linalg.LinAlgError:
                assert "Singular homography matrix"

            inliers_ij = matches_ij[mask.ravel().astype(bool)]
            topk_inliers_ij = list(sorted(inliers_ij, key=lambda x: x.conf, reverse=True))[:max_inliers]
            inliers.append(topk_inliers_ij)
    return Hs, inliers, num_inliers


def sequential_alignment(Hs, num_inliers) -> tuple[list[np.ndarray], list[int]]:
    n = len(Hs)
    transforms = [np.eye(3) for _ in range(n)]
    queryIdx = [i for i in range(n)]
    targetIdx = []
    outliersIdx = []

    reper_idx = np.argmax(num_inliers.sum(axis=1))

    targetIdx.append(reper_idx)
    queryIdx.remove(reper_idx)

    while queryIdx:
        a = num_inliers[queryIdx, :][:, targetIdx]
        curr = np.argmax(a.sum(axis=1))
        best_neighb = np.argmax(a[curr])

        if Hs[queryIdx[curr]][targetIdx[best_neighb]] is None:
            assert num_inliers[queryIdx[curr], targetIdx[best_neighb]] == 0, \
                f"None homography, matches = {num_inliers[queryIdx[curr], targetIdx[best_neighb]]}"
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
    return transforms, targetIdx, reper_idx


def recentering_iteration(transforms, img_centers):
    warped_img_centers = []
    for center, H in zip(img_centers, transforms):
        new_center = H @ np.array([center[0], center[1], 1])
        new_center /= new_center[2]
        warped_img_centers.append(new_center[:2])
    warped_img_centers = np.array(warped_img_centers)

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


def recentering(tile_set, n_iterations):
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

    img_centers = []
    homographies = []
    for id in tile_set.order:
        img = tile_set.images[id]
        size = img.orig_size
        img_centers.append([size[0] / 2, size[1] / 2])
        homographies.append(img.homography)

    last_reper_idx = None

    for _ in range(n_iterations):
        new_transforms, reper_idx = recentering_iteration(homographies, img_centers)
        if last_reper_idx == reper_idx:
            break
        last_reper_idx = reper_idx

    return new_transforms, reper_idx


def matches_alignment(matches_data: 'StitchingData') -> 'StitchingData':
    """
    Find homographies between images based on matching data.

    Args:
        matches_data: MatchesData object containing correspondences between images
                      and original sizes.

    Returns:
        AlignData: Data object containing image paths, transformations (homographies),
                   reference index for alignment, and inliers.
    """
    tile_set = matches_data.tile_set
    matches = matches_data.matches

    n = len(tile_set.images)
    confidence_threshold = 0.95
    min_inliers = 5
    max_inliers = 30
    min_inliers_rate = 0.0  # временное решение

    Hs, inliers, num_inliers = find_homographies_and_inliers(
        matches,
        n,
        confidence_threshold,
        min_inliers,
        max_inliers,
        min_inliers_rate
    )

    homographies, new_idx_order, reper_idx = sequential_alignment(Hs, num_inliers)

    outliers_mask = [1 if i not in new_idx_order else 0 for i in range(n)]
    idx_shift = np.cumsum(outliers_mask)

    homographies = [homographies[i] for i in new_idx_order]
    inliers = [
        Match(
            inlier.i - idx_shift[inlier.i],
            inlier.j - idx_shift[inlier.j],
            inlier.xy_i,
            inlier.xy_j,
            inlier.conf
        )
        for inlier in inliers
        if inlier.i in new_idx_order and inlier.j in new_idx_order
    ]
    reper_idx -= idx_shift[reper_idx]
    tile_set.order = [tile_set.order[new_idx] for new_idx in new_idx_order]

    n_iterations = 25
    homographies, reper_idx = recentering(tile_set, n_iterations)

    return StitchingData(tile_set, inliers, reper_idx, None)


def find_translation_and_panorama_size(tile_set: TileSet):
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
    sizes = []
    homographies = []
    for id in tile_set.order:
        img = tile_set.images[id]
        sizes.append(img.orig_size)
        homographies.append(img.homography)

    x_coords, y_coords = [], []
    for (w, h), H in zip(sizes, homographies):
        assert H is not None
        corners = np.array([
            [0, 0, 1],
            [0, h, 1],
            [w, 0, 1],
            [w, h, 1]
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


def translate_and_add_panorama_size(data: 'StitchingData') -> 'StitchingData':
    """
    Finalize alignment by calculating panorama size and applying translation to transformations.

    Args:
        optimize_data: OptimizeData object containing optimized transformations and pivot index.

    Returns:
        AlignmentData: Data object containing final transformations, panorama size, and image paths.
    """
    tile_set = data.tile_set
    T, panorama_size = find_translation_and_panorama_size(tile_set)
    for id in tile_set.order:
        img = tile_set.images[id]
        img.homography = T @ img.homography

    return StitchingData(tile_set, data.matches, data.reper_id, panorama_size)
