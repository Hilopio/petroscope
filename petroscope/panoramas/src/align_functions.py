from classes import StitchingData, Match, TileSet
import numpy as np
import cv2
from logger import logger, log_time


def find_homographies_and_inliers(
    matches: list[Match],
    n: int,
    transformation_type: str,
    confidence_threshold: float,
    min_inliers: int,
    max_inliers: int,
    min_inliers_rate: float,
    reproj_tr: float
) -> tuple[list[list[np.ndarray | None]], list[list[Match]], np.ndarray]:
    """
    Find homographies and inliers between pairs of images based on matching points.

    Args:
        matches (list[Match]): List of match objects containing correspondence points between images.
        n (int): Number of images to process.
        confidence_threshold (float): Minimum confidence score for a match to be considered.
        min_inliers (int): Minimum number of inliers required to accept a homography.
        max_inliers (int): Maximum number of inliers to keep for each pair (top-k by confidence).
        min_inliers_rate (float): Minimum ratio of inliers to total matches to accept a homography.

    Returns:
        tuple: A tuple containing:
            - list[list[np.ndarray | None]]: Matrix of homography matrices (or None if not computed) for each
                image pair.
            - list[list[Match]]: List of lists of inlier matches for each valid image pair.
            - np.ndarray: Matrix of the number of inliers for each image pair.
    """

    inliers: list[Match] = []
    Hs: list[list[np.ndarray | None]] = [[None] * n for _ in range(n)]
    num_inliers: np.ndarray = np.zeros((n, n), dtype=int)

    for matches_ij in matches:
        i = matches_ij.i
        j = matches_ij.j
        conf_mask = matches_ij.conf > confidence_threshold
        xy_i = matches_ij.xy_i[conf_mask]
        xy_j = matches_ij.xy_j[conf_mask]
        conf = matches_ij.conf[conf_mask]

        num_matches_ij = xy_i.shape[0]
        if num_matches_ij < min_inliers:
            continue
        match transformation_type:
            case 'affine':
                M, ransac_mask = cv2.estimateAffine2D(
                    xy_i,
                    xy_j,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=reproj_tr
                )
                if M is None:
                    continue
                H_ij = np.vstack([M, [0, 0, 1]])
            case 'projective':
                H_ij, ransac_mask = cv2.findHomography(
                    xy_i,
                    xy_j,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=reproj_tr,
                )
                if H_ij is None:
                    continue
            case _:
                raise ValueError(f"Unknown transformation type: {transformation_type}")

        num_inliers_ij = ransac_mask.sum()

        if num_inliers_ij < min_inliers:
            continue

        if num_inliers_ij / num_matches_ij < min_inliers_rate:
            continue

        Hs[i][j] = H_ij
        try:
            Hs[j][i] = np.linalg.inv(H_ij)
            Hs[j][i] /= Hs[j][i][2, 2]
        except np.linalg.LinAlgError:
            logger.warning(f"Singular homography matrix for pair ({i}, {j}): {H_ij}. Skipping inverse.")
            continue

        num_inliers[i][j] = num_inliers_ij
        num_inliers[j][i] = num_inliers_ij

        ransac_mask = ransac_mask.squeeze(1).astype(bool)
        xy_i = xy_i[ransac_mask]
        xy_j = xy_j[ransac_mask]
        conf = conf[ransac_mask]

        # Keep top-k by confidence
        if conf.shape[0] > max_inliers:
            topk_indices = np.argpartition(conf, -max_inliers)[-max_inliers:]
            xy_i = xy_i[topk_indices]
            xy_j = xy_j[topk_indices]
            conf = conf[topk_indices]

        inliers_ij = Match(i, j, xy_i, xy_j, conf)
        inliers.append(inliers_ij)

    logger.debug(f"Found {num_inliers.sum() // 2} valid inliers")
    return Hs, inliers, num_inliers


def sequential_alignment(
    Hs: list[list[np.ndarray | None]],
    num_inliers: np.ndarray
) -> tuple[list[np.ndarray | None], list[int], int]:
    """
    Perform sequential alignment of images based on homographies and number of inliers.

    Args:
        Hs (list[list[np.ndarray | None]]): Matrix of homography matrices (or None if not computed) for each image pair.
        num_inliers (np.ndarray): Matrix of the number of inliers for each image pair.

    Returns:
        tuple: A tuple containing:
            - list[np.ndarray | None]: List of transformation matrices for each image.
            - list[int]: List of indices of images in the order they were aligned.
            - int: Index of the reference image used as the starting point for alignment.
    """
    n: int = len(Hs)
    transforms: list[np.ndarray | None] = [None for _ in range(n)]
    query_idx: list[int] = list(range(n))
    target_idx: list[int] = []

    reper_idx = np.argmax(num_inliers.sum(axis=1))

    target_idx.append(reper_idx)
    query_idx.remove(reper_idx)
    transforms[reper_idx] = np.eye(3)

    while query_idx:
        # a = num_inliers[np.ix_(query_idx, target_idx)]
        a = num_inliers[query_idx][:, target_idx]
        curr: int = int(np.argmax(a.sum(axis=1)))
        best_neighb: int = int(np.argmax(a[curr]))

        if Hs[query_idx[curr]][target_idx[best_neighb]] is None:
            break
            # assert num_inliers[query_idx[curr], target_idx[best_neighb]] == 0, (
            #     f"None homography, matches = {num_inliers[query_idx[curr], target_idx[best_neighb]]}"
            # )
            # transforms[query_idx[curr]] = None
            # query_idx.pop(curr)
            # continue

        H: np.ndarray = (
            transforms[target_idx[best_neighb]]
            @ Hs[query_idx[curr]][target_idx[best_neighb]]
        )
        H /= H[2, 2]
        transforms[query_idx[curr]] = H
        target_idx.append(query_idx[curr])
        query_idx.pop(curr)

    if query_idx:
        logger.warning(f"Disconnected components detected: {len(query_idx)} tiles not aligned.")

    return transforms, target_idx, reper_idx


# def recentering_iteration(transforms, img_centers):
#     warped_img_centers = []
#     for center, H in zip(img_centers, transforms):
#         new_center = H @ np.array([center[0], center[1], 1])
#         new_center /= new_center[2]
#         warped_img_centers.append(new_center[:2])
#     warped_img_centers = np.array(warped_img_centers)

#     x_min, x_max = np.min(warped_img_centers[:, 0]), np.max(warped_img_centers[:, 0])
#     y_min, y_max = np.min(warped_img_centers[:, 1]), np.max(warped_img_centers[:, 1])
#     panorama_center = np.array((0.5 * (x_min + x_max), 0.5 * (y_min + y_max)))

#     new_pivot = np.argmin(((warped_img_centers - panorama_center) ** 2).mean(axis=1))

#     inv_pivot_H = np.linalg.inv(transforms[new_pivot])
#     new_transforms = []
#     for H in transforms:
#         new_H = inv_pivot_H @ H
#         new_H /= new_H[2, 2]
#         new_transforms.append(new_H)
#     return new_transforms, new_pivot

def recentering_iteration(transforms, img_centers):
    homographies = np.array(transforms)  # shape: (n, 3, 3)
    centers = np.column_stack((img_centers, np.ones(len(img_centers))))  # shape: (n, 3)

    warped = np.einsum('nij,nj->ni', homographies, centers)  # shape: (n, 3)
    warped /= warped[:, [2]]
    warped_img_centers = warped[:, :2]

    x_min, x_max = np.min(warped_img_centers[:, 0]), np.max(warped_img_centers[:, 0])
    y_min, y_max = np.min(warped_img_centers[:, 1]), np.max(warped_img_centers[:, 1])
    panorama_center = np.array((0.5 * (x_min + x_max), 0.5 * (y_min + y_max)))

    new_pivot = np.argmin(((warped_img_centers - panorama_center) ** 2).mean(axis=1))

    inv_pivot_H = np.linalg.inv(transforms[new_pivot])  # shape: (3, 3)
    new_transforms = np.einsum('ij,njk->nik', inv_pivot_H, homographies)  # shape: (n, 3, 3)
    # new_transforms /= new_transforms[:, [2], [2]]
    new_transforms /= new_transforms[:, 2, 2][:, None, None]

    # new_transforms = [t for t in new_transforms]
    return list(new_transforms), new_pivot


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
        homographies, reper_idx = recentering_iteration(homographies, img_centers)
        if last_reper_idx == reper_idx:
            break
        last_reper_idx = reper_idx

    return homographies, reper_idx


@log_time("Sequential alignment done for", logger)
def matches_alignment(matches_data: StitchingData, transformation_type: str, confidence_tr: float, min_inliers: int,
                      max_inliers: int, min_inliers_rate: float, reproj_tr: float, n_iterations: int
                      ) -> StitchingData:
    """
    Find homographies between images based on matching data and perform alignment.

    Args:
        matches_data (StitchingData): Object containing correspondences between images and image set data.

    Returns:
        StitchingData: Updated data object with transformation matrices, reference index for alignment, and inliers.
    """
    tile_set: TileSet = matches_data.tile_set
    matches: list[Match] = matches_data.matches

    n = len(tile_set.images)
    Hs, inliers, num_inliers = find_homographies_and_inliers(
        matches,
        n,
        transformation_type,
        confidence_tr,
        min_inliers,
        max_inliers,
        min_inliers_rate,
        reproj_tr
    )
    homographies, new_idx_order, reper_idx = sequential_alignment(Hs, num_inliers)

    homographies = [homographies[idx] for idx in new_idx_order]  # if homographies[i] is not None
    tile_set.order = [tile_set.order[idx] for idx in new_idx_order]
    for id, H in zip(tile_set.order, homographies):
        tile_set.images[id].homography = H

    reverse_permute = {v: i for i, v in enumerate(new_idx_order)}
    inliers = [
        Match(
            reverse_permute[inlier.i],
            reverse_permute[inlier.j],
            inlier.xy_i,
            inlier.xy_j,
            inlier.conf
        )
        for inlier in inliers
        if inlier.i in new_idx_order and inlier.j in new_idx_order
    ]
    reper_idx = reverse_permute[reper_idx]

    homographies, reper_idx = recentering(tile_set, n_iterations)

    for id, H in zip(tile_set.order, homographies):
        tile_set.images[id].homography = H

    return StitchingData(
        tile_set=tile_set,
        matches=inliers,
        reper_idx=reper_idx,
        panorama_size=None,
        canvas=None
    )


def find_translation_and_panorama_size(tile_set: TileSet) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Calculate translation matrix and panorama size based on transformed image corners.

    Args:
        tile_set (TileSet): Set of images with their transformation matrices and original sizes.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Translation matrix to shift the panorama to positive coordinates.
            - tuple[int, int]: Tuple representing the panorama size (width, height).
    """
    sizes = np.array([tile_set.images[id].orig_size for id in tile_set.order])  # shape: (n, 2)
    homographies = np.array([tile_set.images[id].homography for id in tile_set.order])  # shape: (n, 3, 3)

    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # shape: (4, 2)
    corners = corners[None, :, :] * sizes[:, None, [0, 1]]  # shape: (n, 4, 2)
    corners = np.concatenate((corners, np.ones((len(sizes), 4, 1))), axis=2)  # shape: (n, 4, 3)

    new_corners = np.einsum('nij,nkj->nki', homographies, corners)  # shape: (n, 4, 3)
    new_corners /= new_corners[:, :, [2]]  # Нормализация по z-координате

    x_coords = new_corners[:, :, 0].ravel()  # shape: (n*4,)
    y_coords = new_corners[:, :, 1].ravel()  # shape: (n*4,)

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    panorama_size = (int(np.ceil(x_max - x_min)), int(np.ceil(y_max - y_min)))
    return T, panorama_size


@log_time("Translation and adding panorama size done for", logger)
def translate_and_add_panorama_size(data: StitchingData) -> StitchingData:
    """
    Finalize alignment by calculating panorama size and applying translation to transformations.

    Args:
        data (StitchingData): Object containing image set data, matches, and homographies.

    Returns:
        StitchingData: Updated data object with final transformations, panorama size, and other alignment data.
    """
    tile_set = data.tile_set
    T, panorama_size = find_translation_and_panorama_size(tile_set)
    for id in tile_set.order:
        img = tile_set.images[id]
        img.homography = T @ img.homography

    return StitchingData(
        tile_set=tile_set,
        matches=data.matches,
        reper_idx=data.reper_idx,
        panorama_size=panorama_size,
        canvas=None
    )
