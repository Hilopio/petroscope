import numpy as np
import cv2
import maxflow
from logger import logger, log_time

from utils import _warp
from classes import StitchingData


def diff(img1, img2):
    diff = np.sum((img1 - img2) ** 2, axis=2)
    diff = np.sqrt(diff)
    return diff


def grad(img):
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return grad


def GraphCut(img1, img2, only1_mask, only2_mask):
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(img1.shape[:-1])

    sigma = 1.0
    blurred_1 = cv2.GaussianBlur(img1, (0, 0), sigma)
    blurred_2 = cv2.GaussianBlur(img2, (0, 0), sigma)

    grad_1 = grad(blurred_1)
    grad_2 = grad(blurred_2)
    difference = diff(blurred_1, blurred_2)
    grad_difference = diff(grad_1, grad_2)

    alpha = 2
    smooth_map = difference + alpha * grad_difference
    g.add_grid_edges(nodeids, smooth_map, symmetric=True)

    left_inf = nodeids[only1_mask.astype(bool)]
    g.add_grid_tedges(np.array(left_inf), np.inf, 0)

    right_inf = nodeids[only2_mask.astype(bool)]
    g.add_grid_tedges(np.array(right_inf), 0, np.inf)

    g.maxflow()
    lbls_mask = g.get_grid_segments(nodeids)  # True if img1, False if img2
    lbls_mask = np.int_(np.logical_not(lbls_mask))  # 0 if img1, 1 if img2
    # lbls_mask = np.int_(lbls_mask)
    return lbls_mask


def find_overlap_region(mask1, mask2, eps=200):
    h, w = mask1.shape
    overlap_mask = mask1 & mask2
    overlap_idx = np.nonzero(overlap_mask)
    assert overlap_idx[0].size != 0, "нет области пересечения"

    y_min, y_max = np.min(overlap_idx[0]), np.max(overlap_idx[0])
    x_min, x_max = np.min(overlap_idx[1]), np.max(overlap_idx[1])
    Y_MIN, Y_MAX = max(y_min - eps, 0), min(y_max + eps, h),
    X_MIN, X_MAX = max(x_min - eps, 0), min(x_max + eps, w)

    small_window_slice = slice(y_min, y_max + 1), slice(x_min, x_max + 1)
    wide_window_slice = slice(Y_MIN, Y_MAX + 1), slice(X_MIN, X_MAX + 1)
    small_in_wide_slice = slice(y_min-Y_MIN, y_max-Y_MIN + 1), slice(x_min-X_MIN, x_max-X_MIN + 1)

    slices = (
        small_window_slice,
        wide_window_slice,
        small_in_wide_slice
    )
    return slices


def scaled_graph_cut(img1, img2, sure1, sure2, scale=2):
    if not sure2.any():
        return np.ones(img1.shape[:2], dtype='float32')
    orig_size = np.array((img1.shape[1], img1.shape[0]))
    new_size = (orig_size / scale).astype(int)
    smaller_1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LANCZOS4)
    smaller_2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LANCZOS4)
    smaller_sure1 = cv2.resize(sure1.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)
    smaller_sure2 = cv2.resize(sure2.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)

    lbls_mask = GraphCut(smaller_1, smaller_2, smaller_sure1, smaller_sure2)

    lbls_mask = cv2.resize(lbls_mask, orig_size, interpolation=cv2.INTER_NEAREST)
    return lbls_mask.astype('float32')


def labels2seam(lbls_mask, width=7):
    dilated = cv2.dilate(lbls_mask, kernel=np.ones((width, width), dtype=np.uint8))
    seam = dilated - lbls_mask
    return seam


def seam2lane(seam_mask, width=200):
    dilated = cv2.dilate(seam_mask, kernel=np.ones((width, width), dtype=np.uint8))
    return dilated


def coarse_to_fine_optimal_seam(img1, img2, mask1, mask2, small_in_wide_slice,
                                coarse_scale=8, fine_scale=2, lane_width=200):
    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1

    if only2.sum() < 0.001 * only2.size:
        return np.ones(img1.shape[:2], dtype='float32')

    coarse_labels = scaled_graph_cut(img1, img2, only1, only2, scale=coarse_scale)

    coarse_labels = coarse_labels[small_in_wide_slice]
    coarse_seam = labels2seam(coarse_labels, width=3)
    lane = seam2lane(coarse_seam, width=lane_width)

    sure1 = only1
    sure1[small_in_wide_slice] = (coarse_labels - lane).clip(0, 1)
    sure2 = only2
    sure2[small_in_wide_slice] = (1 - coarse_labels - lane).clip(0, 1)

    fine_labels = scaled_graph_cut(img1, img2, sure1, sure2, scale=fine_scale)

    return fine_labels


def find_graphcut_mask(images, transforms, panorama_size, coarse_scale=16, fine_scale=4, lane_width=200):
    n_images = len(images)
    pano, pano_mask = _warp(images[0], transforms[0], panorama_size)
    img_indexes = pano_mask.astype('int8') - np.ones_like(pano_mask, dtype='int8')

    for i in range(1, n_images):
        warped_img, warped_mask = _warp(images[i], transforms[i], panorama_size)

        small_window_slice, wide_window_slice, small_in_wide_slice = find_overlap_region(pano_mask, warped_mask)

        inter_img1 = pano[wide_window_slice]
        inter_mask1 = pano_mask[wide_window_slice]
        inter_img2 = warped_img[wide_window_slice]
        inter_mask2 = warped_mask[wide_window_slice]

        labels = coarse_to_fine_optimal_seam(inter_img1, inter_img2, inter_mask1, inter_mask2,
                                             small_in_wide_slice, coarse_scale=coarse_scale,
                                             fine_scale=fine_scale, lane_width=lane_width)

        # warped_mask[wide_window_slice] = np.where(labels, False, True)
        warped_mask[small_window_slice] = np.where(labels[small_in_wide_slice], False, True)
        pano = np.where(warped_mask[..., np.newaxis], warped_img, pano)
        img_indexes = np.where(warped_mask, i, img_indexes)
        pano_mask = warped_mask | pano_mask

    return img_indexes


def _warp_coarse_to_fine(images, transforms, panorama_size, coarse_scale=16, fine_scale=4, lane_width=200):
    n = len(images)
    pano, pano_mask = _warp(images[0], transforms[0], panorama_size)

    for i in range(1, n):
        warped_img, warped_mask = _warp(images[i], transforms[i], panorama_size)

        small_window_slice, wide_window_slice, small_in_wide_slice = find_overlap_region(pano_mask, warped_mask)

        inter_img1 = pano[wide_window_slice]
        inter_mask1 = pano_mask[wide_window_slice]
        inter_img2 = warped_img[wide_window_slice]
        inter_mask2 = warped_mask[wide_window_slice]

        labels = coarse_to_fine_optimal_seam(inter_img1, inter_img2, inter_mask1, inter_mask2,
                                             small_in_wide_slice, coarse_scale=coarse_scale,
                                             fine_scale=fine_scale, lane_width=lane_width)

        warped_mask[wide_window_slice] = np.where(labels, False, True)
        pano = np.where(warped_mask[..., np.newaxis], warped_img, pano)
        pano_mask = warped_mask | pano_mask

    return pano


@log_time("Graphcut done for", logger)
def apply_graphcut(
        data: StitchingData,
        use_gains: bool = True,
        coarse_scale: int = 16,
        fine_scale: int = 4,
        lane_width: int = 200
) -> StitchingData:
    tile_set = data.tile_set
    n_images = len(tile_set.order)
    panorama_size = data.panorama_size

    images: list[np.ndarray] = []
    homographies: list[np.ndarray] = []
    for id in data.tile_set.order:
        tile = tile_set.images[id]
        if use_gains:
            images.append(tile.image_compensated)
        else:
            images.append(tile.image)
        homographies.append(tile.homography)

    pano, pano_mask = _warp(images[0], homographies[0], panorama_size)
    canvas = pano_mask.astype('int') - np.ones_like(pano_mask, dtype='int')

    for i in range(1, n_images):
        warped_img, warped_mask = _warp(images[i], homographies[i], panorama_size)

        small_window_slice, wide_window_slice, small_in_wide_slice = find_overlap_region(pano_mask, warped_mask)

        inter_img1 = pano[wide_window_slice]
        inter_mask1 = pano_mask[wide_window_slice]
        inter_img2 = warped_img[wide_window_slice]
        inter_mask2 = warped_mask[wide_window_slice]

        labels = coarse_to_fine_optimal_seam(inter_img1, inter_img2, inter_mask1, inter_mask2,
                                             small_in_wide_slice, coarse_scale=coarse_scale,
                                             fine_scale=fine_scale, lane_width=lane_width)
        # 0 is panorama, 1 is new image
        # warped_mask[wide_window_slice] = np.where(labels, False, True)
        warped_mask[small_window_slice] = np.where(labels[small_in_wide_slice] == 0, True, False)
        pano = np.where(warped_mask[..., np.newaxis], warped_img, pano)
        canvas = np.where(warped_mask, i, canvas)
        pano_mask = warped_mask | pano_mask

    data.canvas = canvas
    return data
