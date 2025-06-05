import cv2
import numpy as np

from utils import _warp, _warp_img
from graphcut_functions import coarse_to_fine_optimal_seam

def find_overlap_region(mask1, mask2, eps=200):
    h, w = mask1.shape[:2]
    overlap_mask = mask1 & mask2
    overlap_idx = np.nonzero(overlap_mask)
    assert overlap_idx[0].size != 0, "нет области пересечения"

    y_min, y_max = np.min(overlap_idx[0]), np.max(overlap_idx[0])
    x_min, x_max = np.min(overlap_idx[1]), np.max(overlap_idx[1])
    Y_MIN, Y_MAX = max(y_min - eps, 0), min(y_max + eps, h),
    X_MIN, X_MAX = max(x_min - eps, 0), min(x_max + eps, w)

    small_window_slice = slice(y_min, y_max), slice(x_min, x_max)
    wide_window_slice = slice(Y_MIN, Y_MAX), slice(X_MIN, X_MAX)
    small_in_wide_slice = slice(y_min-Y_MIN, y_max-Y_MIN), slice(x_min-X_MIN, x_max-X_MIN)

    slices = (
        small_window_slice,
        wide_window_slice,
        small_in_wide_slice
    )
    return slices


def get_gaussian_level(img, sigmas, k):
    if k == 0:
        return img
    else:
        return cv2.GaussianBlur(img, (0, 0), sigmas[k-1], borderType=cv2.BORDER_REPLICATE)


def get_laplacian_level(img, sigmas, k):
    if k == len(sigmas):
        return get_gaussian_level(img, sigmas, k)
    else:
        return get_gaussian_level(img, sigmas, k) - get_gaussian_level(img, sigmas, k + 1)


def multi_band_blending(images, masks, transforms, panorama_size, levels):
    sigmas = [2.0 ** k for k in range(levels - 1)]
    # sigma0 = 1
    # sigmas = [np.sqrt(2.0 * k + 1) * sigma0 for k in range(levels - 1)]

    w, h = panorama_size
    pano = np.zeros((h, w, 3))
    n_images = len(images)
    for k in range(levels):
        curr_band = np.zeros((h, w, 3))
        curr_weights = np.zeros((h, w))
        for i in range(n_images):
            warped_image = _warp_img(images[i], transforms[i], panorama_size)
            weights = get_gaussian_level(masks[i].astype('float32'), sigmas, k)
            band = get_laplacian_level(warped_image, sigmas, k)
            curr_band += band * weights[..., np.newaxis]
            curr_weights += weights
            del warped_image, weights, band
        pano += curr_band / (curr_weights[..., np.newaxis] + 1e-6)
    return pano


def find_graphcut_mask(images, transforms, panorama_size):
    n = len(images)
    pano, pano_mask = _warp(images[0], transforms[0], panorama_size)
    img_indexes = pano_mask.astype('int8') - np.ones_like(pano_mask, dtype='int8')

    for i in range(1, n):
        warped_img, warped_mask = _warp(images[i], transforms[i], panorama_size)

        small_window_slice, wide_window_slice, small_in_wide_slice = find_overlap_region(pano_mask, warped_mask)

        inter_img1 = pano[wide_window_slice]
        inter_mask1 = pano_mask[wide_window_slice]
        inter_img2 = warped_img[wide_window_slice]
        inter_mask2 = warped_mask[wide_window_slice]

        labels = coarse_to_fine_optimal_seam(inter_img1, inter_img2, inter_mask1, inter_mask2,
                                             small_in_wide_slice, coarse_scale=16, fine_scale=4, lane_width=200)

        warped_mask[wide_window_slice] = np.where(labels, False, True)
        pano = np.where(warped_mask[..., np.newaxis], warped_img, pano)
        img_indexes = np.where(warped_mask, i, img_indexes)
        pano_mask = warped_mask | pano_mask

    return img_indexes