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

def masked_blur(img, mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    img_sum = cv2.filter2D(img * mask, -1, kernel)
    weight = cv2.filter2D(mask.astype(np.float32), -1, kernel)

    with np.errstate(divide='ignore', invalid='ignore'):
        blurred = np.where(weight > 0, img_sum / weight, 0)

    updated_mask = (weight > 0).astype('uint8')
    result = np.where(mask == 1, img, blurred)

    return result, updated_mask

def warpPerspectiveBlurBorder(img, H, panorama_size, sigma):
    kernel_size = 11
    window_size = 3 * int(sigma)
    num_blurs = window_size // (kernel_size // 2)

    warped_img = cv2.warpPerspective(
        img,
        H,
        panorama_size,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    img_mask = np.ones_like(img, dtype='float32')
    warped_img_mask = cv2.warpPerspective(
        img_mask,
        H,
        panorama_size,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    warped_img_mask = (warped_img_mask == 1)

    y, x, c = np.where(warped_img_mask)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_min = max(0, x_min - window_size)
    x_max = min(panorama_size[0], x_max + window_size)
    y_min = max(0, y_min - window_size)
    y_max = min(panorama_size[1], y_max + window_size)

    blurred = warped_img[y_min:y_max+1, x_min:x_max+1]
    blurred_mask = warped_img_mask[y_min:y_max+1, x_min:x_max+1].astype('uint8')
    for _ in range(num_blurs):
        blurred, blurred_mask = masked_blur(blurred, blurred_mask, kernel_size=11) 
    gaussian_blurred_border = cv2.GaussianBlur(blurred, (0, 0), 5)
    bbox = np.where(warped_img_mask[y_min:y_max+1, x_min:x_max+1], warped_img[y_min:y_max+1, x_min:x_max+1], gaussian_blurred_border)

    final = warped_img
    final[y_min:y_max+1, x_min:x_max+1] = bbox
    return final


def multi_band_blending(images, masks, transforms, panorama_size, levels):
    sigmas = [2.0 ** k for k in range(levels - 1)]
    # sigma0 = 1
    # sigmas = [np.sqrt(2.0 * k + 1) * sigma0 for k in range(levels - 1)]

    warped_images = [
        # warped_image = _warp_img(images[i], transforms[i], panorama_size)
        warpPerspectiveBlurBorder(images[i], transforms[i], panorama_size, sigmas[-1])
        for i in range(len(images))
    ]
    w, h = panorama_size
    pano = np.zeros((h, w, 3))
    n_images = len(images)
    for k in range(levels):
        curr_band = np.zeros((h, w, 3))
        curr_weights = np.zeros((h, w))
        for i in range(n_images):
            # warped_image = _warp_img(images[i], transforms[i], panorama_size)
            # warped_image = warpPerspectiveBlurBorder(images[i], transforms[i], panorama_size, sigmas[-1])
            
            weights = get_gaussian_level(masks[i].astype('float32'), sigmas, k)
            band = get_laplacian_level(warped_images[i], sigmas, k)
            curr_band += band * weights[..., np.newaxis]
            curr_weights += weights
            del weights, band
        pano += curr_band / (curr_weights[..., np.newaxis] + 1e-6)
    
    pano_mask = np.zeros((h, w))
    for mask in masks:
        pano_mask = pano_mask + mask.astype('float32')
    pano_mask = np.where(pano_mask > 0, 1, 0)

    return pano * pano_mask[..., np.newaxis]


def find_graphcut_mask(images, transforms, panorama_size, coarse_scale=16, fine_scale=4, lane_width=200):
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
                                             small_in_wide_slice, coarse_scale=coarse_scale,
                                             fine_scale=fine_scale, lane_width=lane_width)

        # warped_mask[wide_window_slice] = np.where(labels, False, True)
        warped_mask[small_window_slice] = np.where(labels[small_in_wide_slice], False, True)
        pano = np.where(warped_mask[..., np.newaxis], warped_img, pano)
        img_indexes = np.where(warped_mask, i, img_indexes)
        pano_mask = warped_mask | pano_mask

    return img_indexes