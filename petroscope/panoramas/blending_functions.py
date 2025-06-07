import cv2
import numpy as np

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
