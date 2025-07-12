import cv2
import numpy as np
from classes import StitchingData, Panorama

borderValue = 0.0


def make_collage(data: StitchingData, use_gains=False) -> Panorama:
    """
    Create a collage-style panorama from optimized image data.

    Args:
        optimize_data: OptimizeData object containing image paths and optimized transformations.

    Returns:
        PanoramaData: Data object representing the stitched collage panorama.
    """
    images = []
    homographies = []
    for id in data.tile_set.order:
        img = data.tile_set.images[id]
        if use_gains:
            images.append(img.image_compensated)
        else:
            images.append(img.image)
        homographies.append(img.homography)

    w, h = data.panorama_size
    panorama = np.full(shape=(h, w, 3), fill_value=borderValue, dtype='float32')

    for image, H in zip(images, homographies):
        cv2.warpPerspective(
            image,
            H,
            (w, h),
            dst=panorama,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    return Panorama(panorama, None)


def make_mosaic(data: StitchingData, use_gains=False) -> Panorama:
    """
    Create a collage-style panorama from optimized image data.

    Args:
        optimize_data: OptimizeData object containing image paths and optimized transformations.

    Returns:
        PanoramaData: Data object representing the stitched collage panorama.
    """
    canvas = data.canvas
    images = []
    homographies = []
    for id in data.tile_set.order:
        img = data.tile_set.images[id]
        if use_gains:
            images.append(img.image_compensated)
        else:
            images.append(img.image)
        homographies.append(img.homography)
    w, h = data.panorama_size
    panorama = np.full(shape=(h, w, 3), fill_value=borderValue, dtype='float32')

    for i, (image, H) in enumerate(zip(images, homographies)):
        warped = cv2.warpPerspective(
            image,
            H,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
            borderValue=borderValue
        )
        panorama = np.where((canvas == i)[..., np.newaxis], warped, panorama)

    return Panorama(panorama, canvas)
