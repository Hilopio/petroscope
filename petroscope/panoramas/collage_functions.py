import cv2
import numpy as np
from PIL import Image
from classes import OptimizeData, PanoramaData

borderValue = 0.0

def make_collage(optimize_data: 'OptimizeData') -> 'PanoramaData':
    """
    Create a collage-style panorama from optimized image data.
    
    Args:
        optimize_data: OptimizeData object containing image paths and optimized transformations.
    
    Returns:
        PanoramaData: Data object representing the stitched collage panorama.
    """
    images = []
    for path in optimize_data.img_paths:
        img = Image.open(path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255
        images.append(img)
    
    transforms = optimize_data.transforms
    panorama_size = (transforms[0].shape[0], transforms[0].shape[1])  # Placeholder for actual size calculation
    w, h = panorama_size
    panorama = np.full(shape=(h, w, 3), fill_value=borderValue, dtype='float32')

    for image, H in zip(images, transforms):
        cv2.warpPerspective(
            image,
            H,
            panorama_size,
            dst=panorama,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    return PanoramaData(panorama)
