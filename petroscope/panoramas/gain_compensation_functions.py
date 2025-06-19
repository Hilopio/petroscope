import numpy as np
from classes import AlignmentData, GaincompData

def find_mean_color(images: list[np.ndarray]) -> np.ndarray:
    """
    Calculate the mean color value (RGB) across all images.
    
    Args:
        images: List of image arrays with shape (height, width, 3).
    
    Returns:
        np.ndarray: Mean color value as an array of shape (3,) representing (R, G, B).
    """
    mean_color = np.zeros(3)
    for img in images:
        mean_color += np.mean(img, axis=(0, 1))
    mean_color /= len(images)
    return mean_color

def compensate_mean_color(img: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Adjust the image color by a scaling factor to compensate for mean color differences.
    
    Args:
        img: Input image array with shape (height, width, 3).
        scale: Scaling factor for color compensation as an array of shape (3,) for (R, G, B).
    
    Returns:
        np.ndarray: Color-adjusted image array.
    """
    return img * scale[None, None, :]

def gain_compensation(images: list[np.ndarray], transforms: list[np.ndarray], panorama_size: tuple) -> list[np.ndarray]:
    """
    Perform gain compensation on images to balance color and brightness.
    
    Args:
        images: List of input image arrays.
        transforms: List of transformation matrices for each image.
        panorama_size: Tuple representing the size of the panorama (width, height).
    
    Returns:
        list[np.ndarray]: List of gain-compensated image arrays.
    """
    # Placeholder for gain compensation logic
    # This function should implement the actual gain compensation algorithm
    return images

def apply_gain_compensation(alignment_data: 'AlignmentData') -> 'GaincompData':
    """
    Apply gain compensation to aligned images.
    
    Args:
        alignment_data: AlignmentData object containing aligned image information.
    
    Returns:
        GaincompData: Data object containing gain-compensated image data.
    """
    img_paths = alignment_data.img_paths
    transforms = alignment_data.transforms
    panorama_size = alignment_data.panorama_size
    
    images = []
    for path in img_paths:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255
        images.append(img)
    
    mean_before = find_mean_color(images)
    images = gain_compensation(images, transforms, panorama_size)
    scale = mean_before / (find_mean_color(images) + 1e-6)
    images = [compensate_mean_color(img, scale).astype('float32') for img in images]
    
    return GaincompData(images, alignment_data.transforms, alignment_data.panorama_size)
