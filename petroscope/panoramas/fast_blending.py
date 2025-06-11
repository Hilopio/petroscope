import numpy as np
import cv2
from typing import List, Tuple


def multi_band_blending(images: List[np.ndarray], 
                       masks: List[np.ndarray], 
                       transforms: List[np.ndarray], 
                       panorama_size: Tuple[int, int], 
                       n_levels: int = 5) -> np.ndarray:
    """
    Multi-band blending for panorama stitching using Gaussian and Laplacian pyramids.
    
    Args:
        images: List of (H, W, 3) float32 numpy arrays
        masks: List of binary masks indicating which pixels belong to each image in final panorama
        transforms: List of 3x3 homography matrices from image to panorama coordinates
        panorama_size: (W, H) tuple of panorama dimensions
        n_levels: Number of pyramid levels for blending
    
    Returns:
        Blended panorama as (H, W, 3) float32 array
    """
    W, H = panorama_size
    
    # Create sigma values: 1, 2, 4, ..., 2^(n_levels-1)
    sigmas = [2**i for i in range(n_levels)]
    border_size = 5 * sigmas[-1]
    
    # Initialize output panorama
    panorama = np.zeros((H, W, 3), dtype=np.float32)
    
    # Process each image
    warped_images = []
    warped_masks = []
    warped_original_masks = []  # Store original masks for final cropping
    bounding_boxes = []
    
    for i, (img, mask, transform) in enumerate(zip(images, masks, transforms)):
        # Extend image with border for better blending
        extended_img = cv2.copyMakeBorder(
            img, border_size, border_size, border_size, border_size,
            cv2.BORDER_REFLECT_101
        )
        
        # Create mask for original image area (without border)
        h, w = img.shape[:2]
        original_mask = np.zeros((h + 2*border_size, w + 2*border_size), dtype=np.float32)
        original_mask[border_size:border_size+h, border_size:border_size+w] = 1.0
        
        # Update transform to account for border
        border_transform = np.array([
            [1, 0, -border_size],
            [0, 1, -border_size], 
            [0, 0, 1]
        ], dtype=np.float32)
        adjusted_transform = transform @ border_transform
        
        # Warp extended image
        warped_img = cv2.warpPerspective(
            extended_img, adjusted_transform, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        
        # Warp original area mask with linear interpolation
        warped_original_mask = cv2.warpPerspective(
            original_mask, adjusted_transform, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        
        # Keep only pixels that are exactly 1.0 to avoid black border leakage
        warped_original_mask = (warped_original_mask == 1.0).astype(np.float32)
        
        # Find bounding box based on warped image content (not just original mask)
        coords = np.where(np.any(warped_img > 0, axis=2))
        if len(coords[0]) == 0:
            continue
            
        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1
        
        # Store cropped warped image, GraphCut mask, and original mask with bounding box info
        warped_images.append(warped_img[y_min:y_max, x_min:x_max])
        warped_masks.append(mask[y_min:y_max, x_min:x_max].astype(np.float32))
        warped_original_masks.append(warped_original_mask)  # Store full-size original mask
        bounding_boxes.append((x_min, y_min, x_max, y_max))
    
    if not warped_images:
        return panorama
    
    # Build Gaussian and Laplacian pyramids for each image
    gaussian_pyramids = []
    laplacian_pyramids = []
    gaussian_mask_pyramids = []
    
    for warped_img, warped_mask in zip(warped_images, warped_masks):
        # Build Gaussian pyramid for image
        gaussian_pyr = build_gaussian_pyramid(warped_img, sigmas)
        gaussian_pyramids.append(gaussian_pyr)
        
        # Build Laplacian pyramid for image
        laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)
        laplacian_pyramids.append(laplacian_pyr)
        
        # Build Gaussian pyramid for mask
        mask_pyr = build_gaussian_pyramid(warped_mask, sigmas)
        gaussian_mask_pyramids.append(mask_pyr)
    
    # Blend using multi-band approach
    for level in range(n_levels):
        # Create full-size arrays for this level
        curr_band = np.zeros((H, W, 3), dtype=np.float32)
        curr_weights = np.zeros((H, W), dtype=np.float32)
        
        for i in range(len(warped_images)):
            x_min, y_min, x_max, y_max = bounding_boxes[i]
            
            # Get band and weights for this image at current level
            if level < len(laplacian_pyramids[i]):
                band = laplacian_pyramids[i][level]
                weights = gaussian_mask_pyramids[i][level]
                
                # Create full-size arrays for this image contribution
                band_full = np.zeros((H, W, 3), dtype=np.float32)
                weights_full = np.zeros((H, W), dtype=np.float32)
                
                # Place the band and weights in their bounding box location
                band_full[y_min:y_max, x_min:x_max] = band
                weights_full[y_min:y_max, x_min:x_max] = weights
                
                # Accumulate weighted bands
                curr_band += band_full * weights_full[..., np.newaxis]
                curr_weights += weights_full
        
        # Normalize and add to panorama
        valid_mask = curr_weights > 1e-6
        curr_band[valid_mask] /= curr_weights[valid_mask, np.newaxis]
        panorama += curr_band
    
    # Create final mask to exclude border extensions
    final_valid_mask = np.zeros((H, W), dtype=np.float32)
    for original_mask in warped_original_masks:
        final_valid_mask = np.maximum(final_valid_mask, original_mask)
    
    # Apply final mask to remove border artifacts
    panorama = panorama * final_valid_mask[..., np.newaxis]
    
    return np.clip(panorama, 0, 1)


def build_gaussian_pyramid(image: np.ndarray, sigmas: List[int]) -> List[np.ndarray]:
    """Build Gaussian pyramid with specified sigma values (same size, different blur)."""
    pyramid = []
    
    for sigma in sigmas:
        # Apply Gaussian blur with current sigma
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        pyramid.append(blurred)
    
    return pyramid



def build_laplacian_pyramid(gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:
    """Build Laplacian pyramid from Gaussian pyramid (same size levels)."""
    laplacian_pyramid = []
    
    for i in range(len(gaussian_pyramid) - 1):
        # Get current and next level (both same size, different blur)
        current = gaussian_pyramid[i]
        next_level = gaussian_pyramid[i + 1]
        
        # Compute Laplacian as difference (no upsampling needed)
        laplacian = current - next_level
        laplacian_pyramid.append(laplacian)
    
    # Add the most blurred level as the final Laplacian level
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid