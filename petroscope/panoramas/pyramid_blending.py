import numpy as np
import cv2
from typing import List, Tuple


def multi_band_blending(images: List[np.ndarray], 
                       masks: List[np.ndarray], 
                       transforms: List[np.ndarray], 
                       panorama_size: Tuple[int, int], 
                       n_levels: int = 5) -> np.ndarray:
    """
    Multi-band blending for panorama stitching using downscaled Gaussian and Laplacian pyramids.
    
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
    
    # Calculate pyramid sizes for each level
    pyramid_sizes = calculate_pyramid_sizes(W, H, n_levels)
    
    # Максимальный делитель для выравнивания bounding box'ов
    max_divisor = 2 ** (n_levels - 1)
    border_size = 5 * max_divisor  # Используем max_divisor вместо sigma[-1]
    
    # Process each image
    warped_images = []
    warped_masks = []
    warped_original_masks = []
    aligned_bounding_boxes = []
    
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
        
        # Find bounding box based on warped image content
        coords = np.where(np.any(warped_img > 0, axis=2))
        if len(coords[0]) == 0:
            continue
            
        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1
        
        # Выравниваем только x_min и y_min чтобы координаты были кратны max_divisor
        x_min_aligned = (x_min // max_divisor) * max_divisor
        y_min_aligned = (y_min // max_divisor) * max_divisor
        
        # x_max и y_max оставляем как есть
        x_max_aligned = x_max
        y_max_aligned = y_max
        
        # Убеждаемся что не выходим за границы панорамы
        x_min_aligned = max(0, x_min_aligned)
        y_min_aligned = max(0, y_min_aligned)
        x_max_aligned = min(W, x_max_aligned)
        y_max_aligned = min(H, y_max_aligned)
        
        # Store cropped warped image and mask
        warped_images.append(warped_img[y_min_aligned:y_max_aligned, x_min_aligned:x_max_aligned])
        warped_masks.append(mask[y_min_aligned:y_max_aligned, x_min_aligned:x_max_aligned].astype(np.float32))
        warped_original_masks.append(warped_original_mask)  # Store full-size original mask
        aligned_bounding_boxes.append((x_min_aligned, y_min_aligned))
    
    if not warped_images:
        return np.zeros((H, W, 3), dtype=np.float32)
    
    # Build pyramids for each image
    gaussian_pyramids = []
    laplacian_pyramids = []
    gaussian_mask_pyramids = []
    
    for warped_img, warped_mask in zip(warped_images, warped_masks):
        # Build Gaussian pyramid for image
        gaussian_pyr = build_gaussian_pyramid(warped_img, n_levels)
        gaussian_pyramids.append(gaussian_pyr)
        
        # Build Laplacian pyramid for image
        laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)
        laplacian_pyramids.append(laplacian_pyr)
        
        # Build Gaussian pyramid for mask
        mask_pyr = build_gaussian_pyramid(warped_mask, n_levels)
        gaussian_mask_pyramids.append(mask_pyr)
    
    # Создаем панораму начиная с самого маленького уровня
    panorama = np.zeros(pyramid_sizes[-1][::-1] + (3,), dtype=np.float32)  # (H, W, 3)
    
    # Идем от самого маленького к самому большому уровню
    for level in range(n_levels - 1, -1, -1):
        curr_w, curr_h = pyramid_sizes[level]
        scale = 2 ** level
        
        # Создаем изображение и веса для текущего уровня
        curr_band = np.zeros((curr_h, curr_w, 3), dtype=np.float32)
        curr_weights = np.zeros((curr_h, curr_w), dtype=np.float32)
        
        if level != n_levels - 1:
            # Увеличиваем предыдущий результат (кроме самого маленького уровня)
            panorama = cv2.pyrUp(panorama, dstsize=(curr_w, curr_h))
        
        # Добавляем вклад каждого изображения на текущем уровне
        for i in range(len(warped_images)):
            x_min, y_min = aligned_bounding_boxes[i]
            
            # Вычисляем offset для текущего масштаба
            offset_x = x_min // scale
            offset_y = y_min // scale
            
            # Проверяем что offset выровнен правильно
            assert x_min % scale == 0, f"x_min {x_min} not divisible by scale {scale}"
            assert y_min % scale == 0, f"y_min {y_min} not divisible by scale {scale}"
            
            if level < len(laplacian_pyramids[i]):
                band = laplacian_pyramids[i][level]
                weights = gaussian_mask_pyramids[i][level]
                
                # Получаем размеры band
                band_h, band_w = band.shape[:2]
                
                # Жестко проверяем что не выходим за границы
                assert offset_y + band_h <= curr_h, f"Band height overflow: {offset_y + band_h} > {curr_h}"
                assert offset_x + band_w <= curr_w, f"Band width overflow: {offset_x + band_w} > {curr_w}"
                
                # Добавляем weighted band
                curr_band[offset_y:offset_y + band_h, offset_x:offset_x + band_w] += band * weights[..., np.newaxis]
                curr_weights[offset_y:offset_y + band_h, offset_x:offset_x + band_w] += weights
        
        # Нормализуем текущий уровень
        curr_band /= (curr_weights[..., np.newaxis] + 1e-6)
        
        # Добавляем к панораме
        panorama += curr_band
    
    # Create final mask to exclude border extensions
    final_valid_mask = np.zeros((H, W), dtype=np.float32)
    for original_mask in warped_original_masks:
        final_valid_mask = np.maximum(final_valid_mask, original_mask)
    
    # Apply final mask to remove border artifacts
    panorama = panorama * final_valid_mask[..., np.newaxis]
    
    return panorama


def calculate_pyramid_sizes(width: int, height: int, levels: int) -> List[Tuple[int, int]]:
    """
    Вычисляет размеры для каждого уровня пирамиды.
    pyrDown делает размер (w // 2 + w % 2, h // 2 + h % 2)
    """
    sizes = [(width, height)]
    w, h = width, height
    
    for _ in range(levels - 1):
        w = w // 2 + w % 2
        h = h // 2 + h % 2
        sizes.append((w, h))
    
    return sizes


def build_gaussian_pyramid(image, levels):
    pyramid = [image.copy()]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        expanded = cv2.pyrUp(
            gaussian_pyramid[i + 1], 
            dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        )
        laplacian = gaussian_pyramid[i] - expanded
        laplacian_pyramid.append(laplacian)
    
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid