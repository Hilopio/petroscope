from classes import OptimizeData, AlignmentData
from align_functions import find_translation_and_panorama_size

def alignment(optimize_data: 'OptimizeData') -> 'AlignmentData':
    """
    Finalize alignment by calculating panorama size and applying translation to transformations.
    
    Args:
        optimize_data: OptimizeData object containing optimized transformations and pivot index.
    
    Returns:
        AlignmentData: Data object containing final transformations, panorama size, and image paths.
    """
    img_paths = optimize_data.img_paths
    transforms = optimize_data.transforms
    pivot = optimize_data.pivot
    T, panorama_size = find_translation_and_panorama_size(img_paths, transforms)
    final_transforms = [T @ H for H in transforms]
    return AlignmentData(img_paths, final_transforms, panorama_size, pivot)
