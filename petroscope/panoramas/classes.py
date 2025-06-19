from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image


@dataclass
class ImageStructure:
    id : int
    img_path : Path
    image : np.ndarray
    orig_size : np.ndarray
    gain : np.ndarray

    @property
    def Image(self) -> np.ndarray:
        """
        Property to get the image. If not loaded, it loads the image from the specified path as float RGB in range [0, 1].
        
        Returns:
            np.ndarray: The loaded image as a numpy array in range [0, 1].
        """
        if self.image is None:
            self.image = np.array(Image.open(self.img_path)).astype(np.float32) / 255.0
        return self.image

    @property
    def ImageCompensated(self) -> np.ndarray:
        """
        Property to get the gain-compensated image. If not loaded, it loads the image from the specified path
        as float RGB in range [0, 1] and applies gain compensation.
        
        Returns:
            np.ndarray: The gain-compensated image as a numpy array.
        """
        if self.image is None:
            self.image = np.array(Image.open(self.img_path)).astype(np.float32) / 255.0
        return self.image * self.gain

@dataclass
class MatchStructure:
    i : int
    j : int
    xy_i : np.ndarray
    xy_j : np.ndarray
    conf : float

@dataclass
class StitchingData:
    images : list[ImageStructure]
    matches : list[MatchStructure]
    transforms : list[np.ndarray]
    reper_id : int
    panorama_size : tuple

@dataclass
class PanoramaData:
    panorama : np.ndarray
    canvas : np.ndarray

@dataclass
class MatchesData:
    """
    Data class to store information about image matches.
    
    Attributes:
        img_paths (list[Path]): List of file paths to the images.
        matches (list[np.ndarray]): List of arrays containing matching points between images.
        orig_sizes (list[np.ndarray]): List of original sizes of the images.
    """
    img_paths : list[Path]
    matches : list[np.ndarray]
    orig_sizes : list[np.ndarray]

@dataclass
class AlignData:
    """
    Data class to store alignment data for images.
    
    Attributes:
        img_paths (list[Path]): List of file paths to the images.
        transforms (list[np.ndarray]): List of transformation matrices for aligning images.
        reference_idx (int): Index of the reference image used for alignment.
        inliers (list[list]): List of inlier points used for alignment.
    """
    img_paths : list[Path]
    transforms : list[np.ndarray]
    reference_idx : int # reper_id???
    inliers : list[list]

@dataclass
class OptimizeData:
    """
    Data class to store optimized transformation data for images.
    
    Attributes:
        img_paths (list[Path]): List of file paths to the images.
        transforms (list[np.ndarray]): List of optimized transformation matrices.
        reference_idx (int): Index of the reference image used for optimization.
    """
    img_paths : list[Path]
    transforms : list[np.ndarray]
    reference_idx : int

@dataclass
class AlignmentData:
    """
    Data class to store final alignment data for creating a panorama.
    
    Attributes:
        img_paths (list[Path]): List of file paths to the images.
        transforms (list[np.ndarray]): List of final transformation matrices for alignment.
        reference_idx (int): Index of the reference image used for alignment.
        panorama_size (tuple): Tuple representing the size of the panorama (width, height).
    """
    img_paths : list[Path]
    transforms : list[np.ndarray]
    reference_idx : int
    panorama_size : tuple

@dataclass
class GaincompData:
    """
    Data class to store gain-compensated image data.
    
    Attributes:
        images (list[np.ndarray]): List of gain-compensated image arrays.
        transforms (list[np.ndarray]): List of transformation matrices for aligning images.
        panorama_size (tuple): Tuple representing the size of the panorama (width, height).
    """
    images : list[np.ndarray]
    transforms : list[np.ndarray]
    panorama_size : tuple

@dataclass
class MosaicData:
    """
    Data class to store mosaic canvas data after graphcut application.
    
    Attributes:
        canvas (np.ndarray): Array representing the mosaic canvas after applying graphcut masks.
        images (list[np.ndarray]): List of image arrays used in the mosaic.
        transforms (list[np.ndarray]): List of transformation matrices for aligning images.
        panorama_size (tuple): Tuple representing the size of the panorama (width, height).
    """
    canvas : np.ndarray
    images : list[np.ndarray]
    transforms : list[np.ndarray]
    panorama_size : tuple

@dataclass
class PanoramaData:
    """
    Data class to store the final panorama image data.
    
    Attributes:
        panorama (np.ndarray): Array representing the final stitched panorama image.
        images (list[np.ndarray]): List of image arrays used in the panorama.
        transforms (list[np.ndarray]): List of transformation matrices for aligning images.
        panorama_size (tuple): Tuple representing the size of the panorama (width, height).
    """
    panorama : np.ndarray
    images : list[np.ndarray]
    transforms : list[np.ndarray]
    panorama_size : tuple
    
    def save(self, path: Path) -> None:
        """
        Save the panorama image to the specified path.
        
        Args:
            path: Path where the panorama image will be saved.
        """
        image = (self.panorama.clip(0, 1) * 255).astype('uint8')
        output_image = Image.fromarray(image)
        output_image.save(path, quality=95)
