from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import torch


@dataclass
class Tile:
    """
    Data class to store information about an individual image used in stitching.

    Attributes:
        id (int): Unique identifier for the image.
        img_path (Path): File path to the image.
        image (np.ndarray): Loaded image data as a numpy array.
        orig_size (np.ndarray): Original dimensions of the image.
        gain (np.ndarray): Gain coefficients for color channels.
    """
    id: int
    img_path: Path
    _image: np.ndarray
    orig_size: np.ndarray
    homography: np.ndarray
    gain: np.ndarray
    _tensor: torch.Tensor

    def _load_image(self):
        self._image = np.array(Image.open(self.img_path)).astype(np.float32) / 255.0
        h, w = self._image.shape[:2]
        # c = 1 if self._image.ndim == 2 else self._image.shape[2]
        self.orig_size = np.array((w, h))

    @property
    def image(self) -> np.ndarray:
        """
        Property to access the image data. If the image is not already loaded, it reads the image from the specified
        file path and converts it to a numpy array with float values in the RGB color space, normalized to the range
        [0, 1].

        Returns:
            np.ndarray: The loaded image as a numpy array with values in the range [0, 1].
        """
        if self._image is None:
            self._load_image()

        return self._image

    @property
    def image_compensated(self) -> np.ndarray:
        """
        Property to access the gain-compensated image data. If the image is not already loaded, it reads the image
        from the specified file path, converts it to a numpy array with float values in the RGB color space,
        normalized to the range [0, 1], and applies gain compensation using the stored gain coefficients for each
        color channel.

        Returns:
            np.ndarray: The gain-compensated image as a numpy array with values adjusted by the gain coefficients.
        """
        if self._image is None:
            self._load_image()
        return self._image * self.gain.astype('float32')

    def get_loftr_tensor(self, size: np.ndarray) -> torch.Tensor:
        if size.shape != (2,) or not np.issubdtype(size.dtype, np.integer):
            raise ValueError("size must be a 2-element numpy array of integers")

        if self._image is None:
            self._load_image()

        c = 1 if self._image.ndim == 2 else self._image.shape[2]
        if c not in [1, 3]:
            raise ValueError(f"Unsupported number of channels: {c}. Expected 1 (grayscale) or 3 (RGB).")

        # size is (width, height), tensor shape is (1, 1, height, width)
        if self._tensor is None or self._tensor.shape != (1, 1, size[1], size[0]):
            grayscale = cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY) if c == 3 else self._image
            downscale = cv2.resize(grayscale, size, interpolation=cv2.INTER_LANCZOS4)
            self._tensor = torch.from_numpy(downscale)[None, None]

        return self._tensor


@dataclass
class TileSet:
    """
    Data class to store a set of images with their processing order.

    Attributes:
        order (list[int]): List of image IDs representing the processing order.
        images (dict[int, ImageStructure]): Dictionary mapping image IDs to their ImageStructure objects.
    """
    order: list[int]
    images: dict[int, Tile]


@dataclass
class Match:
    """
    Data class to store information about matches between two images.

    Attributes:
        i (int): Index of the first image.
        j (int): Index of the second image.
        xy_i (np.ndarray): Coordinates of matching points in the first image.
        xy_j (np.ndarray): Coordinates of matching points in the second image.
        conf (float): Confidence score of the matches.
    """
    i: int
    j: int
    xy_i: np.ndarray
    xy_j: np.ndarray
    conf: np.ndarray


@dataclass
class StitchingData:
    """
    Data class to store data required for stitching images into a panorama.

    Attributes:
        images (list[ImageStructure]): List of image structures containing image data.
        matches (list[MatchStructure]): List of match structures containing matching points between images.
        transforms (list[np.ndarray]): List of transformation matrices for aligning images.
        reper_id (int): Index of the reference image used for alignment.
        panorama_size (tuple): Tuple representing the size of the panorama (width, height).
    """
    tile_set: TileSet
    matches: list[Match]
    reper_idx: int
    panorama_size: tuple
    canvas: np.ndarray

    @property
    def images(self) -> dict[int, Tile]:
        """
        Property to access the images dictionary from the tile_set for backward compatibility.

        Returns:
            dict[int, Tile]: Dictionary mapping image IDs to their Tile objects.
        """
        return self.tile_set.images


@dataclass
class Panorama:
    """
    Data class to store the final panorama and canvas data.

    Attributes:
        panorama (np.ndarray): Array representing the final stitched panorama image.
        canvas (np.ndarray): Array representing the canvas used for stitching.
    """
    panorama: np.ndarray
    canvas: np.ndarray

    def save_panorama(self, path: Path, mode: str = 'RGB') -> None:
        """
        Save the panorama image to the specified path in a high-quality format.

        Args:
            path (Path): Path where the panorama image will be saved, including the file extension (e.g., .jpg, .png).
            mode (str): Output mode for the image, either 'RGB' or 'RGBA'. If 'RGBA', the mask is used as the alpha
                channel (default: 'RGB').

        Returns:
            None

        Notes:
            The image is saved with a quality setting of 95 for JPEG format to balance file size and visual fidelity.
        """
        image = (self.panorama.clip(0, 1) * 255).astype('uint8')
        if mode.upper() == 'RGBA':
            alpha = (self.mask * 255).astype('uint8')
            if image.shape[-1] == 3:
                image = np.dstack((image, alpha))
            else:
                image[:, :, 3] = alpha

        c = 1 if image.ndim == 2 else image.shape[2]
        if c == 1:
            image = image.squeeze(-1)
        output_image = Image.fromarray(image)
        output_image.save(path, quality=95)

    def save_canvas(self, path: Path, colormap: str = 'viridis') -> None:
        """
        Save the canvas image to the specified path, visualizing unique values with a specified colormap.

        Args:
            path (Path): Path where the canvas image will be saved, including the file extension (e.g., .jpg, .png).
            colormap (str): Name of the matplotlib colormap to use for visualizing unique values (default: 'viridis').

        Returns:
            None

        Notes:
            The canvas values are normalized based on their unique range before applying the colormap, and the image is
            saved with a quality setting of 95 for JPEG format.
        """
        unique_values = np.unique(self.canvas)
        if len(unique_values) > 1:
            norm = plt.Normalize(vmin=min(unique_values), vmax=max(unique_values))
        else:
            norm = plt.Normalize(vmin=0, vmax=1)

        cmap = cm.get_cmap(colormap)
        colored_canvas = cmap(norm(self.canvas))
        colored_canvas = (colored_canvas[:, :, :3] * 255).astype('uint8')

        output_image = Image.fromarray(colored_canvas)
        output_image.save(path, quality=95)

    def save_mask(self, path: Path) -> None:
        """
        Save the mask image derived from the canvas to the specified path as a grayscale image.

        Args:
            path (Path): Path where the mask image will be saved, including the file extension (e.g., .jpg, .png).

        Returns:
            None

        Notes:
            The mask is converted to a grayscale image with binary values (0 or 255) and saved with a quality setting
            of 95 for JPEG format.
        """
        mask_image = (self.mask * 255).astype('uint8')
        output_image = Image.fromarray(mask_image)
        output_image.save(path, quality=95)

    @property
    def mask(self) -> np.ndarray:
        """
        Property to get a mask from the canvas. The mask is True where values are >= 0, False otherwise.

        Returns:
            np.ndarray: Boolean mask array derived from the canvas.
        """
        return np.where(self.canvas >= 0, True, False)
