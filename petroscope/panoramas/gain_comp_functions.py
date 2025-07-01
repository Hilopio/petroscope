import numpy as np
from classes import StitchingData
from utils import _warp
from scipy.optimize import least_squares, Bounds
from logger import logger, log_time


def find_mean_color(images: list[np.ndarray], gains: list[int] | None = None) -> np.ndarray:
    """
    Calculate the mean color value (RGB) across all images.

    Args:
        images: List of image arrays with shape (height, width, 3).

    Returns:
        np.ndarray: Mean color value as an array of shape (3,) representing (R, G, B).
    """
    if gains is None:
        gains = np.ones(len(images))

    n_channels = images[0].shape[-1]
    mean_color = np.zeros(n_channels)
    for i, img in enumerate(images):
        mean_color += np.mean(img, axis=(0, 1)) * gains[i]
    mean_color /= len(images)
    return mean_color


def fun(gains, Sums):
    n = Sums.shape[0]
    n_channels = Sums.shape[-1]
    g = gains.reshape(n, n_channels)
    i_idx, j_idx = np.triu_indices(n, k=1)
    result = (1 + g[i_idx]) * Sums[i_idx, j_idx] - (1 + g[j_idx]) * Sums[j_idx, i_idx]

    assert result.shape == (n * (n - 1) // 2, n_channels)
    return result.ravel()


def compute_gains(images: np.ndarray, transforms: list[np.ndarray], panorama_size: tuple[int]) -> list[float]:
    n = len(images)
    n_channels = images[0].shape[-1]

    warpeds, masks = [], []
    Sums = np.empty((n, n, n_channels))

    for i in range(n):
        warped_img, warped_mask = _warp(images[i], transforms[i], panorama_size)
        warpeds.append(warped_img)
        masks.append(warped_mask)

    for i in range(n-1):
        for j in range(i+1, n):
            mask = masks[i] & masks[j]
            Sums[i, j] = warpeds[i][mask].sum(axis=(0, 1))
            Sums[j, i] = warpeds[j][mask].sum(axis=(0, 1))

    g = np.zeros(n * n_channels)
    bounds = Bounds(
        -0.3 * np.ones_like(g),
        0.3 * np.ones_like(g)
    )
    res = least_squares(fun, g, method="trf", xtol=1e-10, ftol=1e-10, args=(Sums,), bounds=bounds)
    gains = 1 + res.x.reshape(n, n_channels)
    return gains


@log_time("Gain compensation done for", logger)
def apply_gain_comp(data: StitchingData, save_mean_color: bool):
    images: list[np.arrays] = []
    homographies: list[np.arrays] = []
    for id in data.tile_set.order:
        img = data.tile_set.images[id]
        images.append(img.image)
        homographies.append(img.homography)

    if save_mean_color:
        mean_color_before = find_mean_color(images)
        gains = compute_gains(images, homographies, data.panorama_size)
        mean_color_after = find_mean_color(images, gains)
        scale = mean_color_before / (mean_color_after + 1e-6)
        gains = [gain * scale for gain in gains]
    else:
        gains = compute_gains(images, homographies, data.panorama_size)

    for id, gain in zip(data.tile_set.order, gains):
        data.tile_set.images[id].gain = gain

    return data
