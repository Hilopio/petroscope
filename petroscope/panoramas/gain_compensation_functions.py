import numpy as np
from scipy.optimize import least_squares, Bounds
from utils import _warp


def find_mean_color(images):
    share_of_borders = 0.3
    maen_colors = []
    for img in images:
        height, width = img.shape[:-1]
        center_slise = (
            slice(int(height * share_of_borders), height - int(height * share_of_borders)),
            slice(int(width * share_of_borders), width - int(width * share_of_borders))
        )
        mean_color = np.mean(img[center_slise], axis=(0, 1))
        maen_colors.append(mean_color)
    return np.array(maen_colors).mean(axis=0)


def compensate_mean_color(image, color_scale):
    assert color_scale.shape == (3,)
    image = image * color_scale
    return image


def gain_compensation(images, transforms, panorama_size):
    n = len(images)

    def fun(g, Sums):
        output = []
        for i in range(n):
            for j in range(i+1, n):
                output.append((1 + g[i]) * Sums[i][j] - (1 + g[j]) * Sums[j][i])
        return np.array(output)

    warpeds, masks = [], []
    Sums = [[None] * n for _ in range(n)]

    for i in range(n):
        warped_img, warped_mask = _warp(images[i], transforms[i], panorama_size)
        warpeds.append(warped_img)
        masks.append(warped_mask)

    for i in range(n-1):
        for j in range(i+1, n):
            mask = masks[i] & masks[j]
            Sums[i][j] = warpeds[i][mask].sum()
            Sums[j][i] = warpeds[j][mask].sum()

    g = np.zeros(n, dtype=np.float32)
    bounds = Bounds([-0.3] * n, [0.3] * n)
    res = least_squares(fun, g, method="trf", xtol=1e-10, ftol=1e-10, args=(Sums,), bounds=bounds)
    images = [img * (1 + res.x[i]) for i, img in enumerate(images)]
    return images


def gain_compensation3(images, transforms, panorama_size):
    n = len(images)

    def fun(g, Sums):
        G = g.reshape((n, 3))
        output = []
        for i in range(n):
            for j in range(i+1, n):
                output.append((1 + G[i]) * Sums[i][j] - (1 + G[j]) * Sums[j][i])
        return np.array(output).reshape(-1)

    warpeds, masks = [], []
    Sums = [[None] * n for _ in range(n)]

    for i in range(n):
        warped_img, warped_mask = _warp(images[i], transforms[i], panorama_size)
        warpeds.append(warped_img)
        masks.append(warped_mask)

    for i in range(n-1):
        for j in range(i+1, n):
            mask = masks[i] & masks[j]
            Sums[i][j] = warpeds[i][mask].sum(axis=0)
            Sums[j][i] = warpeds[j][mask].sum(axis=0)

    g = np.zeros((n * 3), dtype=np.float32)
    bounds = Bounds([-0.3] * (3 * n), [0.3] * (3 * n))
    res = least_squares(fun, g, method="trf", xtol=1e-6, ftol=1e-6, args=(Sums,), bounds=bounds)
    g = res.x.reshape((n, 3))

    images = [img * (1 + g[i]).astype('float32') for i, img in enumerate(images)]
    return images
