import gc
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import kornia.feature as KF
import numpy as np
import torch
import torchvision
from PIL import Image

from logger import logger
from bundle_adjustment import optimize


def find_translation_and_panorama_size(sizes, transformations):
    x_coords, y_coords = [], []
    for size, H in zip(sizes, transformations):
        if H is None:
            continue
        corners = np.array([
            [0, 0, 1],
            [0, size[1], 1],
            [size[0], 0, 1],
            [size[0], size[1], 1]
        ])
        new_corners = H @ corners.T

        new_corners /= new_corners[2]
        x_coords += new_corners[0].tolist()
        y_coords += new_corners[1].tolist()

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]])

    panorama_size = (int(np.ceil(x_max - x_min)), int(np.ceil(y_max - y_min)))
    return T, panorama_size


class Aligner:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cpu")
        self.matcher = KF.LoFTR(pretrained="outdoor").to(self.device)
        self.size = np.array((600, 400))

    def _load_torch_tensors(
        self, img_paths: Iterable[Path]
    ) -> Tuple[List[Tuple[float]], List[torch.Tensor]]:
        images = []
        orig_sizes = []
        for path in img_paths:
            img = Image.open(path).convert("L")  # Convert to grayscale
            orig_sizes.append(np.array(img.size))  # Save the original size
            img = img.resize((600, 400), resample=Image.Resampling.LANCZOS)
            img = torchvision.transforms.functional.pil_to_tensor(img)
            img = img.unsqueeze(dim=0)
            images.append(img)  # Append the processed tensor to the list

        return orig_sizes, images

    def only_transforms(
        self,
        img_paths: Iterable[Path],
    ) -> Image:
        n = len(img_paths)
        orig_sizes, images = self._load_torch_tensors(img_paths)

        batch1 = []
        batch2 = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                batch1.append(images[i])
                batch2.append(images[j])

        batch1 = torch.cat(batch1) / 255.0
        batch2 = torch.cat(batch2) / 255.0

        all_corr = []
        batch_size = 10
        total_infer = n * (n - 1) // 2
        batch_num = (total_infer - 1) // batch_size + 1

        # Run the LoFTR model on the images
        for i in range(batch_num):
            input_dict = {
                "image0": batch1[batch_size * i: batch_size * (i + 1)].to(
                    self.device
                ),
                "image1": batch2[batch_size * i: batch_size * (i + 1)].to(
                    self.device
                ),
            }
            with torch.inference_mode():
                correspondences = self.matcher(input_dict)
            tmp = {
                "batch_indexes": correspondences["batch_indexes"]
                .detach()
                .cpu(),
                "keypoints0": correspondences["keypoints0"].detach().cpu(),
                "keypoints1": correspondences["keypoints1"].detach().cpu(),
                "confidence": correspondences["confidence"].detach().cpu(),
            }
            all_corr.append(tmp)
            del correspondences
            torch.cuda.empty_cache()
            gc.collect()

        inliers = []
        diff_corr = []
        for batch_corr in all_corr:
            for i in range(batch_size):
                idx = batch_corr["batch_indexes"] == i
                if not idx.any():  # мб баг
                    break
                kp0 = batch_corr["keypoints0"][idx]
                kp1 = batch_corr["keypoints1"][idx]
                conf = batch_corr["confidence"][idx]
                diff_corr.append(
                    np.concatenate([kp0, kp1, conf[..., None]], axis=-1)
                )
                # print(sum(idx).item(), end=' ')

        # параметры подобраны эксперементально (все равно не идеально)
        confidence_threshold = 0.95
        min_inliers = 5
        max_inliers = 30

        Hs = [[None] * n for _ in range(n)]
        num_matches = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            for j in range(i + 1, n):
                corrs = diff_corr.pop(0)

                if corrs.shape[0] < min_inliers:
                    continue

                # Rescale to original resolution
                corrs[:, 0:2] *= orig_sizes[i] / self.size
                corrs[:, 2:4] *= orig_sizes[j] / self.size

                # Filter by confidence
                corrs = corrs[corrs[:, 4] > confidence_threshold]
                if corrs.shape[0] < min_inliers:
                    continue

                # Run RANSAC
                H_ij, mask = cv2.findHomography(
                    corrs[:, 0:2], corrs[:, 2:4],
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=1.0,
                    # maxIters=1000,
                    # confidence=0.99,
                    # refineIters=50,
                )
                if H_ij is None or mask.sum() < min_inliers:
                    continue

                num_matches[i][j] = int(mask.sum())
                num_matches[j][i] = num_matches[i][j]

                # Save forward H, and compute inverse for backward
                Hs[i][j] = H_ij
                try:
                    Hs[j][i] = np.linalg.inv(H_ij)
                    Hs[j][i] /= Hs[j][i][2, 2]
                except np.linalg.LinAlgError:
                    assert "Singular homography matrix"
                    # Hs[j][i] = None

                # Store top-k inliers (sorted by confidence)
                inliers_ij = corrs[mask.ravel().astype(bool)]
                topk = inliers_ij[inliers_ij[:, -1].argsort()[::-1]][:max_inliers]
                topk = topk[:, :-1]  # Remove confidence
                inliers += [[i, j, *inl] for inl in topk]

        # Initialize the transformations for each image
        transforms = [np.eye(3) for i in range(n)]

        queryIdx = [i for i in range(n)]
        targetIdx = []
        outliersIdx = []

        # pivot = np.argmax(num_matches.sum(axis=1))
        pivot = np.random.randint(0, n - 1)
        targetIdx.append(pivot)
        queryIdx.remove(pivot)

        while queryIdx:
            a = num_matches[queryIdx, :][:, targetIdx]
            # curr, best_neighb = np.unravel_index(
            #     np.argmax(a, axis=None), a.shape
            # )

            curr = np.argmax(a.sum(axis=1))
            best_neighb = np.argmax(a[curr])
     
            if Hs[queryIdx[curr]][targetIdx[best_neighb]] is None:
                assert num_matches[queryIdx[curr], targetIdx[best_neighb]] == 0, \
                    f"aaaa None homography, matches = {num_matches[queryIdx[curr], targetIdx[best_neighb]]}"
                transforms[queryIdx[curr]] = None
                outliersIdx.append(queryIdx[curr])
                queryIdx.pop(curr)
                continue

            H = (
                transforms[targetIdx[best_neighb]]
                @ Hs[queryIdx[curr]][targetIdx[best_neighb]]
            )
            H /= H[2, 2]
            transforms[queryIdx[curr]] = H
            targetIdx.append(queryIdx[curr])
            queryIdx.pop(curr)

        assert (set(targetIdx) | set(outliersIdx)) == set(range(n)), "sth wrong in lists"
        outliers_mask = [1 if i in outliersIdx else 0 for i in range(n)]
        idx_shift = np.cumsum(outliers_mask)

        # real_transforms = []
        # for i in range(len(transforms)):
        #     if i in outliersIdx:
        #         assert transforms[i] is None
        #         continue
        #     assert transforms[i] is not None
        #     real_transforms.append(transforms[i])

        real_transforms = [transforms[i] for i in targetIdx]

        real_inliers = []
        for inl in inliers:
            if inl[0] in outliersIdx or inl[1] in outliersIdx:
                # print('WTF???')
                continue
            inl[0], inl[1] = inl[0] - idx_shift[inl[0]], inl[1] - idx_shift[inl[1]]
            real_inliers.append(inl)
        pivot -= idx_shift[pivot]

        # recentering
        sizes = [orig_sizes[i] for i in targetIdx]
        n_recenterings = 25
        for _ in range(n_recenterings):
            real_transforms, new_pivot = recentering(real_transforms, sizes)
            if new_pivot == pivot:
                break
            pivot = new_pivot

        final_transforms, init_error, optim_error = optimize(
            real_transforms, real_inliers, pivot
        )
        logger.debug(f'final error = {optim_error}')

        # second recentering
        for _ in range(n_recenterings):
            final_transforms, new_pivot = recentering(final_transforms, sizes)
            if new_pivot == pivot:
                break
            pivot = new_pivot
        # final_transforms, _ = recentering(final_transforms, sizes)

        T, panorama_size = find_translation_and_panorama_size(orig_sizes, final_transforms)  # скорее всего orig_sizes -> sizes
        final_transforms = [T @ H for H in final_transforms]

        # Проверка на адекватность размера панорамы
        assert 0 < panorama_size[0] <= 20000 and 0 < panorama_size[1] <= 20000, \
            f"Invalid panorama size: {panorama_size}"

        # reordering
        new_img_paths = [img_paths[i] for i in targetIdx]
        targetIdx = [targetIdx[i] - idx_shift[targetIdx[i]] for i in range(len(targetIdx))]
        final_transforms = [final_transforms[i] for i in targetIdx]

        if len(outliersIdx) > 0:
            logger.debug(f'{len(outliersIdx)} image{"s" if len(outliersIdx) > 1 else ""} cannot be aligned')
        else:
            logger.debug(f'all images aligned')

        return final_transforms, panorama_size, new_img_paths

def recentering(transforms, sizes):
    N = len(transforms)
    assert len(sizes) == N

    img_centers = [ 
        [size[0] / 2, size[1] / 2] for size in sizes
    ]
    warped_img_centers = []
    for center, H in zip(img_centers, transforms):
        new_center = H @ np.array([center[0], center[1], 1])
        new_center /= new_center[2]
        warped_img_centers.append(new_center[:2])
    warped_img_centers = np.array(warped_img_centers)
    assert warped_img_centers.shape == (N, 2)

    # panorama_center = np.mean(warped_img_centers, axis=0)

    x_min, x_max = np.min(warped_img_centers[:, 0]), np.max(warped_img_centers[:, 0])
    y_min, y_max = np.min(warped_img_centers[:, 1]), np.max(warped_img_centers[:, 1])
    panorama_center = np.array((0.5 * (x_min + x_max), 0.5 * (y_min + y_max)))

    new_pivot = np.argmin(((warped_img_centers - panorama_center) ** 2).mean(axis=1))

    inv_pivot_H = np.linalg.inv(transforms[new_pivot])
    new_transforms = []
    for H in transforms:
        new_H = inv_pivot_H @ H
        new_H /= new_H[2, 2]
        new_transforms.append(new_H)
    return new_transforms, new_pivot