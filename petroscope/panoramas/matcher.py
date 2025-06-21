import numpy as np
import torch
import kornia.feature as KF

from classes import TileSet, StitchingData, Match


class Matcher:
    def __init__(self, device: str) -> None:
        """
        Initialize the Matcher with a specified device.

        Args:
            device: The device to be used for matching operations (e.g., 'cpu' or 'cuda').
        """
        self.device = device if device else torch.device("cpu")
        self.matcher = KF.LoFTR(pretrained="outdoor").to(self.device)
        self.size = np.array((600, 400))

    def match(self, tile_set: TileSet, batch_size=10) -> Match:
        """
        Match features between images to find correspondences.

        Args:
            img_paths: List of file paths to the images to be matched.

        Returns:
            MatchesData: Data object containing matching information between images.
        """
        n = len(tile_set.order)

        batch1 = []
        batch2 = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                id_i = tile_set.order[i]
                id_j = tile_set.order[j]
                tensor_i = tile_set.images[id_i].ImageGrayscaleDownscaled
                tensor_j = tile_set.images[id_j].ImageGrayscaleDownscaled
                batch1.append(tensor_i)
                batch2.append(tensor_j)
        batch1 = torch.cat(batch1)
        batch2 = torch.cat(batch2)

        loftr_results = []  # list[list[np.ndarray]]
        total_infer = n * (n - 1) // 2
        batch_num = (total_infer - 1) // batch_size + 1

        # Run the LoFTR model on the images
        for i in range(batch_num):
            input_dict = {
                "image0": batch1[batch_size * i: batch_size * (i + 1)].to(self.device),
                "image1": batch2[batch_size * i: batch_size * (i + 1)].to(self.device),
            }
            with torch.inference_mode():
                correspondences = self.matcher(input_dict)
            batch = {
                "batch_indexes": correspondences["batch_indexes"].detach().cpu(),
                "keypoints0": correspondences["keypoints0"].detach().cpu(),
                "keypoints1": correspondences["keypoints1"].detach().cpu(),
                "confidence": correspondences["confidence"].detach().cpu(),
            }

            for i in range(batch_size):
                idx = batch["batch_indexes"] == i
                if not idx.any():  # мб баг
                    break
                kp0 = batch["keypoints0"][idx]
                kp1 = batch["keypoints1"][idx]
                conf = batch["confidence"][idx]
                loftr_results.append(
                    np.concatenate([kp0, kp1, conf[..., None]], axis=-1)
                )

        matches = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                corrs = loftr_results.pop(0)
                id_i = tile_set.order[i]
                id_j = tile_set.order[j]
                xy_i = corrs[idx, 0:2] * tile_set.images[id_i].orig_size / self.size
                xy_j = corrs[idx, 2:4] * tile_set.images[id_j].orig_size / self.size
                conf = corrs[idx, 4]
                matches.extend([
                    Match(i, j, xy_i[idx], xy_j[idx], conf[idx])
                    for idx in range(corrs.shape[0])
                ])

        return StitchingData(tile_set, matches, None, None)
