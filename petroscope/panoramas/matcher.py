import numpy as np
import torch
import kornia.feature as KF
import gc

from logger import logger, log_time
from classes import TileSet, StitchingData, Match


class Matcher:
    def __init__(self, device: str, batch_size: int = 10, inference_size: list = [600, 400]) -> None:
        """
        Initialize the Matcher with a specified device.

        Args:
            device: The device to be used for matching operations (e.g., 'cpu' or 'cuda').
        """
        self.device = device if device else torch.device("cpu")
        self.model = KF.LoFTR(pretrained="outdoor").to(self.device)
        self.batch_size = batch_size
        self.inference_size = inference_size

    @log_time("Matching done for", logger)
    def match(self, tile_set: TileSet, batch_size: int = None, inference_size: list = None) -> StitchingData:
        """
        Match features between images to find correspondences using the LoFTR model.

        Args:
            tile_set (TileSet): A set of images to be matched, containing image data and processing order.
            batch_size (int, optional): The number of image pairs to process in each batch. Defaults to 10.

        Returns:
            StitchingData: Data object containing matching information between images, including matched points
                and confidence scores.
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        inference_size = self.inference_size if inference_size is None else inference_size

        n: int = len(tile_set.order)

        batch1: list[torch.Tensor] = []
        batch2: list[torch.Tensor] = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                id_i: int = tile_set.order[i]
                id_j: int = tile_set.order[j]

                img_i = tile_set.images[id_i]
                img_j = tile_set.images[id_j]

                img_i.inference_size = inference_size
                img_j.inference_size = inference_size

                array_i = img_i.image_grayscale_downscaled
                array_j = img_j.image_grayscale_downscaled

                tensor_i = torch.from_numpy(array_i).unsqueeze(0).unsqueeze(0)
                tensor_j = torch.from_numpy(array_j).unsqueeze(0).unsqueeze(0)

                batch1.append(tensor_i)
                batch2.append(tensor_j)

        batch1_tensor = torch.cat(batch1, dim=0)
        batch2_tensor = torch.cat(batch2, dim=0)

        loftr_results: list[np.ndarray] = []
        total_infer: int = n * (n - 1) // 2
        batch_num: int = (total_infer - 1) // batch_size + 1

        for i in range(batch_num):
            input_dict = {
                "image0": batch1_tensor[batch_size * i: batch_size * (i + 1)].to(self.device),
                "image1": batch2_tensor[batch_size * i: batch_size * (i + 1)].to(self.device),
            }
            with torch.inference_mode():
                correspondences = self.model(input_dict)
            batch_result = {
                "batch_indexes": correspondences["batch_indexes"].detach().cpu(),
                "keypoints0": correspondences["keypoints0"].detach().cpu(),
                "keypoints1": correspondences["keypoints1"].detach().cpu(),
                "confidence": correspondences["confidence"].detach().cpu(),
            }
            del correspondences
            gc.collect()
            torch.cuda.empty_cache()

            for i in range(batch_size):
                idx = batch_result["batch_indexes"] == i
                if not idx.any():  # Possible bug in batch processing
                    break
                kp0 = batch_result["keypoints0"][idx]
                kp1 = batch_result["keypoints1"][idx]
                conf = batch_result["confidence"][idx]
                loftr_results.append(
                    np.concatenate([kp0.numpy(), kp1.numpy(), conf.numpy()[..., None]], axis=-1)
                )

        inference_size = np.array(inference_size)
        matches: list[Match] = []
        result_index = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                corrs = loftr_results[result_index]
                result_index += 1
                id_i = tile_set.order[i]
                id_j = tile_set.order[j]
                # idx: np.ndarray = np.arange(corrs.shape[0])
                xy_i = corrs[:, 0:2] * tile_set.images[id_i].orig_size / self.inference_size
                xy_j = corrs[:, 2:4] * tile_set.images[id_j].orig_size / self.inference_size
                conf = corrs[:, 4]
                # print(conf.shape)
                # matches.extend([
                #     Match(id_i, id_j, xy_i[k], xy_j[k], conf[k])
                #     for k in range(corrs.shape[0])
                # ])
                matches.append(Match(id_i, id_j, xy_i, xy_j, conf))

        return StitchingData(
            tile_set=tile_set,
            matches=matches,
            reper_idx=None,
            panorama_size=None,
            canvas=None
        )
