import numpy as np
import torch
import kornia.feature as KF
import gc

from logger import logger, log_time
from classes import TileSet, StitchingData, Match


class Matcher:
    def __init__(
        self, device: str, weights: str = 'outdoor',
        batch_size: int = 10, inference_size: list = [600, 400]
    ) -> None:
        """
        Initialize the Matcher with a specified device.

        Args:
            device: The device to be used for matching operations (e.g., 'cpu' or 'cuda').
        """
        self.device = device if device else torch.device("cpu")
        self.batch_size = batch_size
        self.inference_size = inference_size

        if weights == 'outdoor' or weights == 'indoor':
            self.model = KF.LoFTR(pretrained=weights).to(self.device)
        else:
            self.model = KF.LoFTR(pretrained=None)
            checkpoint = torch.load(weights, map_location=device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(device)

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
        inference_size = np.array(self.inference_size if inference_size is None else inference_size)

        n = len(tile_set.order)
        # Генерируем пары индексов с помощью triu_indices (i < j)
        ord_i, ord_j = np.triu_indices(n, k=1)
        pairs = list(zip(ord_i, ord_j))
        total_infer = len(pairs)
        batch_num = (total_infer - 1) // batch_size + 1

        matches: list[Match] = []

        for batch_idx in range(batch_num):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_infer)
            current_pairs = pairs[start:end]

            batch1: list[torch.Tensor] = []
            batch2: list[torch.Tensor] = []
            for pair in current_pairs:
                i, j = pair
                id_i = tile_set.order[i]
                id_j = tile_set.order[j]

                img_i = tile_set.images[id_i]
                img_j = tile_set.images[id_j]

                img_i.inference_size = inference_size.tolist()
                img_j.inference_size = inference_size.tolist()

                array_i = img_i.image_grayscale_downscaled
                array_j = img_j.image_grayscale_downscaled

                tensor_i = torch.from_numpy(array_i).unsqueeze(0).unsqueeze(0)
                tensor_j = torch.from_numpy(array_j).unsqueeze(0).unsqueeze(0)

                batch1.append(tensor_i)
                batch2.append(tensor_j)

            if not batch1:
                raise Exception("Batch is empty")

            batch1_tensor = torch.cat(batch1, dim=0).to(self.device)
            batch2_tensor = torch.cat(batch2, dim=0).to(self.device)

            input_dict = {
                "image0": batch1_tensor,
                "image1": batch2_tensor,
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

            for local_i, (i, j) in enumerate(current_pairs):
                id_i = tile_set.order[i]
                id_j = tile_set.order[j]

                idx = batch_result["batch_indexes"] == local_i
                if not idx.any():
                    continue

                kp0 = batch_result["keypoints0"][idx].numpy()
                kp1 = batch_result["keypoints1"][idx].numpy()
                conf = batch_result["confidence"][idx].numpy()

                xy_i = kp0 * tile_set.images[id_i].orig_size / inference_size
                xy_j = kp1 * tile_set.images[id_j].orig_size / inference_size

                matches.append(Match(id_i, id_j, xy_i, xy_j, conf))

        return StitchingData(
            tile_set=tile_set,
            matches=matches,
            reper_idx=None,
            panorama_size=None,
            canvas=None
        )
