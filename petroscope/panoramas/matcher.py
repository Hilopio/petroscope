from pathlib import Path
import gc
import numpy as np
import torch
import torchvision
from PIL import Image
import kornia.feature as KF

from classes import MatchesData

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

    def _load_torch_tensors(self, img_paths: list[Path]) -> tuple[list[tuple[float]], list[torch.Tensor]]:
        """
        Load images as torch tensors for processing.
        
        Args:
            img_paths: List of file paths to the images to be loaded.
            
        Returns:
            tuple: A tuple containing:
                - List of original sizes of the images.
                - List of torch tensors representing the processed images.
        """
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

    def match(self, img_paths: list[Path]) -> 'MatchesData':
        """
        Match features between images to find correspondences.
        
        Args:
            img_paths: List of file paths to the images to be matched.
            
        Returns:
            MatchesData: Data object containing matching information between images.
        """
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
                "image0": batch1[batch_size * i: batch_size * (i + 1)].to(self.device),
                "image1": batch2[batch_size * i: batch_size * (i + 1)].to(self.device),
            }
            with torch.inference_mode():
                correspondences = self.matcher(input_dict)
            tmp = {
                "batch_indexes": correspondences["batch_indexes"].detach().cpu(),
                "keypoints0": correspondences["keypoints0"].detach().cpu(),
                "keypoints1": correspondences["keypoints1"].detach().cpu(),
                "confidence": correspondences["confidence"].detach().cpu(),
            }
            all_corr.append(tmp)
            del correspondences
            torch.cuda.empty_cache()
            gc.collect()

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

        # Restore real coordinates from downscaled ones
        for idx, corr in enumerate(diff_corr):
            if corr.shape[0] > 0:
                pair_idx = idx // (n - 1)  # First image index in pair
                second_idx = idx % (n - 1)  # Second image index relative to first
                if second_idx >= pair_idx:
                    second_idx += 1
                corr[:, 0:2] *= orig_sizes[pair_idx] / self.size  # Rescale keypoints for first image
                corr[:, 2:4] *= orig_sizes[second_idx] / self.size  # Rescale keypoints for second image
                diff_corr[idx] = corr

        return MatchesData(img_paths, diff_corr, orig_sizes)
