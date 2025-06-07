import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from logger import logger
import pickle
from tqdm.auto import tqdm
os.environ["HYDRA_FULL_ERROR"] = "1"


from alignment import Aligner
from utils import _warp_collage, _save, _load_transforms, _load_images, _warp_masked_collage
from gain_compensation_functions import gain_compensation, compensate_mean_color, find_mean_color
from graphcut_functions import find_graphcut_mask, _warp_coarse_to_fine
from blending_functions import multi_band_blending

@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def stitch_all_panoramas(cfg: DictConfig):
    logger.debug(f"Processing datasets dir: {cfg.dirs.datasets_dir}")
    logger.debug(f"Output panoramas dir: {cfg.dirs.panoramas_dir}")

    # Ensure output directories
    Path(cfg.dirs.transforms_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.dirs.panoramas_dir).mkdir(parents=True, exist_ok=True)

    # Initialize aligner if needed
    if cfg.stages.align:
        try:
            aligner = Aligner(device=cfg.device)
        except Exception as e:
            logger.error(f"Failed to initialize Aligner: {e}")
            return

    datasets = [d for d in Path(cfg.dirs.datasets_dir).iterdir() if d.is_dir()]

    with tqdm(datasets, desc="Datasets", position=0, leave=True, dynamic_ncols=True) as dataset_pbar:
        for dataset in dataset_pbar:
            dataset_pbar.set_postfix_str(f"{dataset.name}")
            logger.debug(f"Processing dataset: {dataset.name}")

            series_list = [s for s in dataset.iterdir() if s.is_dir()]
            with tqdm(series_list, desc="Series", position=1, leave=False, dynamic_ncols=True) as series_pbar:
                for series in series_pbar:
                    series_pbar.set_postfix_str(f"{series.name}")
                    logger.debug(f"Processing series: {series.name}")

                    # Stage vars
                    transforms = None
                    panorama_size = None
                    img_paths = None
                    images = None
                    panorama = None
                    masks = None
                    n_images = None

                    # 1. Alignment or loading transforms
                    if cfg.stages.align:
                        try:
                            img_paths = [p for p in series.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.tiff','.tif')]
                            transforms, panorama_size, img_paths = aligner.only_transforms(img_paths=img_paths)
                            logger.debug("Alignment completed")
                        except Exception as e:
                            logger.error(f"[Alignment] {dataset.name}/{series.name} failed: {e}")
                            continue
                    elif cfg.load_transforms:
                        try:
                            data_path = Path(cfg.dirs.transforms_dir) / dataset.name / f"{series.name}_transforms.pkl"
                            transforms, panorama_size, img_paths = _load_transforms(data_path)
                            logger.debug("Loaded existing transforms")
                        except Exception as e:
                            logger.error(f"[Transforms Loading] {dataset.name}/{series.name} failed: {e}")
                            continue
                    else:
                        raise ValueError("No alignment or transforms configuration enabled.")

                    # Loading images
                    try:
                        images = _load_images(img_paths)
                        n_images = len(images)
                        logger.debug("Images loaded")
                    except Exception as e:
                        logger.error(f"[Images Loading] {dataset.name}/{series.name} failed: {e}")
                        continue

                    # Saving transforms if requested
                    if cfg.save_transforms:
                        try:
                            out_dir = Path(cfg.dirs.transforms_dir) / dataset.name
                            out_dir.mkdir(parents=True, exist_ok=True)
                            with open(out_dir / f"{series.name}_transforms.pkl", 'wb') as f:
                                pickle.dump({
                                    'transforms': transforms,
                                    'panorama_size': panorama_size,
                                    'img_paths': img_paths
                                }, f)
                            logger.debug("Transforms saved")
                        except Exception as e:
                            logger.error(f"[Save Transforms] {dataset.name}/{series.name} failed: {e}")

                    # 2.1 Gain compensation
                    if cfg.stages.gaincomp:
                        try:
                            mean_before = find_mean_color(images)
                            images = gain_compensation(images, transforms, panorama_size)
                            scale = mean_before / (find_mean_color(images) + 1e-6)
                            images = [compensate_mean_color(img, scale).astype('float32') for img in images]
                            logger.debug("Gain compensation done")
                        except Exception as e:
                            logger.error(f"[Gain Compensation] {dataset.name}/{series.name} failed: {e}")

                    # 2.2 Graphcut
                    try:
                        if cfg.stages.graphcut:
                            img_indxes = find_graphcut_mask(
                                images, transforms, panorama_size,
                                coarse_scale=cfg.graphcut.coarse_scale,
                                fine_scale=cfg.graphcut.fine_scale,
                                lane_width=cfg.graphcut.lane_width
                            )
                            masks = [img_indxes == i for i in range(n_images)]
                            logger.debug("Graphcut warp done")
                    except Exception as e:
                        logger.error(f"[Warp] {dataset.name}/{series.name} failed: {e}")
                        continue

                    # 2.3 Blending
                    if cfg.stages.blending:
                        try:
                            panorama = multi_band_blending(
                                images, masks, transforms, panorama_size,
                                levels=cfg.blending.levels
                            )
                            logger.debug("Blending done")
                        except Exception as e:
                            logger.error(f"[Blending] {dataset.name}/{series.name} failed: {e}")
                    elif cfg.stages.graphcut:
                        try:
                            panorama = _warp_masked_collage(images, transforms, panorama_size, masks)
                            logger.debug("Collage warp done")
                        except Exception as e:
                            logger.error(f"[Warp] {dataset.name}/{series.name} failed: {e}")
                    else:
                        try:
                            panorama = _warp_collage(images, transforms, panorama_size)
                            logger.debug("Collage warp done")
                        except Exception as e:
                            logger.error(f"[Warp] {dataset.name}/{series.name} failed: {e}")

                    # Saving panorama
                    try:
                        out_dir = Path(cfg.dirs.panoramas_dir) / dataset.name
                        out_dir.mkdir(parents=True, exist_ok=True)
                        _save(panorama, out_dir / f"{series.name}_pano.jpg")
                        logger.debug("Panorama saved")
                    except Exception as e:
                        logger.error(f"[Save Panorama] {dataset.name}/{series.name} failed: {e}")

if __name__ == "__main__":
    stitch_all_panoramas()
