import hydra
from omegaconf import DictConfig
from pathlib import Path
from logger import logger
import pickle
from tqdm.auto import tqdm

from alignment import Aligner
from utils import _warp_collage, _save, _load_transforms, _load_images
from gain_compensation_functions import gain_compensation, compensate_mean_color, find_mean_color
from graphcut_functions import _warp_coarse_to_fine
from blending_functions import multi_band_blending, find_graphcut_mask

@hydra.main(version_base="1.2", config_path=".", config_name="config2.yaml")
def stitch_all_panoramas(cfg: DictConfig):
    logger.debug(f"Processing: {cfg.dirs.datasets_dir}")
    logger.debug(f"Results: {cfg.dirs.panoramas_dir}")
    Path(cfg.dirs.transforms_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.dirs.panoramas_dir).mkdir(parents=True, exist_ok=True)

    if cfg.stages.align:
        aligner = Aligner(device=cfg.device)

    datasets_dir = Path(cfg.dirs.datasets_dir)
    datasets = [d for d in datasets_dir.iterdir() if d.is_dir()]
    
    with tqdm(datasets, desc="Datasets", position=0, leave=True, dynamic_ncols=True) as dataset_pbar:
        for dataset in dataset_pbar:
            dataset_pbar.set_postfix_str(f"Current: {dataset.name}")
            logger.debug(f"Processing dataset: {dataset.name}")
            
            series_list = [s for s in dataset.iterdir() if s.is_dir()]
            
            with tqdm(series_list, desc="Series", position=1, leave=False, dynamic_ncols=True) as series_pbar:
                for series in series_pbar:
                    series_pbar.set_postfix_str(f"Current: {series.name}")
                    logger.debug(f"Processing series: {series.name}")
                    try:
                        if cfg.stages.align:
                            img_paths = [
                                img_p for img_p in series.iterdir() 
                                if img_p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif")
                            ]
                            transforms, panorama_size, img_paths = aligner.only_transforms(img_paths=img_paths)
                        elif cfg.load_transforms:
                            data_path = Path(cfg.dirs.transforms_dir) / dataset.name / f"{series.name}_transforms.pkl"
                            transforms, panorama_size, img_paths = _load_transforms(data_path)
                        else:
                            raise ValueError("No data for alignment. Use cfg.stages.align or cfg.load_transforms")
                        
                        images = _load_images(img_paths)

                        if cfg.save_transforms:
                            output_path = Path(cfg.dirs.transforms_dir) / dataset.name / f"{series.name}_transforms.pkl"
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_path, "wb") as f:
                                pickle.dump({
                                    "transforms": transforms,
                                    "panorama_size": panorama_size,
                                    "img_paths": img_paths
                                }, f)
                        
                        if cfg.stages.gaincomp:
                            target_mean_color = find_mean_color(images)
                            images = gain_compensation(images, transforms, panorama_size)
                            color_scale = target_mean_color / (find_mean_color(images) + 1e-6)
                            images = [compensate_mean_color(img, color_scale).astype('float32') for img in images]
                            logger.debug(f"Gain compensation done")

                        
                        if cfg.stages.graphcut:
                            panorama = _warp_coarse_to_fine(
                                images, transforms, panorama_size,
                                coarse_scale=cfg.graphcut.coarse_scale,
                                fine_scale=cfg.graphcut.fine_scale,
                                lane_width=cfg.graphcut.lane_width
                            )
                            logger.debug(f"Graphcut done")

                        else:
                            panorama = _warp_collage(images, transforms, panorama_size)
                            logger.debug(f"Collage done")
                        
                        if cfg.stages.blending:
                            img_indexes = find_graphcut_mask(images, transforms, panorama_size)
                            masks = [img_indexes == i for i in range(len(images))]
                            panorama = multi_band_blending(images, masks, transforms, panorama_size, levels=cfg.blending.levels)
                            logger.debug(f"Blending done")
                            
                        output_path = Path(cfg.dirs.panoramas_dir) / dataset.name / f"{series.name}_pano.jpg"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        _save(panorama, output_path)
                        
                    except Exception as e:
                        logger.error(f"Error processing {series.name}: {str(e)}", exc_info=True)
                        continue

if __name__ == "__main__":
    stitch_all_panoramas()