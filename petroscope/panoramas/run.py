import hydra
from omegaconf import DictConfig
from pathlib import Path
from logger import logger
import pickle

from alignment import Aligner
from utils import _warp_collage, _save, _load_transforms, _load_images
from gain_compensation_functions import gain_compensation, compensate_mean_color, find_mean_color
from graphcut_functions import _warp_coarse_to_fine

@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def stitch_all_panoramas(cfg: DictConfig):
    if cfg.stages.align:
        aligner = Aligner(device=cfg.device)

    datasets_dir = Path(cfg.dirs.datasets_dir)
    for dataset in datasets_dir.iterdir():
        if not dataset.is_dir():
            logger.warning(f"{dataset} is not a directory")

        for series in dataset.iterdir():
            if not series.is_dir():
                logger.warning(f"{series} is not a directory")

            if cfg.stages.align:
                img_paths = [
                    img_p
                    for img_p in series.iterdir()
                    if img_p.suffix in (".jpg", ".png", ".tiff")
                ]

                transforms, panorama_size, img_paths = aligner.only_transforms(
                    image_paths=img_paths
                )

            elif cfg.load_transforms:
                data_file = series.name + "_transforms.pkl"
                data_path = Path(cfg.dirs.panoramas_dir) / dataset.name / Path(data_file)
                transforms, panorama_size, img_paths = _load_transforms(data_path)
                images = _load_images(img_paths)

            else:
                raise ValueError("No data for alignement. Use cfg.stages.align or cfg.load_transforms")

            if cfg.save_transforms:
                output_file = series.name + "_transforms.pkl"
                output_path = Path(cfg.dirs.transforms_dir) / dataset.name / Path(output_file)
                data_to_save = {
                    "transforms": transforms,
                    "panorama_size": panorama_size,
                    "img_paths": img_paths
                }
                with open(output_path, "wb") as f:
                    pickle.dump(data_to_save, f)
            
            if cfg.stages.gaincomp:
                target_mean_color = find_mean_color(images)
                images = gain_compensation(images, transforms, panorama_size)
                new_mean_color = find_mean_color(images)
                color_scale = target_mean_color / (new_mean_color + 1e-6)
                images = [compensate_mean_color(img, color_scale) for img in images]
            
            if cfg.stages.graphcut:
                panorama = _warp_coarse_to_fine(images, transforms, panorama_size,
                                                coarse_scale=cfg.graphcut.coarse_scale,
                                                fine_scale=cfg.graphcut.fine_scale,
                                                lane_width=cfg.graphcut.lane_width)
            else:
                panorama = _warp_collage(images, transforms, panorama_size)

            output_file = series.name + "pano.jpg"
            output_path = Path(cfg.dirs.panoramas_dir) / dataset.name / Path(output_file)
            _save(panorama, output_path)    

if __name__ == "__main__":
    stitch_all_panoramas()