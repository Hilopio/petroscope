from matcher import Matcher
from new_stitcher import Stitcher
# from pathlib import Path
from time import time
import os
import hydra
from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    matcher = Matcher(cfg.device)
    stitcher = Stitcher(
        matcher=matcher,
        align=cfg.stages.align,
        gaincomp=cfg.stages.gaincomp,
        graphcut=cfg.stages.graphcut,
        blending=cfg.stages.blending,

        save_transforms=cfg.save_transforms,
        load_transforms=cfg.load_transforms,

        datasets_dir=cfg.dirs.datasets_dir,
        matches_dir=cfg.dirs.matches_dir,
        transforms_dir=cfg.dirs.transforms_dir,
        panoramas_dir=cfg.dirs.panoramas_dir,

        coarse_scale=cfg.graphcut.coarse_scale,
        fine_scale=cfg.graphcut.fine_scale,
        lane_width=cfg.graphcut.lane_width,

        levels=cfg.blending.levels,
    )
    start_time = time()
    stitcher.process_collection(cfg.dirs.datasets_dir, cfg.dirs.panoramas_dir, mode='full')
    print(f'processed for {time() - start_time} seconds')


if __name__ == "__main__":
    main()
