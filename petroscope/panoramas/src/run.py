from matcher import Matcher
from stitcher import Stitcher
from runner import Runner

import os
import hydra
from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    matcher = Matcher(
        cfg.match.device,
        cfg.match.matcher_weights,
        cfg.match.batch_size,
        cfg.match.inference_size
    )

    stitcher = Stitcher(
        matcher=matcher,
        load_matches=cfg.load_matches,
        transformation_type=cfg.align.transformation_type,
        confidence_tr=cfg.align.confidence_tr,
        min_inliers=cfg.align.min_inliers,
        max_inliers=cfg.align.max_inliers,
        min_inlier_rate=cfg.align.min_inlier_rate,
        reproj_tr=cfg.align.reproj_tr,
        n_recenterings=cfg.align.n_recenterings,
        use_BA=cfg.align.use_BA,

        use_gain_comp=cfg.compose.use_gain_comp,
        use_graphcut=cfg.compose.use_graphcut,
        use_blending=cfg.compose.use_blending,

        save_mean_color=cfg.gain_comp.save_mean_color,

        coarse_scale=cfg.graphcut.coarse_scale,
        fine_scale=cfg.graphcut.fine_scale,
        lane_width=cfg.graphcut.lane_width,

        n_levels=cfg.blending.n_levels,

        detailed_log=cfg.log.detailed_log,

        draw_inliers=cfg.vizualization.draw_inliers,
        draw_connections=cfg.vizualization.draw_connections,

        custom_undistortion=cfg.custom_undistortion,
        n_undistortions=cfg.n_undistortions,
        stitching_mode=cfg.stitching_mode
    )

    runner = Runner()
    runner.process_collection(
        stitcher=stitcher,
        tiles_metadir=cfg.dirs.tiles_dir,
        cache_metadir=cfg.dirs.cache_dir,
        output_metadir=cfg.dirs.panoramas_dir
    )


if __name__ == "__main__":
    main()
