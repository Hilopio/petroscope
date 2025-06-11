import os
import hydra
from omegaconf import DictConfig

from stitcher import Stitcher

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    stchr = Stitcher(cfg)
    stchr.run()


if __name__ == "__main__":
    main()