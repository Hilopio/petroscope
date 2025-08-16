from stitcher import Stitcher
from pathlib import Path
from logger import logger, log_time
from tqdm import tqdm


class Runner:
    @log_time("Total processing time:", logger)
    def process_collection(
        self,
        stitcher: Stitcher,
        tiles_metadir: Path | str,
        cache_metadir: Path | str,
        output_metadir: Path | str
    ) -> None:
        tiles_metadir = Path(tiles_metadir)
        cache_metadir = Path(cache_metadir)
        output_metadir = Path(output_metadir)

        datasets = [d for d in tiles_metadir.iterdir() if d.is_dir()]
        datasets.sort(key=lambda path: path.name)  # for deterministic order

        with tqdm(datasets, desc="Datasets", position=0, leave=True, dynamic_ncols=True) as dataset_pbar:
            for dataset in dataset_pbar:
                dataset_pbar.set_postfix_str(f"{dataset.name}")
                logger.info(f"Processing dataset: {dataset.name}")

                series_list = [s for s in dataset.iterdir() if s.is_dir()]
                series_list.sort(key=lambda path: path.name)  # for deterministic order
                with tqdm(series_list, desc="Series", position=1, leave=False, dynamic_ncols=True) as series_pbar:
                    for series in series_pbar:
                        series_pbar.set_postfix_str(f"{series.name}")
                        logger.info(f"Processing series: {series.name}")

                        tiles_dir = tiles_metadir / dataset.name / series.name
                        cache_dir = cache_metadir / dataset.name / series.name
                        output_file = output_metadir / dataset.name / f"{series.name}.jpg"
                        Path(output_file.parent).mkdir(parents=True, exist_ok=True)

                        try:
                            stitcher.stitch(tiles_dir, output_file, cache_dir)
                        except Exception as e:
                            logger.error(f"Runner error: {str(e)}")
