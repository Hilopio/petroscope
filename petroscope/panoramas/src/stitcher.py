from pathlib import Path
from classes import Tile, TileSet, StitchingData, Panorama
from tqdm import tqdm
from logger import logger, log_time

from matcher import Matcher
from align_functions import matches_alignment, translate_and_add_panorama_size
from optimizer import Optimizer
from collage_functions import make_collage, make_mosaic, make_collage_with_inliers

from gain_comp_functions import apply_gain_comp
from graphcut_functions import apply_graphcut
from blending_functions import apply_blending

from serializer import Serializer


class Stitcher:
    """
    A class to handle the stitching of multiple images into a single panorama.
    This class orchestrates the process of image matching, alignment, and composition
    to create a seamless panorama from a set of input images using various techniques
    like homography estimation, bundle adjustment, gain compensation, graphcut, and
    blending.
    """
    def __init__(self, matcher: Matcher, transformation_type: str = "projective",
                 confidence_tr: float = 0.95, min_inliers: int = 5,
                 max_inliers: int = 200, min_inlier_rate: float = 0.0, reproj_tr: float = 10.0,
                 n_recenterings: int = 5, use_bundle_adjustment: bool = True, save_mean_color: bool = True,
                 coarse_scale: int = 4, fine_scale: int = 16, lane_width: int = 200, n_levels: int = 7,
                 use_gain_comp: bool = True, use_graphcut: bool = True, use_blending: bool = True,
                 detailed_log: bool = True, draw_inliers: bool = False
                 ) -> None:
        """
        Initialize the Stitcher with a matcher object and configuration parameters.
        Args:
            matcher (Matcher): An instance of Matcher class used for matching images.
            confidence_tr (float): Confidence threshold for matching. Defaults to 0.95.
            min_inliers (int): Minimum number of inliers required for a match. Defaults to 5.
            max_inliers (int): Maximum number of inliers to consider for a match. Defaults to 30.
            min_inlier_rate (float): Minimum inlier rate for matching. Defaults to 0.0.
            reproj_tr (float): Reprojection threshold for alignment. Defaults to 1.0.
            coarse_scale (int): Coarse scale factor for processing. Defaults to 4.
            fine_scale (int): Fine scale factor for processing. Defaults to 16.
            lane_width (int): Width of the lane for stitching. Defaults to 200.
            n_levels (int): Number of levels for multi-scale processing. Defaults to 7.
        """
        self.matcher = matcher

        self.transformation_type = transformation_type
        self.confidence_tr = confidence_tr
        self.min_inliers = min_inliers
        self.max_inliers = max_inliers
        self.min_inlier_rate = min_inlier_rate
        self.reproj_tr = reproj_tr
        self.n_recenterings = n_recenterings
        self.use_bundle_adjustment = use_bundle_adjustment

        self.use_gain_comp = use_gain_comp
        self.use_graphcut = use_graphcut
        self.use_blending = use_blending
        self.save_mean_color = save_mean_color
        self.coarse_scale = coarse_scale
        self.fine_scale = fine_scale
        self.lane_width = lane_width
        self.n_levels = n_levels

        self.detailed_log = detailed_log

    def _parse_dir(self, dir_path: Path) -> TileSet:
        """
        Parse a directory to create an ImageSet object for image files.
        Args:
            dir_path (Path): Path to the directory containing image files.
        Returns:
            ImageSet: An ImageSet object containing the order and dictionary of
                ImageStruct objects representing the images in the directory.
        """
        try:
            img_paths = [
                img_p
                for img_p in dir_path.iterdir()
                if img_p.suffix in (".jpg", ".png", ".tiff", ".TIF")
            ]

            img_paths.sort(key=lambda x: x.name)

            order, images = [], []
            for id, path in enumerate(img_paths):
                order.append(id)
                images.append(Tile(
                    id=id,
                    img_path=path,
                    inference_size=None,
                    _image=None, orig_size=None,
                    homography=None,
                    gain=None
                ))
        except Exception as e:
            logger.error(f"Error parsing directory: {e}")
            return None

        return TileSet(order=order, images=images)

    def _align(self, tile_set: TileSet, transformation_type: str = None,
               confidence_tr: bool = None, min_inliers: int = None,
               max_inliers: int = None, min_inlier_rate: float = None, reproj_tr: float = None,
               n_recenterings: int = None, use_bundle_adjustment: bool = None, detailed_log: bool = None
               ) -> StitchingData:
        """
        Align a set of images using matching and transformation techniques.
        Args:
            images (ImageSet): Set of images to be aligned, containing image data
                and processing order.
        Returns:
            StitchingData: Data object containing alignment information,
                including transformation matrices and panorama size.
        Raises:
            RuntimeError: If the alignment process fails due to issues in
                matching, homography estimation, or bundle adjustment.
        """
        transformation_type = transformation_type if transformation_type is not None else self.transformation_type
        confidence_tr = confidence_tr if confidence_tr is not None else self.confidence_tr
        min_inliers = min_inliers if min_inliers is not None else self.min_inliers
        max_inliers = max_inliers if max_inliers is not None else self.max_inliers
        min_inlier_rate = min_inlier_rate if min_inlier_rate is not None else self.min_inlier_rate
        reproj_tr = reproj_tr if reproj_tr is not None else self.reproj_tr
        n_recenterings = n_recenterings if n_recenterings is not None else self.n_recenterings
        use_bundle_adjustment = use_bundle_adjustment if \
            use_bundle_adjustment is not None else self.use_bundle_adjustment
        detailed_log = detailed_log if detailed_log is not None else self.detailed_log

        try:

            data = self.matcher.match(tile_set)

            data = matches_alignment(
                data, transformation_type, confidence_tr,
                min_inliers, max_inliers, min_inlier_rate, reproj_tr, n_recenterings
            )

            if use_bundle_adjustment:
                data = Optimizer(transformation_type, data).bundle_adjustment()

            data = translate_and_add_panorama_size(data)

            return data

        except Exception as e:
            logger.error(f"Alignment failed: {str(e)}")
            return None

    def _compose(self, data: StitchingData, use_gain_comp: bool = True, use_graphcut: bool = True,
                 coarse_scale: int = None, fine_scale: int = None, lane_width: int = None,
                 use_blending: bool = None,  n_levels: int = None, detailed_log: bool = None) -> Panorama:
        """
        Compose a panorama from aligned image data using gain compensation,
        graphcut, and blending techniques.
        Args:
            data (StitchingData): StitchingData object containing aligned image
                information, including transformations and matches.
        Returns:
            PanoramaData: Data object representing the composed panorama with the
                final image and canvas.
        Raises:
            RuntimeError: If the composition process fails due to issues in gain
                compensation, graphcut, or blending.
        """
        use_gain_comp = use_gain_comp if use_gain_comp is not None else self.use_gain_comp
        use_graphcut = use_graphcut if use_graphcut is not None else self.use_graphcut
        use_blending = use_blending if use_blending is not None else self.use_blending

        coarse_scale = coarse_scale if coarse_scale is not None else self.coarse_scale
        fine_scale = fine_scale if fine_scale is not None else self.fine_scale
        lane_width = lane_width if lane_width is not None else self.lane_width
        n_levels = n_levels if n_levels is not None else self.n_levels
        detailed_log = detailed_log if detailed_log is not None else self.detailed_log

        if use_blending and not use_graphcut:
            raise NotImplementedError

        try:
            if use_gain_comp:
                data = apply_gain_comp(data, save_mean_color=True)

            if use_graphcut:
                data = apply_graphcut(data, use_gains=use_gain_comp, coarse_scale=coarse_scale,
                                      fine_scale=fine_scale, lane_width=lane_width)

            if use_blending:
                panorama_data = apply_blending(data, n_levels=n_levels, use_gains=use_gain_comp)

            return panorama_data

        except Exception as e:
            logger.error(f"Composition failed: {str(e)}")
            return None

    def _stitch_full_pipline(self, tile_set: TileSet) -> Panorama:
        """
        Perform the full stitching pipeline to create a seamless panorama from
        input images.
        Args:
            images (ImageSet): Set of images to be stitched, containing image
                data and processing order.
        Returns:
            PanoramaData: Data object representing the final stitched panorama
                with the composed image and canvas.
        Raises:
            RuntimeError: If any step in the stitching pipeline (alignment or
                composition) fails.
        """
        try:
            alignment_data = self._align(tile_set)
            panorama_data = self._compose(alignment_data)
            return panorama_data
        except Exception as e:

            logger.error(f"Full stitching pipeline failed: {str(e)}")
            return None

    def _stitch_collage(self, tile_set: TileSet) -> Panorama:
        """
        Create a collage-style panorama from input images with minimal blending.
        Args:
            images (ImageSet): Set of images to be stitched into a collage,
                containing image data and processing order.
        Returns:
            PanoramaData: Data object representing the stitched collage panorama
                with the composed image and canvas.
        Raises:
            RuntimeError: If any step in the collage stitching process
                (alignment or collage creation) fails.
        """
        try:
            alignment_data = self._align(tile_set)
            panorama_data = make_collage(alignment_data)
            return panorama_data
        except Exception as e:

            logger.error(f"Collage stitching failed: {str(e)}")
            return None

    def _stitch_collage_no_optimize(self, tile_set: TileSet) -> Panorama:
        """
        Create a collage-style panorama from input images with minimal blending.
        Args:
            images (ImageSet): Set of images to be stitched into a collage,
                containing image data and processing order.
        Returns:
            PanoramaData: Data object representing the stitched collage panorama
                with the composed image and canvas.
        Raises:
            RuntimeError: If any step in the collage stitching process
                (alignment or collage creation) fails.
        """
        try:
            alignment_data = self._align(tile_set, use_bundle_adjustment=False)
            panorama_data = make_collage(alignment_data)
            return panorama_data
        except Exception as e:

            logger.error(f"Collage stitching failed: {str(e)}")
            return None

    def _stitch_compensated_collage(self, tile_set: TileSet) -> Panorama:
        """
        Perform the full stitching pipeline to create a seamless panorama from
        input images.
        Args:
            images (ImageSet): Set of images to be stitched, containing image
                data and processing order.
        Returns:
            PanoramaData: Data object representing the final stitched panorama
                with the composed image and canvas.
        Raises:
            RuntimeError: If any step in the stitching pipeline (alignment or
                composition) fails.
        """
        try:
            alignment_data = self._align(tile_set)
            data = apply_gain_comp(alignment_data)
            panorama_data = make_collage(data, use_gains=True)
            return panorama_data
        except Exception as e:

            logger.error(f"Full stitching pipeline failed: {str(e)}")
            return None

    def _stitch_compensated_mosaic(self, tile_set: TileSet) -> Panorama:
        """
        Perform the full stitching pipeline to create a seamless panorama from
        input images.
        Args:
            images (ImageSet): Set of images to be stitched, containing image
                data and processing order.
        Returns:
            PanoramaData: Data object representing the final stitched panorama
                with the composed image and canvas.
        Raises:
            RuntimeError: If any step in the stitching pipeline (alignment or
                composition) fails.
        """
        try:
            alignment_data = self._align(tile_set)
            data = apply_gain_comp(alignment_data)
            data = apply_graphcut(data)
            panorama_data = make_mosaic(data, use_gains=True)
            return panorama_data
        except Exception as e:

            logger.error(f"Full stitching pipeline failed: {str(e)}")
            return None

    def save_matches(self, tile_set: TileSet, output_file: Path) -> None:
        data = self.matcher.match(tile_set)
        Serializer().save(data, output_file)

    def stitch_with_loaded_matches(self, input_file: Path) -> TileSet:

        transformation_type = self.transformation_type
        confidence_tr = self.confidence_tr
        min_inliers = self.min_inliers
        max_inliers = self.max_inliers
        min_inlier_rate = self.min_inlier_rate
        reproj_tr = self.reproj_tr
        n_recenterings = self.n_recenterings

        data = Serializer().load(input_file)

        data = matches_alignment(
            data, transformation_type, confidence_tr, min_inliers,
            max_inliers, min_inlier_rate, reproj_tr, n_recenterings
        )

        data = Optimizer(transformation_type, data).bundle_adjustment()

        data = translate_and_add_panorama_size(data)

        panorama_data = make_collage(data)
        return panorama_data

    def _stitch_with_loaded_matches_draw_inliers(self, input_file: Path) -> TileSet:

        transformation_type = self.transformation_type
        confidence_tr = self.confidence_tr
        min_inliers = self.min_inliers
        max_inliers = self.max_inliers
        min_inlier_rate = self.min_inlier_rate
        reproj_tr = self.reproj_tr
        n_recenterings = self.n_recenterings

        data = Serializer().load(input_file)

        data = matches_alignment(
            data, transformation_type, confidence_tr, min_inliers,
            max_inliers, min_inlier_rate, reproj_tr, n_recenterings
        )

        data = Optimizer(transformation_type, data).bundle_adjustment()

        data = translate_and_add_panorama_size(data)

        panorama_data = make_collage_with_inliers(data)
        return panorama_data

    @log_time("Panorama done for", logger)
    def stitch(self, input_dir: Path, output_file: Path, cache_path: Path = None, mode: str = None) -> None:
        """
        Stitch images from a directory into a panorama with the specified mode
        and save the result to a file.
        Args:
            input_dir (Path): Directory path containing the images to be stitched
                (supports .jpg, .png, .tiff formats).
            output_file (Path): Path where the resulting panorama will be saved,
                including the file extension.
            mode (str): Stitching mode, can be 'full' (full pipeline with
                blending), 'auto' (automatic selection, defaults to full), or
                'collage' (minimal blending). Defaults to 'auto'.
        Returns:
            None
        Notes:
            Images are processed in a consistent order if frozen_order is True
                during parsing. The output format and quality depend on the file
                extension provided in output_file.
        """
        # mode = self.sticthing_mode if mode is None else mode
        tile_set = self._parse_dir(input_dir)
        match mode:
            case 'full' | 'auto':
                panorama_data = self._stitch_full_pipline(tile_set)
            case 'collage':
                panorama_data = self._stitch_collage(tile_set)
            case 'gaincomp collage':
                panorama_data = self._stitch_compensated_collage(tile_set)
            case 'mosaic':
                panorama_data = self._stitch_compensated_mosaic(tile_set)
            case 'save_matches':
                self.save_matches(tile_set, cache_path / 'matches.pkl')
                return
            case 'load_matches':
                panorama_data = self.stitch_with_loaded_matches(cache_path / 'matches.pkl')
            case 'collage_no_optimize':
                panorama_data = self._stitch_collage_no_optimize(tile_set)
            case 'load_matches_draw_inliers':
                panorama_data = self._stitch_with_loaded_matches_draw_inliers(cache_path / 'matches.pkl')
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        try:
            panorama_data.save_panorama(output_file)
        except Exception as e:
            logger.error(f"Failed to save panorama to {output_file}: {str(e)}")
            return

    @log_time("Total processing time:", logger)
    def process_collection(self, input_dir: Path, output_dir: Path, cache_dir: Path, mode: str = None) -> None:
        # mode = self.sticthing_mode if mode is None else mode
        datasets = [d for d in input_dir.iterdir() if d.is_dir()]
        datasets.sort(key=lambda path: path.name)

        with tqdm(datasets, desc="Datasets", position=0, leave=True, dynamic_ncols=True) as dataset_pbar:
            for dataset in dataset_pbar:
                dataset_pbar.set_postfix_str(f"{dataset.name}")
                logger.info(f"Processing dataset: {dataset.name}")

                series_list = [s for s in dataset.iterdir() if s.is_dir()]
                series_list.sort(key=lambda path: path.name)
                with tqdm(series_list, desc="Series", position=1, leave=False, dynamic_ncols=True) as series_pbar:
                    for series in series_pbar:
                        series_pbar.set_postfix_str(f"{series.name}")
                        logger.info(f"Processing series: {series.name}")

                        input_path = input_dir / dataset.name / series.name
                        output_path = output_dir / dataset.name / (series.name + ".jpg")
                        cache_path = cache_dir / dataset.name / series.name
                        Path(output_path.parent).mkdir(parents=True, exist_ok=True)

                        self.stitch(input_path, output_path, cache_path, mode)
