from pathlib import Path
import shutil
import numpy as np
from logger import logger, log_time

from classes import Tile, TileSet, StitchingData, Panorama

from matcher import Matcher
from align_functions import matches_alignment, translate_and_add_panorama_size
from optimizer import Optimizer
from distortion_optimizer import DistortionOptimizer
from collage_functions import make_collage, make_mosaic

from gain_comp_functions import apply_gain_comp
from graphcut_functions import apply_graphcut
from blending_functions import apply_blending

from serializer import Serializer

from utils import undistort_dir


class Stitcher:
    """
    A class to handle the stitching of multiple images into a single panorama.
    This class orchestrates the process of image matching, alignment, and composition
    to create a seamless panorama from a set of input images using various techniques
    like homography estimation, bundle adjustment, gain compensation, graphcut, and
    blending.
    """
    def __init__(self, matcher: Matcher, load_matches: bool = False, transformation_type: str = "projective",
                 confidence_tr: float = 0.95, min_inliers: int = 5,
                 max_inliers: int = 30, min_inlier_rate: float = 0.0, reproj_tr: float = 1.0,
                 n_recenterings: int = 5, use_BA: bool = True, save_mean_color: bool = True,
                 coarse_scale: int = 4, fine_scale: int = 16, lane_width: int = 200, n_levels: int = 7,
                 use_gain_comp: bool = True, use_graphcut: bool = True, use_blending: bool = True,
                 detailed_log: bool = True, draw_inliers: bool = False, draw_connections: bool = False,
                 custom_undistortion: bool = False, n_undistortions: int = 1, stitching_mode: str = "collage"
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
        self.device = matcher.device
        self.matcher = matcher

        self.load_matches = load_matches

        self.transformation_type = transformation_type
        self.confidence_tr = confidence_tr
        self.min_inliers = min_inliers
        self.max_inliers = max_inliers
        self.min_inlier_rate = min_inlier_rate
        self.reproj_tr = reproj_tr
        self.n_recenterings = n_recenterings
        self.use_BA = use_BA

        self.use_gain_comp = use_gain_comp
        self.use_graphcut = use_graphcut
        self.use_blending = use_blending
        self.save_mean_color = save_mean_color
        self.coarse_scale = coarse_scale
        self.fine_scale = fine_scale
        self.lane_width = lane_width
        self.n_levels = n_levels

        self.detailed_log = detailed_log
        self.draw_inliers = draw_inliers
        self.draw_connections = draw_connections
        self.custom_undistortion = custom_undistortion
        self.n_undistortions = n_undistortions
        self.stitching_mode = stitching_mode

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
                if img_p.suffix in (".jpg", ".png", ".tiff", ".tif", ".TIF")
            ]

            img_paths.sort(key=lambda x: x.name)

            order, images = [], []
            for id, path in enumerate(img_paths):
                order.append(id)
                images.append(Tile(
                    id=id,
                    img_path=path,
                    _image=None,
                    _tensor=None,
                    orig_size=None,
                    homography=None,
                    gain=None
                ))
        except Exception as e:
            logger.error(f"Error parsing directory: {e}")
            return None

        return TileSet(order=order, images=images)

    def _align(self, data: StitchingData, transformation_type: str = None,
               confidence_tr: bool = None, min_inliers: int = None,
               max_inliers: int = None, min_inlier_rate: float = None, reproj_tr: float = None,
               n_recenterings: int = None, use_BA: bool = None, detailed_log: bool = None
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
        use_BA = use_BA if use_BA is not None else self.use_BA
        detailed_log = detailed_log if detailed_log is not None else self.detailed_log

        try:
            data = matches_alignment(
                data, transformation_type, confidence_tr,
                min_inliers, max_inliers, min_inlier_rate, reproj_tr, n_recenterings
            )

            if use_BA:
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

    def _stitch_full_pipline(self, data: StitchingData) -> Panorama:
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
            alignment_data = self._align(data)
            panorama_data = self._compose(alignment_data)
            return panorama_data
        except Exception as e:

            logger.error(f"Full stitching pipeline failed: {str(e)}")
            return None

    def _stitch_collage(self, data: StitchingData) -> Panorama:
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
            alignment_data = self._align(data)

            panorama_data = make_collage(
                alignment_data,
                use_gains=False,
                draw_inliers=self.draw_inliers,
                draw_connections=self.draw_connections
            )
            return panorama_data
        except Exception as e:

            logger.error(f"Collage stitching failed: {str(e)}")
            return None

    def _stitch_compensated_collage(self, data: StitchingData) -> Panorama:
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
            alignment_data = self._align(data)
            data = apply_gain_comp(alignment_data)
            panorama_data = make_collage(data, use_gains=True)
            return panorama_data
        except Exception as e:

            logger.error(f"Full stitching pipeline failed: {str(e)}")
            return None

    def _stitch_compensated_mosaic(self, data: StitchingData) -> Panorama:
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
            alignment_data = self._align(data)
            data = apply_gain_comp(alignment_data)
            data = apply_graphcut(data)
            panorama_data = make_mosaic(data, use_gains=True)
            return panorama_data
        except Exception as e:

            logger.error(f"Full stitching pipeline failed: {str(e)}")
            return None

    def _CUBA(  # Custom Undistortion Bundle Adjustment
        self,
        data: StitchingData,
    ) -> TileSet:

        transformation_type = self.transformation_type
        confidence_tr = self.confidence_tr
        min_inliers = self.min_inliers
        max_inliers = self.max_inliers
        min_inlier_rate = self.min_inlier_rate
        reproj_tr = self.reproj_tr
        n_recenterings = self.n_recenterings

        # lr_f: float = 1239.9967836846104
        lr_log_f: float = 1e-2
        lr_c: float = 3.585612610345396
        lr_k1: float = 0.07556810141274425
        lr_k2: float = 0.001260466458564947
        lr_k3: float = 5.727904470799619e-07
        lr_p: float = 0.0003795853142670637
        h_gamma: float = 0.9
        d_gamma: float = 0.9

        data = matches_alignment(
            data, transformation_type, confidence_tr, min_inliers,
            max_inliers, min_inlier_rate, reproj_tr, n_recenterings
        )

        f = 1e4
        some_id = data.tile_set.order[0]
        w, h = data.tile_set.images[some_id].orig_size
        cx, cy = w // 2, h // 2

        d_optimizer = DistortionOptimizer('affine', self.device, data, f=f, cx=cx, cy=cy)
        try:
            data = d_optimizer.bundle_adjustment(
                # lr_f=lr_f,
                lr_log_f=lr_log_f,
                lr_c=lr_c,
                lr_k1=lr_k1,
                lr_k2=lr_k2,
                lr_k3=lr_k3,
                lr_p=lr_p,
                h_gamma=h_gamma,
                d_gamma=d_gamma,
                max_iter=2000,
                verbose='core'
            )
        except Exception as e:
            print(f"Affine bundle adjustment failed: {e}")

        affine_cm = d_optimizer.get_camera_matrix_batch().squeeze().cpu().detach().numpy()
        affine_dp = d_optimizer.get_distortion_params_batch().cpu().detach().numpy()[:, :5]

        return affine_cm, affine_dp

    def preproccess_undictortion(
        self,
        tiles_dir: Path,
        cache_dir: Path,
        n_undistortions: int = None
    ) -> tuple[np.ndarray, np.ndarray]:

        current_dir = Path(str(tiles_dir))
        tmp_dir = cache_dir / 'tmp'
        for _ in range(n_undistortions):

            tile_set = self._parse_dir(current_dir)
            data = self.matcher.match(tile_set)
            camera_matrix, distortion_params = self._CUBA(data)

            if iter == n_undistortions - 1:
                break

            undistort_dir(current_dir, tmp_dir, camera_matrix, distortion_params)
            current_dir = tmp_dir

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        return camera_matrix, distortion_params

    @log_time("Panorama done for", logger)
    def stitch(self, tiles_dir: Path, output_file: Path, cache_path: Path = None,
               load_matches: bool = None, mode: str = None) -> None:
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
        if self.custom_undistortion:
            camera_matrix, distortion_params = self.preproccess_undictortion(
                tiles_dir,
                cache_path,
                n_undistortions=self.n_undistortions
            )
            tmp_dir = cache_path / 'tmp'
            undistort_dir(tiles_dir, tmp_dir, camera_matrix, distortion_params)
            tiles_dir = tmp_dir

        load_matches = self.load_matches if load_matches is None else load_matches
        if load_matches:
            matches_dir = cache_path / 'matches.pkl'
            data = Serializer().load(matches_dir)
        else:
            tile_set = self._parse_dir(tiles_dir)
            data = self.matcher.match(tile_set)

        mode = self.stitching_mode if mode is None else mode
        match mode:
            case 'save_matches':
                matches_dir = cache_path / 'matches.pkl'
                Serializer().save(data, matches_dir)
                return
            case 'full' | 'auto':
                panorama_data = self._stitch_full_pipline(data)
            case 'collage':
                panorama_data = self._stitch_collage(data)
            case 'gaincomp collage':
                panorama_data = self._stitch_compensated_collage(data)
            case 'mosaic':
                panorama_data = self._stitch_compensated_mosaic(data)
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        try:
            panorama_data.save_panorama(output_file)
        except Exception as e:
            logger.error(f"Failed to save panorama to {output_file}: {str(e)}")

        if self.custom_undistortion:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
