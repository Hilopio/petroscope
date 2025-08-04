import numpy as np
import torch
from pathlib import Path
import shutil
import cv2
from itertools import product
from typing import List, Tuple


from matcher import Matcher
from stitcher import Stitcher
from serializer import Serializer

from align_functions import matches_alignment
from distortion_optimizer import DistortionOptimizer, AffineDistortionOptimizer


def process_datasets(
    tiles_root_dir: Path,
    cache_root_dir: Path,
    output_pano_dir: Path,
    device: torch.device,
    serializer: Serializer,
    stitcher: Stitcher,
    min_inliers: int = 5,
    confidence_tr: float = 0.95,
    min_inliers_rate: float = 0.0,
    n_recenterings: int = 25,
    max_inliers: int = 200,
    reproj_tr: float = 10.0,
    f: float = 1e5,
    cx: float = 3396/2,
    cy: float = 2547/2,
    lr_f: float = 1239.9967836846104,
    lr_c: float = 3.585612610345396,
    lr_k1: float = 0.07556810141274425,
    lr_k2: float = 0.001260466458564947,
    lr_k3: float = 5.727904470799619e-07,
    lr_p: float = 0.0003795853142670637,
    h_gamma: float = 0.9,
    d_gamma: float = 0.9
) -> List[Tuple[str, float]]:
    """
    Process multiple datasets with different optimization configurations,
    apply undistortion, and create panoramas.

    Args:
        input_root_dir: Root directory containing subdirectories with image datasets
        output_pano_dir: Directory to store output panoramas
        device: PyTorch device for computations
        serializer: Object with load method for matches.pkl
        stitcher: Object with stitch method
        ... (other parameters same as in original code)

    Returns:
        List of tuples containing (output_filename, final_loss)
    """
    # Create output directory
    output_pano_dir.mkdir(parents=True, exist_ok=True)

    # Configuration combinations
    freeze_principal_points = [True, False]
    freeze_tangentials = [True, False]
    transformation_types = ['affine', 'projective']

    results = []

    # Iterate through all subdirectories
    for cache_dir in cache_root_dir.iterdir():
        if not cache_dir.is_dir():
            continue

        tiles_dir = tiles_root_dir / cache_dir.name

        print(f"\nProcessing dataset: {cache_dir.name}")

        # Load data

        # Process each configuration combination
        for freeze_pp, freeze_tan, trans_type in product(
            freeze_principal_points,
            freeze_tangentials,
            transformation_types
        ):
            print(f"\nConfiguration: {trans_type}, freeze_pp={freeze_pp}, freeze_tan={freeze_tan}")

            try:
                data = serializer.load(cache_dir / 'matches.pkl')
            except Exception as e:
                print(f"Failed to load matches.pkl from {cache_dir}: {e}")
                continue

            max_iter = 1000 if trans_type == 'affine' else 5000

            # Align matches
            try:
                data = matches_alignment(
                    data,
                    transformation_type=trans_type,
                    confidence_tr=confidence_tr,
                    min_inliers=min_inliers,
                    max_inliers=max_inliers,
                    min_inliers_rate=min_inliers_rate,
                    reproj_tr=reproj_tr,
                    n_iterations=n_recenterings
                )
            except Exception as e:
                print(f"Failed to align matches: {e}")
                continue

            # Initialize optimizer
            optimizer_class = AffineDistortionOptimizer if trans_type == 'affine' else DistortionOptimizer
            optimizer = optimizer_class(
                device=device,
                data=data,
                f=f,
                cx=cx,
                cy=cy,
                freeze_principal_point=freeze_pp,
                freeze_tangential=freeze_tan
            )

            # Run bundle adjustment
            try:
                final_loss = optimizer.bundle_adjustment(
                    lr_f=lr_f,
                    lr_c=lr_c,
                    lr_k1=lr_k1,
                    lr_k2=lr_k2,
                    lr_k3=lr_k3,
                    lr_p=lr_p,
                    h_gamma=h_gamma,
                    d_gamma=d_gamma,
                    max_iter=max_iter,
                    plot=True,
                    verbose='full'
                )
            except Exception as e:
                print(f"Bundle adjustment failed: {e}")
                continue

            # Get camera parameters
            cm = optimizer.get_camera_matrix_batch().squeeze().cpu().detach().numpy()
            dp = optimizer.get_distortion_params_batch().cpu().detach().numpy()[:, :5]

            # Create temporary directory for undistorted images
            temp_dir = cache_dir / 'undistorted_temp'
            # Undistort images
            try:
                undistort_dir(
                    input_dir=tiles_dir,
                    output_dir=temp_dir,
                    camera_matrix=cm,
                    distortion_params=dp
                )
            except Exception as e:
                print(f"Undistortion failed: {e}")

            # Create panorama filename
            pano_name = (
                f"{cache_dir.name}_"
                f"{trans_type}_"
                f"pp{'F' if freeze_pp else 'U'}_"
                f"tan{'F' if freeze_tan else 'U'}.jpg"
            )
            output_file = output_pano_dir / pano_name

            # Stitch panorama
            try:
                stitcher.stitch(
                    input_dir=temp_dir,
                    output_file=output_file,
                    cache_path=cache_dir,
                    mode='collage'
                )
                print(f"Created panorama: {output_file}")
                results.append((str(output_file), final_loss))
            except Exception as e:
                print(f"Stitching failed: {e}")
            finally:
                # Clean up temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

    return results


def undistort_dir(
    input_dir: Path,
    output_dir: Path,
    camera_matrix: np.ndarray,
    distortion_params: np.ndarray
):
    """Undistort images in input_dir and save to output_dir"""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    images = []
    for ext in img_extensions:
        images.extend(input_dir.glob(ext))

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Не удалось прочитать {img_path}, пропускаем.")
            continue

        h, w = img.shape[:2]

        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_params, (w, h), alpha=0)
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, distortion_params, None, new_camera_mtx, (roi[2], roi[3]), cv2.CV_32FC1
        )
        undistorted_roi = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        undistorted = cv2.resize(undistorted_roi, (w, h), interpolation=cv2.INTER_LINEAR)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), undistorted)
        print(f"Обработано {img_path.name}")


if __name__ == '__main__':
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    matcher = Matcher(device=device, weights='../weights/loftr_outdoor.ckpt')
    stitcher = Stitcher(matcher=matcher)
    serializer = Serializer()

    tiles_metadir = Path('/home/g.nikolaev/data/tiles/RawLumenStone')
    cache_metadir = Path('/home/g.nikolaev/data/cache/RawLumenStone')
    output_pano_dir = Path('/home/g.nikolaev/data/panoramas/RawLumenStone_metasearch')

    results = process_datasets(
        tiles_metadir,
        cache_metadir,
        output_pano_dir,
        device,
        serializer,
        stitcher
    )

    with open(output_pano_dir / 'results.txt', 'w') as f:
        for result in results:
            f.write(f"{result[0]}\t{result[1]}\n")
