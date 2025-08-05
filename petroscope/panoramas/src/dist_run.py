import numpy as np
import torch
from pathlib import Path
import shutil
import cv2
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
    max_inliers: int = 30,
    reproj_tr: float = 1.0,
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
    Process datasets with sequential affine and projective optimization,
    creating panoramas for both without saving intermediate parameters.

    Args:
        tiles_root_dir: Root directory containing subdirectories with image datasets
        cache_root_dir: Directory containing cached matches
        output_pano_dir: Directory to store output panoramas
        device: PyTorch device for computations
        serializer: Object with load method for matches.pkl
        stitcher: Object with stitch method
        ... (other parameters same as in original code)

    Returns:
        List of tuples containing (output_filename, final_loss)
    """
    output_pano_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for cache_dir in cache_root_dir.iterdir():
        if not cache_dir.is_dir():
            continue

        tiles_dir = tiles_root_dir / cache_dir.name
        print(f"\nProcessing dataset: {cache_dir.name}")

        try:
            data = serializer.load(cache_dir / 'matches.pkl')
        except Exception as e:
            print(f"Failed to load matches.pkl from {cache_dir}: {e}")
            continue

        # Affine optimization
        try:
            data = matches_alignment(
                data,
                transformation_type='affine',
                confidence_tr=confidence_tr,
                min_inliers=min_inliers,
                max_inliers=max_inliers,
                min_inliers_rate=min_inliers_rate,
                reproj_tr=reproj_tr,
                n_iterations=n_recenterings
            )
        except Exception as e:
            print(f"Failed to align matches for affine: {e}")
            continue

        affine_optimizer = AffineDistortionOptimizer(
            device=device,
            data=data,
            f=f,
            cx=cx,
            cy=cy,
            freeze_principal_point=False,
            freeze_tangential=False
        )

        try:
            affine_loss = affine_optimizer.bundle_adjustment(
                lr_f=lr_f,
                lr_c=lr_c,
                lr_k1=lr_k1,
                lr_k2=lr_k2,
                lr_k3=lr_k3,
                lr_p=lr_p,
                h_gamma=h_gamma,
                d_gamma=d_gamma,
                max_iter=1000,
                plot=True,
                verbose='full'
            )
        except Exception as e:
            print(f"Affine bundle adjustment failed: {e}")
            continue

        # Get affine parameters
        affine_cm = affine_optimizer.get_camera_matrix_batch().squeeze().cpu().detach().numpy()
        affine_dp = affine_optimizer.get_distortion_params_batch().cpu().detach().numpy()[:, :5]

        # Undistort images with affine parameters
        affine_temp_dir = cache_dir / 'undistorted_affine_temp'
        try:
            undistort_dir(
                input_dir=tiles_dir,
                output_dir=affine_temp_dir,
                camera_matrix=affine_cm,
                distortion_params=affine_dp
            )
        except Exception as e:
            print(f"Affine undistortion failed: {e}")

        # Stitch affine panorama
        affine_pano_name = f"{cache_dir.name}_affine.jpg"
        affine_output_file = output_pano_dir / affine_pano_name
        try:
            stitcher.stitch(
                input_dir=affine_temp_dir,
                output_file=affine_output_file,
                cache_path=cache_dir,
                mode='collage'
            )
            print(f"Created affine panorama: {affine_output_file}")
            results.append((str(affine_output_file), affine_loss))
        except Exception as e:
            print(f"Affine stitching failed: {e}")
        finally:
            if affine_temp_dir.exists():
                shutil.rmtree(affine_temp_dir)

        projective_optimizer = DistortionOptimizer(
            device=device,
            data=data,
            f=affine_cm[0, 0],  # Use affine focal length as initial value
            cx=affine_cm[0, 2],
            cy=affine_cm[1, 2],
            k1=affine_dp[0, 0],
            k2=affine_dp[0, 1],
            k3=affine_dp[0, 4],
            p1=affine_dp[0, 2],
            p2=affine_dp[0, 3],
            freeze_principal_point=False,
            freeze_tangential=False
        )

        fine_coeff = 1e-3
        try:
            projective_loss = projective_optimizer.bundle_adjustment(
                lr_f=lr_f * fine_coeff,
                lr_c=lr_c * fine_coeff,
                lr_k1=lr_k1 * fine_coeff,
                lr_k2=lr_k2 * fine_coeff,
                lr_k3=lr_k3 * fine_coeff,
                lr_p=lr_p * fine_coeff,
                h_gamma=h_gamma,
                d_gamma=d_gamma,
                max_iter=5000,
                plot=True,
                verbose='full'
            )
        except Exception as e:
            print(f"Projective bundle adjustment failed: {e}")
            continue

        # Get projective parameters
        projective_cm = projective_optimizer.get_camera_matrix_batch().squeeze().cpu().detach().numpy()
        projective_dp = projective_optimizer.get_distortion_params_batch().cpu().detach().numpy()[:, :5]

        # Undistort images with projective parameters
        projective_temp_dir = cache_dir / 'undistorted_projective_temp'
        try:
            undistort_dir(
                input_dir=tiles_dir,
                output_dir=projective_temp_dir,
                camera_matrix=projective_cm,
                distortion_params=projective_dp
            )
        except Exception as e:
            print(f"Projective undistortion failed: {e}")

        # Stitch projective panorama
        projective_pano_name = f"{cache_dir.name}_projective.jpg"
        projective_output_file = output_pano_dir / projective_pano_name
        try:
            stitcher.stitch(
                input_dir=projective_temp_dir,
                output_file=projective_output_file,
                cache_path=cache_dir,
                mode='collage'
            )
            print(f"Created projective panorama: {projective_output_file}")
            results.append((str(projective_output_file), projective_loss))
        except Exception as e:
            print(f"Projective stitching failed: {e}")
        finally:
            if projective_temp_dir.exists():
                shutil.rmtree(projective_temp_dir)

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
            print(f"Failed to read {img_path}, skipping.")
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
        # print(f"Processed {img_path.name}")


if __name__ == '__main__':
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    matcher = Matcher(device=device, weights='../weights/loftr_outdoor.ckpt')
    stitcher = Stitcher(matcher=matcher)
    serializer = Serializer()

    tiles_metadir = Path('/home/g.nikolaev/data/tiles/RawLumenStone')
    cache_metadir = Path('/home/g.nikolaev/data/cache/RawLumenStone')
    output_pano_dir = Path('/home/g.nikolaev/data/panoramas/RawLumenStone_sequential')

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
