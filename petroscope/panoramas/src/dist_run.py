import torch
import numpy as np
from pathlib import Path
from serializer import Serializer
from matcher import Matcher
from stitcher import Stitcher
import shutil
import cv2
import os

from align_functions import matches_alignment
from diatortion_optimizer import DistortionOptimizer


def undistort_dir(
    input_dir: Path,
    output_dir: Path,
    camera_matrix: np.ndarray,
    distortion_params: np.ndarray
):
    # Пересоздаем выходную папку
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


def undistort_image(image_path, camera_matrix, dist_coeffs, output_path=None):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    camera_matrix = np.array(camera_matrix, dtype=np.float32)
    dist_coeffs = np.array(dist_coeffs, dtype=np.float32)

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1.0
    )

    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_undistorted.jpg"

    cv2.imwrite(output_path, undistorted_img)
    return output_path


confidence_tr = 0.95
min_inliers = 5
max_inliers = 200
min_inlier_rate = 0.0
reproj_tr = 10.0
n_recenterings = 25

f = 1e5
cx = 3396 / 2
cy = 2547 / 2

input_dir = Path('/home/g.nikolaev/data/tiles/RawLumenStone')
cache_dir = Path('/home/g.nikolaev/data/cache/RawLumenStone')
intermediate_dir = Path('/home/g.nikolaev/data/tiles/RawLumenStone_custom_udist')
output_dir = Path('/home/g.nikolaev/data/panoramas/RawLumenStone')

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
matcher = Matcher(device=device, weights='../weights/loftr_outdoor.ckpt')
stitcher = Stitcher(matcher=matcher)

for series in input_dir.iterdir():
    if series.is_dir():

        data = Serializer().load(cache_dir / series.name / 'matches.pkl')
        data = matches_alignment(
            data, confidence_tr, min_inliers, max_inliers,
            min_inlier_rate, reproj_tr, n_recenterings
        )

        optimizer = DistortionOptimizer(data, f=f, cx=cx, cy=cy)

        lr_f = 1239.9967836846104
        lr_c = 3.585612610345396
        lr_k1 = 0.07556810141274425
        lr_k2 = 0.001260466458564947
        lr_k3 = 5.727904470799619e-07
        lr_p = 0.0003795853142670637

        h_gamma = 0.9
        d_gamma = 0.9

        result = optimizer.bundle_adjustment(
            lr_f=lr_f, lr_c=lr_c, lr_k1=lr_k1,
            lr_k2=lr_k2, lr_k3=lr_k3, lr_p=lr_p,
            h_gamma=h_gamma, d_gamma=d_gamma,
            max_iter=5000, plot=True, verbose='full'
        )

        cm = optimizer.get_camera_matrix().cpu().detach().numpy()
        dp = optimizer.get_distortion_params().cpu().detach().numpy()[:, :5]

        print(cm)
        print(dp)

        undistort_dir(
            input_dir=input_dir / series.name,
            output_dir=intermediate_dir / series.name,
            camera_matrix=cm,
            distortion_params=dp
        )

        stitcher.stitch(
            input_dir=intermediate_dir / series.name,
            output_file=output_dir / f'{series.name}.jpg',
            cache_path=None,
            mode='collage'
        )

        undistort_image(
            image_path=intermediate_dir / series.name / '1.1.jpg',
            camera_matrix=cm,
            dist_coeffs=dp,
            output_path=intermediate_dir / series.name / '1.1_udistorted.jpg'
        )
