import cv2
import numpy as np
from PIL import Image
import pickle
import shutil
from pathlib import Path

borderValue = 0.0


def _load_images(img_paths):
    images = []
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255
        images.append(img)

    return images


def _load_transforms(transforms_file):
    with open(transforms_file, "rb") as f:
        loaded_data = pickle.load(f)
        transforms = loaded_data["transforms"]
        panorama_size = loaded_data["panorama_size"]
        img_paths = loaded_data["img_paths"]
    return transforms, panorama_size, img_paths


def _save(image, path):
    image = (image.clip(0, 1) * 255).astype('uint8')
    output_image = Image.fromarray(image)
    output_image.save(path, quality=95)


def _warp_collage(images, transforms, panorama_size):
    w, h = panorama_size
    panorama = np.full(shape=(h, w, 3), fill_value=borderValue, dtype='float32')

    for image, H in zip(images, transforms):
        cv2.warpPerspective(
            image,
            H,
            panorama_size,
            dst=panorama,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    return panorama


def _warp_img(img, H, panorama_size):
    warped_img = cv2.warpPerspective(
        img,
        H,
        panorama_size,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(borderValue, borderValue, borderValue)
    )
    return warped_img


def _warp_mask(mask, H, panorama_size):
    warped_mask = cv2.warpPerspective(
        mask.astype('float32'),
        H,
        panorama_size,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    warped_mask = (warped_mask == 1)
    return warped_mask


def _warp(image, H, panorama_size):
    warped_mask = _warp_mask(np.ones(image.shape[:-1], dtype=int), H, panorama_size)
    warped_img = _warp_img(image, H, panorama_size)
    return warped_img, warped_mask


def _warp_masked_collage(images, transforms, panorama_size, masks):
    n_images = len(images)
    panorama = np.zeros((*panorama_size[::-1], 3), dtype=np.float32)
    for i in range(n_images):
        warped_img = _warp_img(images[i], transforms[i], panorama_size)
        panorama = np.where(masks[i], warped_img, panorama)
    return panorama


def undistort_dir(
    input_dir: Path,
    output_dir: Path,
    camera_matrix: np.ndarray,
    distortion_params: np.ndarray
):
    """Undistort images in input_dir and save to output_dir."""
    # Проверка существования входной директории
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist.")

    # Если выходная директория совпадает с входной, работаем с копией списка файлов
    process_dir = input_dir
    if output_dir.exists() and not input_dir.samefile(output_dir):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Собираем список изображений
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.TIF']
    images = [img for ext in img_extensions for img in process_dir.glob(ext)]

    for img_path in images:
        # Чтение изображения
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}, skipping.")
            continue

        # Исправление дисторсии
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_params, (w, h), alpha=0)
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, distortion_params, None, new_camera_mtx, (roi[2], roi[3]), cv2.CV_32FC1
        )
        undistorted_roi = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        undistorted = cv2.resize(undistorted_roi, (w, h), interpolation=cv2.INTER_LINEAR)

        # Сохранение результата
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), undistorted)
