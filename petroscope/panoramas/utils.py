import cv2
import numpy as np
from PIL import Image
import pickle

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
