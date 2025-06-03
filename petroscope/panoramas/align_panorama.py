import argparse
from pathlib import Path

from alignment import Aligner
from utils import _load_images, _save, _warp_collage

if __name__ == '__main__':
    device = 'cuda:5'
    aligner = Aligner(device=device)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    assert input_dir.exists(), "Input directory does not exist."
    assert input_dir.is_dir(), "Input directory is not a directory."

    try:
        img_paths = [
            img_p
            for img_p in input_dir.iterdir()
            if img_p.suffix in (".jpg", ".png", ".tiff")
        ]

        transforms, panorama_size, img_paths = aligner.only_transforms(img_paths=img_paths)
        pics = _load_images(img_paths)
        panorama_ans = _warp_collage(pics, transforms, panorama_size)

        _save(panorama_ans, output_file)

    except Exception as e:
        print(e)
