from pathlib import Path
from classes import ImageStruct, ImageSet, StitchingData, PanoramaData
from matcher import Matcher
from align_functions import find_homographies, alignment
from optimizer import Optimizer
from collage_functions import make_collage
from gain_compensation_functions import apply_gain_compensation
from graphcut_functions import apply_graphcut
from blending_functions import apply_blending

class Stitcher:
    """
    A class to handle the stitching of multiple images into a single panorama.
    
    This class orchestrates the process of image matching, alignment, and composition
    to create a seamless panorama from a set of input images using various techniques
    like homography estimation, bundle adjustment, gain compensation, graphcut, and blending.
    """
    def __init__(self, matcher: 'Matcher') -> None:
        """
        Initialize the Stitcher with a matcher object.
        
        Args:
            matcher: An instance of Matcher class used for matching images.
        """
        self.matcher = matcher

    def parse_dir(self, dir_path: Path) -> ImageSet:
        """
        Parse a directory to create an ImageSet object for image files.
        
        Args:
            dir_path (Path): Path to the directory containing image files.
        
        Returns:
            ImageSet: An ImageSet object containing the order and dictionary of ImageStruct objects representing the images in the directory.
        """
        img_paths = [
            img_p
            for img_p in dir_path.iterdir()
            if img_p.suffix in (".jpg", ".png", ".tiff")
        ]
        img_paths = sorted(img_paths)

        order, images = [], []
        for id, path in enumerate(img_paths):
            order.append(id)
            images.append(ImageStruct(id=id, img_path=path, image=None, orig_size=None, gain=None))

        return ImageSet(order=order, images=images)

    def align(self, images: 'ImageSet') -> 'StitchingData':
        """
        Align a set of images using matching and transformation techniques.
        
        Args:
            images (ImageSet): Set of images to be aligned, containing image data and processing order.
            
        Returns:
            StitchingData: Data object containing alignment information, including transformation matrices and panorama size.
        
        Raises:
            RuntimeError: If the alignment process fails due to issues in matching, homography estimation, or bundle adjustment.
        """
        try:
            data = self.matcher.match(images)
            data = find_homographies(data)
            data = Optimizer(data).bundle_adjustment()
            data = alignment(data)
            return data
        except Exception as e:
            raise RuntimeError(f"Alignment failed: {str(e)}")
    
    def compose(self, data: 'StitchingData') -> 'PanoramaData':
        """
        Compose a panorama from aligned image data using gain compensation, graphcut, and blending techniques.
        
        Args:
            data (StitchingData): StitchingData object containing aligned image information, including transformations and matches.
            
        Returns:
            PanoramaData: Data object representing the composed panorama with the final image and canvas.
        
        Raises:
            RuntimeError: If the composition process fails due to issues in gain compensation, graphcut, or blending.
        """
        try:
            data = apply_gain_compensation(data)
            data = apply_graphcut(data)
            panorama_data = apply_blending(data)
            return panorama_data
        except Exception as e:
            raise RuntimeError(f"Composition failed: {str(e)}")
    
    def stitch_full_pipline(self, images: 'ImageSet') -> 'PanoramaData':
        """
        Perform the full stitching pipeline to create a seamless panorama from input images.
        
        Args:
            images (ImageSet): Set of images to be stitched, containing image data and processing order.
            
        Returns:
            PanoramaData: Data object representing the final stitched panorama with the composed image and canvas.
        
        Raises:
            RuntimeError: If any step in the stitching pipeline (alignment or composition) fails.
        """
        try:
            alignment_data = self.align(images)
            panorama_data = self.compose(alignment_data)
            return panorama_data
        except Exception as e:
            raise RuntimeError(f"Full stitching pipeline failed: {str(e)}")
    
    def stitch_collage(self, images: 'ImageSet') -> 'PanoramaData':
        """
        Create a collage-style panorama from input images with minimal blending.
        
        Args:
            images (ImageSet): Set of images to be stitched into a collage, containing image data and processing order.
            
        Returns:
            PanoramaData: Data object representing the stitched collage panorama with the composed image and canvas.
        
        Raises:
            RuntimeError: If any step in the collage stitching process (alignment or collage creation) fails.
        """
        try:
            alignment_data = self.align(images)
            panorama_data = make_collage(alignment_data)
            return panorama_data
        except Exception as e:
            raise RuntimeError(f"Collage stitching failed: {str(e)}")

    def stitch(self, input_dir: Path, output_file: Path, mode: str = 'auto') -> None:
        """
        Stitch images from a directory into a panorama with the specified mode and save the result to a file.
        
        Args:
            input_dir (Path): Directory path containing the images to be stitched (supports .jpg, .png, .tiff formats).
            output_file (Path): Path where the resulting panorama will be saved, including the file extension.
            mode (str): Stitching mode, can be 'full' (full pipeline with blending), 'auto' (automatic selection, defaults to full), or 'collage' (minimal blending). Defaults to 'auto'.
        
        Returns:
            None
        
        Notes:
            Images are processed in a consistent order if frozen_order is True during parsing. The output format and quality depend on the file extension provided in output_file.
        """
        image_set = self.parse_dir(input_dir)
        match mode:
            case 'full' | 'auto':
                panorama_data = self.stitch_full_pipline(image_set)
            case 'collage':
                panorama_data = self.stitch_collage(image_set)
        
        panorama_data.save_panorama(output_file)
