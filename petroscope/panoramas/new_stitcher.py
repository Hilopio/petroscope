from pathlib import Path
from classes import AlignData, MatchesData, OptimizeData, AlignmentData, GaincompData, MosaicData, PanoramaData
from matcher import Matcher
from align_functions import find_homographies, find_translation_and_panorama_size, alignment
from bundle_adjustment_functions import bundle_adjustment
from collage_functions import make_collage
from gain_compensation_functions import apply_gain_compensation
from graphcut_functions import apply_graphcut
from blending_functions import apply_blending


# ImageStructure + поле с коэффицентом компенсации и 2 проперти для картинок
# MatchStructure
# инлаеры <- мэтчи
# TransformsData
# PanoramaData два поля(панорама и канва) + проперти маски 
# + save для панорамы, канваса, альфа канала
# класс для параметров порядка панорамы
# папка кэша


class Stitcher:
    def __init__(self, matcher: 'Matcher') -> None:
        """
        Initialize the Stitcher with a matcher object.
        
        Args:
            matcher: An instance of Matcher class used for matching images.
        """
        self.matcher = matcher

    def align(self, img_paths: list[Path]) -> 'AlignmentData':
        """
        Align a set of images based on their paths.
        
        Args:
            img_paths: List of file paths to the images to be aligned.
            
        Returns:
            AlignmentData: Data object containing alignment information for the images.
        """
        try:
            matches_data = self.matcher.match(img_paths)
            align_data = find_homographies(matches_data)
            optimize_data = bundle_adjustment(align_data)
            alignment_data = alignment(optimize_data)
            return alignment_data
        except Exception as e:
            raise RuntimeError(f"Alignment failed: {str(e)}")
    
    def compose(self, alignment_data: 'AlignmentData') -> 'PanoramaData':
        """
        Compose a panorama from aligned image data.
        
        Args:
            alignment_data: AlignmentData object containing aligned image information.
            
        Returns:
            PanoramaData: Data object representing the composed panorama.
        """
        try:
            gaincomp_data = apply_gain_compensation(alignment_data)
            mosaic_data = apply_graphcut(gaincomp_data, alignment_data.transforms, alignment_data.panorama_size)
            panorama = apply_blending(mosaic_data, alignment_data.transforms, alignment_data.panorama_size)
            return panorama
        except Exception as e:
            raise RuntimeError(f"Composition failed: {str(e)}")
    
    def stitch_full_pipline(self, img_paths: list[Path]) -> 'PanoramaData':
        """
        Perform the full stitching pipeline to create a panorama from input images.
        
        Args:
            img_paths: List of file paths to the images to be stitched.
            
        Returns:
            PanoramaData: Data object representing the final stitched panorama.
        """
        try:
            alignment_data = self.align(img_paths)
            panorama = self.compose(alignment_data)
            return panorama
        except Exception as e:
            raise RuntimeError(f"Full stitching pipeline failed: {str(e)}")
    
    def stitch_collage(self, img_paths: list[Path]) -> 'PanoramaData':
        """
        Create a collage-style panorama from input images.
        
        Args:
            img_paths: List of file paths to the images to be stitched into a collage.
            
        Returns:
            PanoramaData: Data object representing the stitched collage panorama.
        """
        try:
            alignment_data = self.align(img_paths)
            panorama = make_collage(alignment_data)
            return panorama
        except Exception as e:
            raise RuntimeError(f"Collage stitching failed: {str(e)}")

    def stitch(self, img_paths: list[Path], output_file: Path, mode: str = 'auto') -> None:
        """
        Stitch images into a panorama with given modeand save the result to a file.
        
        Args:
            img_paths: List of file paths to the images to be stitched.
            output_file: Path where the resulting panorama will be saved.
            mode: Stitching mode, can be 'full', 'auto', or 'collage'. Defaults to 'auto'.
        """
        match mode:
            case 'full' | 'auto':
                panorama = self.stitch_full_pipline(img_paths)

            case 'collage':
                panorama = self.stitch_collage(img_paths)
        
        panorama.save(output_file)
