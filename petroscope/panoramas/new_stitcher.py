from pathlib import Path
from classes import InputData, AlignData, MatchesData, OptimizeData, AlignmentData, GaincompData, MosaicData, PanoramaData
from matcher import Matcher
from align_functions import find_homographies

def bundle_adjustment(align_data):
    pass

def alignment(optimize_data):
    pass

def make_collage(optimize_data):
    pass

class Stitcher:
    def __init__(self, matcher: Matcher) -> None:
        """
        Initialize the Stitcher with a matcher object.
        
        Args:
            matcher: An instance of Matcher class used for matching images.
        """
        self.matcher = matcher

    def align(self, img_paths: list[Path]) -> AlignmentData:
        """
        Align a set of images based on their paths.
        
        Args:
            img_paths: List of file paths to the images to be aligned.
            
        Returns:
            AlignmentData: Data object containing alignment information for the images.
        """
        input_data = InputData(img_paths)
        matches_data = self.matcher.match(input_data)
        align_data = find_homographies(matches_data)
        optimize_data = bundle_adjustment(align_data)
        alignment_data = alignment(optimize_data)
        return alignment_data
    
    def compose(self, alignment_data: AlignmentData) -> PanoramaData:
        """
        Compose a panorama from aligned image data.
        
        Args:
            alignment_data: AlignmentData object containing aligned image information.
            
        Returns:
            PanoramaData: Data object representing the composed panorama.
        """
        gaincomp_data = GaincompData(alignment_data)
        mosaic_data = MosaicData(gaincomp_data)
        panorama = PanoramaData(mosaic_data)
        return panorama
    
    def stitch_full_pipline(self, img_paths: list[Path]) -> 'PanoramaData':
        """
        Perform the full stitching pipeline to create a panorama from input images.
        
        Args:
            img_paths: List of file paths to the images to be stitched.
            
        Returns:
            PanoramaData: Data object representing the final stitched panorama.
        """
        alignment_data = self.align(img_paths)
        panorama = self.compose(alignment_data)
        return panorama
    
    def stitch_collage(self, img_paths: list[Path]) -> PanoramaData:
        """
        Create a collage-style panorama from input images.
        
        Args:
            img_paths: List of file paths to the images to be stitched into a collage.
            
        Returns:
            PanoramaData: Data object representing the stitched collage panorama.
        """
        alignment_data = self.align(img_paths)
        panorama = make_collage(alignment_data)
        return panorama

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
        panorama.save(output_file)
