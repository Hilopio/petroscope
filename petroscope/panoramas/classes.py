from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class InputData:
    img_paths : list[Path]

@dataclass
class MatchesData:
    img_paths : list[Path]
    matches : list[np.ndarray]
    orig_sizes : list[np.ndarray]

@dataclass
class AlignData:
    img_paths : list[Path]
    transforms : list[np.ndarray]
    reference_idx : int
    inliers : list[list]

@dataclass
class OptimizeData:
    img_paths : list[Path]
    transforms : np.ndarray
    pivot : int

@dataclass
class AlignmentData:
    img_paths : list[Path]
    transforms : list[np.ndarray]
    reference_idx : int
    panorama_size : tuple

@dataclass
class GaincompData:
    img_paths : list[Path]
    alphas : list[int]

@dataclass
class MosaicData:
    img_paths : list[Path]
    canvas : np.ndarray

@dataclass
class PanoramaData:
    img_paths : list[Path]
    panorama : np.ndarray
