from classes import Tile, Match, StitchingData, TileSet
import json
from pathlib import Path
import numpy as np


class Serializer:
    def save(self, obj, path: Path) -> None:
        """
        Save a StitchingData object to the specified path.
        Metadata is saved in JSON format, and numpy arrays are saved as separate binary files using np.save.
        If the directory already exists, it will be deleted along with all its contents before being recreated.

        Args:
            obj: The StitchingData object to serialize.
            path (Path): The directory path where the data will be saved.
        """
        if not isinstance(obj, StitchingData):
            raise ValueError("Serializer can only save StitchingData objects")

        # If the directory exists, delete it along with all contents
        if path.exists():
            import shutil
            shutil.rmtree(path, ignore_errors=True)

        # Create a new empty directory
        path.mkdir(parents=True, exist_ok=True)

        # Prepare metadata dictionary
        metadata = {
            "type": "StitchingData",
            "reper_idx": obj.reper_idx,
            "panorama_size": list(obj.panorama_size) if obj.panorama_size is not None else None,
            "tile_set": self._serialize_tile_set(obj.tile_set, path),
            "matches": self._serialize_matches(obj.matches, path),
            "canvas": self._save_array(obj.canvas, path / "canvas.npy") if obj.canvas is not None else None
        }

        # Save metadata to JSON
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: Path) -> StitchingData:
        """
        Load a StitchingData object from the specified path.
        Reads metadata from JSON and numpy arrays from binary files.

        Args:
            path (Path): The directory path from where to load the data.

        Returns:
            StitchingData: The deserialized StitchingData object.
        """
        if not (path / "metadata.json").exists():
            raise FileNotFoundError(f"No metadata.json found in {path}")

        # Load metadata from JSON
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        if metadata["type"] != "StitchingData":
            raise ValueError("Loaded metadata is not for a StitchingData object")

        # Reconstruct the StitchingData object
        tile_set = (
            self._deserialize_tile_set(metadata["tile_set"], path)
            if "tile_set" in metadata
            else TileSet(order=[], images={})
        )

        matches = self._deserialize_matches(metadata["matches"], path) if "matches" in metadata else []
        canvas = None
        if "canvas" in metadata and metadata["canvas"] is not None:
            canvas_path = path / metadata["canvas"]
            if canvas_path.exists():
                canvas = self._load_array(canvas_path)

        panorama_size = None
        if "panorama_size" in metadata and metadata["panorama_size"] is not None:
            panorama_size = tuple(metadata["panorama_size"])

        reper_idx = metadata.get("reper_idx", 0)
        return StitchingData(
            tile_set=tile_set,
            matches=matches,
            reper_idx=reper_idx,
            panorama_size=panorama_size,
            canvas=canvas
        )

    def _serialize_tile_set(self, tile_set: TileSet, base_path: Path) -> dict:
        """Serialize a TileSet object."""
        tiles_path = base_path / "tiles"
        tiles_path.mkdir(exist_ok=True)

        tiles_dict = {}
        for tile_id in tile_set.order:
            tile = tile_set.images[tile_id]
            tile_data = {
                "id": tile.id,
                "img_path": str(tile.img_path),
                "orig_size": tile.orig_size.tolist(),
                "homography": self._save_array(tile.homography, tiles_path / f"tile_{tile_id}_homography.npy"),
                "gain": self._save_array(tile.gain, tiles_path / f"tile_{tile_id}_gain.npy")
            }
            tiles_dict[tile_id] = tile_data

        return {
            "order": tile_set.order,
            "images": tiles_dict
        }

    def _deserialize_tile_set(self, tile_set_data: dict, base_path: Path) -> TileSet:
        """Deserialize a TileSet object."""
        tiles = {}
        for tile_id, tile_data in tile_set_data["images"].items():
            tiles[int(tile_id)] = Tile(
                id=tile_data["id"],
                img_path=Path(tile_data["img_path"]),
                _image=None,  # Image is not saved, will be loaded on demand
                _tensor=None,
                orig_size=np.array(tile_data["orig_size"]),
                homography=self._load_array(base_path / "tiles" / tile_data["homography"]),
                gain=self._load_array(base_path / "tiles" / tile_data["gain"])
            )

        return TileSet(order=tile_set_data["order"], images=tiles)

    def _serialize_matches(self, matches: list[Match], base_path: Path) -> list:
        """Serialize a list of Match objects."""
        matches_path = base_path / "matches"
        matches_path.mkdir(exist_ok=True)

        matches_data = []
        for idx, match in enumerate(matches):
            match_data = {
                "i": match.i,
                "j": match.j,
                "xy_i": self._save_array(match.xy_i, matches_path / f"match_{idx}_xy_i.npy"),
                "xy_j": self._save_array(match.xy_j, matches_path / f"match_{idx}_xy_j.npy"),
                "conf": self._save_array(match.conf, matches_path / f"match_{idx}_conf.npy")
            }
            matches_data.append(match_data)

        return matches_data

    def _deserialize_matches(self, matches_data: list, base_path: Path) -> list[Match]:
        """Deserialize a list of Match objects."""
        matches = []
        for match_data in matches_data:
            match = Match(
                i=match_data["i"],
                j=match_data["j"],
                xy_i=self._load_array(base_path / "matches" / match_data["xy_i"]),
                xy_j=self._load_array(base_path / "matches" / match_data["xy_j"]),
                conf=self._load_array(base_path / "matches" / match_data["conf"])
            )
            matches.append(match)

        return matches

    def _save_array(self, arr: np.ndarray, filepath: Path) -> str:
        """Save a numpy array to a file and return the relative filepath as a string."""
        np.save(filepath, arr)
        return filepath.name

    def _load_array(self, filepath: Path) -> np.ndarray:
        """Load a numpy array from a file."""
        return np.load(filepath, allow_pickle=True)
