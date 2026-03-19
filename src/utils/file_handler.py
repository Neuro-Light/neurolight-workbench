from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from core.experiment_manager import Experiment


class ImageStackHandler:
    def __init__(self) -> None:
        self.files: List[str] = []
        self._experiment: Optional[Experiment] = None

    def load_image_stack(self, directory_or_files) -> List[str]:
        paths: List[str] = []
        if isinstance(directory_or_files, (list, tuple)):
            for p in directory_or_files:
                if str(p).lower().endswith((".tif", ".tiff")):
                    paths.append(str(p))
        else:
            base = Path(directory_or_files)
            if base.is_dir():
                # Case-insensitive filter for .tif/.tiff files
                for p in sorted(base.iterdir()):
                    if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
                        paths.append(str(p))
        self.files = paths
        return self.files

    def validate_tif_files(self, file_paths: List[str]) -> bool:
        return all(str(p).lower().endswith((".tif", ".tiff")) for p in file_paths)

    def get_image_count(self) -> int:
        return len(self.files)

    def get_image_at_index(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self.files):
            raise IndexError("Image index out of range")
        path = self.files[index]
        suffix = Path(path).suffix.lower()

        # Fast-path TIFFs: tifffile is typically faster and preserves 16-bit well.
        if suffix in (".tif", ".tiff"):
            try:
                import tifffile  # local import keeps base startup fast

                arr = tifffile.imread(path)
                return np.asarray(arr)
            except Exception:
                # Fall back to Pillow below
                pass

        with Image.open(path) as img:
            return np.asarray(img)

    def get_all_frames_as_array(self) -> Optional[np.ndarray]:
        """Load all frames as a 3D numpy array (frames, height, width).
        Reuses approach from Jupyter notebook.
        Preserves original image dtype to avoid precision loss (consistent with get_image_at_index).
        """
        if not self.files:
            return None

        def _load_one(file_path: str) -> np.ndarray:
            # NOTE: We avoid calling get_image_at_index (index lookup) inside threads.
            suffix = Path(file_path).suffix.lower()
            if suffix in (".tif", ".tiff"):
                try:
                    import tifffile

                    img_arr = tifffile.imread(file_path)
                    pixels = np.asarray(img_arr)
                except Exception:
                    with Image.open(file_path) as img:
                        pixels = np.asarray(img)
            else:
                with Image.open(file_path) as img:
                    pixels = np.asarray(img)

            if pixels.ndim == 2:
                return pixels
            if pixels.ndim == 3:
                # Convert RGB/RGBA to grayscale quickly
                return pixels.mean(axis=2)
            return pixels

        # Threaded IO improves total-load time for folder stacks.
        # (This is usually IO-bound + native decoding that releases the GIL.)
        try:
            from concurrent.futures import ThreadPoolExecutor

            max_workers = min(8, len(self.files))
            if max_workers <= 1:
                frames = [_load_one(p) for p in self.files]
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    frames = list(ex.map(_load_one, self.files))
        except Exception:
            frames = [_load_one(p) for p in self.files]

        return np.asarray(frames)

    def associate_with_experiment(self, experiment: Experiment) -> None:
        self._experiment = experiment
        # keeps that path and loads the path to images when loading an expirement
        experiment.image_count = len(self.files)
        if self.files:
            experiment.image_stack_path = str(Path(self.files[0]).parent)
            # Save the actual list of selected files
            experiment.image_stack_files = self.files.copy()
