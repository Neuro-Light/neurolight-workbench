from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
from PIL import Image

from core.experiment_manager import Experiment

_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$")


def _extract_valid_time(raw: object) -> Optional[str]:
    """
    Normalize metadata time values and return only valid HH:MM or HH:MM:SS strings.
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", errors="ignore")
        except Exception:
            return None
    text = str(raw).strip()
    if not text:
        return None
    # EXIF commonly stores "YYYY:MM:DD HH:MM:SS" - use the time token.
    candidate = text.split()[-1] if " " in text else text
    if _TIME_RE.fullmatch(candidate):
        return candidate
    return None


def _get_exif_timestamp(file_path: str) -> Optional[str]:
    """
    Try to extract a DateTimeOriginal (or DateTime) timestamp from TIFF/image EXIF.

    Returns an ISO-format time string "HH:MM:SS" on success, or None if unavailable.
    """
    try:
        with Image.open(file_path) as img:
            exif_data = img._getexif() if hasattr(img, "_getexif") else None
            if exif_data:
                # EXIF tag 36867 = DateTimeOriginal, 306 = DateTime
                for tag_id in (36867, 306):
                    raw = exif_data.get(tag_id)
                    parsed = _extract_valid_time(raw)
                    if parsed is not None:
                        return parsed
    except Exception:
        # Best-effort EXIF read failed; continue to tifffile-based fallback below.
        pass
    # Fallback: check TIFF ImageDescription or DateTime tag via tifffile
    try:
        import tifffile

        with tifffile.TiffFile(file_path) as tif:
            for tag in tif.pages[0].tags.values():
                if tag.name in ("DateTime", "DateTimeOriginal"):
                    parsed = _extract_valid_time(tag.value)
                    if parsed is not None:
                        return parsed
    except Exception:
        # Best-effort TIFF metadata extraction failed; treat timestamp as unavailable.
        pass
    return None


class ImageStackHandler:
    def __init__(self) -> None:
        self.files: List[str] = []
        self._experiment: Optional[Experiment] = None
        self._excluded_frames: Set[int] = set()

    def set_excluded_frames(self, indices: Set[int]) -> None:
        self._excluded_frames = set(indices)

    def get_excluded_frames(self) -> Set[int]:
        return set(self._excluded_frames)

    def get_total_frame_count(self) -> int:
        """Return total number of files regardless of exclusions."""
        return len(self.files)

    def get_included_files(self) -> List[str]:
        """Return the file list with excluded frames removed, in original order."""
        if not self._excluded_frames:
            return list(self.files)
        return [p for i, p in enumerate(self.files) if i not in self._excluded_frames]

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
        self._excluded_frames = set()
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
        """Load all non-excluded frames as a 3D numpy array (frames, height, width).
        Preserves original image dtype to avoid precision loss (consistent with get_image_at_index).
        """
        if not self.files:
            return None

        included_files = [p for i, p in enumerate(self.files) if i not in self._excluded_frames]
        if not included_files:
            return None

        def _load_one(file_path: str) -> np.ndarray:
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
                return pixels.mean(axis=2)
            return pixels

        try:
            from concurrent.futures import ThreadPoolExecutor

            max_workers = min(8, len(included_files))
            if max_workers <= 1:
                frames = [_load_one(p) for p in included_files]
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    frames = list(ex.map(_load_one, included_files))
        except Exception:
            frames = [_load_one(p) for p in included_files]

        return np.asarray(frames)

    def associate_with_experiment(self, experiment: Experiment) -> None:
        self._experiment = experiment
        # keeps that path and loads the path to images when loading an expirement
        experiment.image_count = len(self.files)
        if self.files:
            experiment.image_stack_path = str(Path(self.files[0]).parent)
            # Save the actual list of selected files
            experiment.image_stack_files = self.files.copy()
            # Try to extract EXIF timestamp from the first image; store only if found
            start_time = _get_exif_timestamp(self.files[0])
            if start_time is not None:
                if "acquisition" not in experiment.settings or not isinstance(
                    experiment.settings.get("acquisition"), dict
                ):
                    experiment.settings["acquisition"] = {}
                acquisition = experiment.settings["acquisition"]
                existing_start = acquisition.get("experiment_start_time")
                if existing_start is None or (isinstance(existing_start, str) and not existing_start.strip()):
                    acquisition["experiment_start_time"] = start_time
