from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PySide6.QtCore import QThread, Signal

"""PyInstaller frozen apps on macOS (and sometimes Windows) have known issues with
ProcessPoolExecutor + spawn: infinite process loops, multiple GUI windows
pickling failures. Disable multiprocessing when frozen."""
_FROZEN = getattr(sys, "frozen", False)


# Below this threshold, process-spawn overhead exceeds the parallelism gain.
_MIN_FRAMES_FOR_MP = 10


# ------------------------------------------------------------------
# Module-level worker functions (must be top-level for pickling
# with the 'spawn' start method used on Windows / macOS).
# ------------------------------------------------------------------


def _register_pair(ref_frame, moving_frame, transform_type):
    """Register *moving_frame* to *ref_frame*.  Runs in a worker process."""
    from pystackreg import StackReg

    sr = StackReg(transform_type)
    return sr.register(ref_frame, moving_frame)


def _transform_frame(frame, tmat, transform_type):
    """Apply *tmat* to a single frame.  Runs in a worker process."""
    from pystackreg import StackReg

    sr = StackReg(transform_type)
    result = sr.transform(frame, tmat=tmat)
    # Convert back to uint16 to reduce IPC serialization size.
    # Values are in 0-65535 range (input was uint16); clipping handles
    # any minor interpolation overshoot.
    np.clip(result, 0, 65535, out=result)
    return result.astype(np.uint16)


class AlignmentWorker(QThread):
    """Worker thread that runs the image alignment pipeline off the main thread."""

    progress = Signal(int, int, str)  # (completed, total, message)
    finished = Signal(object, object, object)  # (aligned_stack, tmats, confidence_scores)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        image_stack: np.ndarray,
        transform_type: str = "RIGID_BODY",
        reference: str = "first",
        parent=None,
    ):
        super().__init__(parent)
        self._image_stack = image_stack
        self._transform_type = transform_type
        self._reference = reference
        self._cancel_requested = False

    # ------------------------------------------------------------------
    # Public API (called from main thread)
    # ------------------------------------------------------------------
    def request_cancel(self) -> None:
        """Request cancellation. The worker will stop at the next checkpoint."""
        self._cancel_requested = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_cancel(self, stage: str) -> bool:
        """Return True if cancellation was requested (caller should return)."""
        if self._cancel_requested:
            self.progress.emit(0, 0, f"Cancelled during: {stage}")
            self.cancelled.emit()
            return True
        return False

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: C901
        executor = None
        try:
            from pystackreg import StackReg

            image_stack = self._image_stack
            num_frames = image_stack.shape[0]

            if self._check_cancel("initialization"):
                return

            # Map transform type string to StackReg constant
            transform_map = {
                "translation": StackReg.TRANSLATION,
                "rigid_body": StackReg.RIGID_BODY,
                "scaled_rotation": StackReg.SCALED_ROTATION,
                "affine": StackReg.AFFINE,
                "bilinear": StackReg.BILINEAR,
            }
            transform_const = transform_map.get(self._transform_type.lower(), StackReg.RIGID_BODY)

            sr = StackReg(transform_const)

            # ----------------------------------------------------------
            # Vectorized normalization
            # ----------------------------------------------------------
            self.progress.emit(0, num_frames, "Normalizing image stack...")
            global_min = float(np.min(image_stack))
            global_max = float(np.max(image_stack))
            global_range = global_max - global_min

            if global_range > 0:
                image_stack_uint16 = (
                    (image_stack.astype(np.float32) - global_min) / global_range * 65535.0
                ).astype(np.uint16)
            else:
                image_stack_uint16 = image_stack.astype(np.uint16)

            if self._check_cancel("normalization"):
                return

            # ----------------------------------------------------------
            # Decide whether to use multiprocessing
            # ----------------------------------------------------------
            use_mp = not _FROZEN and num_frames >= _MIN_FRAMES_FOR_MP
            if use_mp:
                n_workers = max(1, min(os.cpu_count() or 1, num_frames))
                # 'spawn' is the only start method safe with Qt on all
                # platforms (fork can deadlock on macOS / crash on Windows).
                ctx = multiprocessing.get_context("spawn")
                executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)

            # ----------------------------------------------------------
            # Registration
            # ----------------------------------------------------------
            # Parallel registration is only correct for 'first' and
            # 'mean' modes where every frame is registered against the
            # same fixed reference.  'previous' mode has inter-frame
            # dependencies so it falls back to the sequential C call.
            if executor is not None and self._reference in ("first", "mean"):
                tmats = self._register_parallel(
                    executor, image_stack_uint16, transform_const, num_frames
                )
                if tmats is None:
                    return  # cancelled
            elif self._reference in ("first", "mean"):
                tmats = self._register_sequential(
                    sr, image_stack_uint16, transform_const, num_frames
                )
                if tmats is None:
                    return  # cancelled
            else:
                self.progress.emit(0, num_frames, "Computing transformation matrices...")
                tmats = sr.register_stack(image_stack_uint16, reference=self._reference)

            if self._check_cancel("registration"):
                return

            # ----------------------------------------------------------
            # Transformation
            # ----------------------------------------------------------
            if executor is not None:
                aligned_stack_uint16 = self._transform_parallel(
                    executor, image_stack_uint16, tmats, transform_const, num_frames
                )
                if aligned_stack_uint16 is None:
                    return  # cancelled
            else:
                aligned_stack_uint16 = self._transform_sequential(
                    sr, image_stack_uint16, tmats, transform_const, num_frames
                )
                if aligned_stack_uint16 is None:
                    return  # cancelled

            del image_stack_uint16

            if self._check_cancel("transformation"):
                return

            # ----------------------------------------------------------
            # Vectorized de-normalization
            # ----------------------------------------------------------
            self.progress.emit(0, num_frames, "De-normalizing aligned stack...")

            if global_range > 0:
                aligned_float = (
                    aligned_stack_uint16.astype(np.float32) * (global_range / 65535.0) + global_min
                )
                del aligned_stack_uint16

                dtype = image_stack.dtype
                if dtype == np.uint8:
                    aligned_stack = np.clip(aligned_float, 0, 255).astype(np.uint8)
                elif dtype == np.uint16:
                    aligned_stack = np.clip(aligned_float, 0, 65535).astype(np.uint16)
                else:
                    aligned_stack = aligned_float.astype(dtype)

                del aligned_float
            else:
                del aligned_stack_uint16
                aligned_stack = image_stack.copy()

            if self._check_cancel("de-normalization"):
                return

            # ----------------------------------------------------------
            # Confidence scores (NCC per frame)
            # ----------------------------------------------------------
            self.progress.emit(0, num_frames, "Calculating confidence scores...")

            confidence_scores: list[float] = []
            mean_reference_frame = None
            if self._reference == "mean":
                mean_reference_frame = np.mean(aligned_stack.astype(np.float32), axis=0)

            for i in range(num_frames):
                if self._check_cancel("confidence calculation"):
                    return

                if self._reference == "first":
                    if i == 0:
                        confidence_scores.append(1.0)
                        continue
                    reference_frame = aligned_stack[0]
                elif self._reference == "previous":
                    if i == 0:
                        confidence_scores.append(1.0)
                        continue
                    reference_frame = aligned_stack[i - 1]
                elif self._reference == "mean":
                    reference_frame = mean_reference_frame
                else:
                    reference_frame = aligned_stack[0]

                self.progress.emit(
                    i,
                    num_frames,
                    f"Calculating confidence for frame {i + 1}/{num_frames}...",
                )

                ref_float = reference_frame.astype(np.float32)
                ali_float = aligned_stack[i].astype(np.float32)

                ref_norm = (ref_float - ref_float.mean()) / (ref_float.std() + 1e-10)
                ali_norm = (ali_float - ali_float.mean()) / (ali_float.std() + 1e-10)
                ncc = float(np.mean(ref_norm * ali_norm))

                confidence_scores.append(max(0.0, min(1.0, ncc)))

            self.progress.emit(num_frames, num_frames, "Alignment complete!")
            self.finished.emit(aligned_stack, tmats, confidence_scores)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if executor is not None:
                executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Parallel helpers
    # ------------------------------------------------------------------

    def _register_parallel(self, executor, stack, transform_const, num_frames):
        """Register every frame against a fixed reference in parallel."""
        self.progress.emit(0, num_frames - 1, "Registering frames (parallel)...")

        if self._reference == "mean":
            ref_frame = np.mean(stack.astype(np.float32), axis=0)
        else:  # 'first'
            ref_frame = stack[0]

        futures = {}
        for i in range(1, num_frames):
            if self._cancel_requested:
                self.cancelled.emit()
                return None
            f = executor.submit(_register_pair, ref_frame, stack[i], transform_const)
            futures[f] = i

        # Collect results as they complete
        results = {}
        completed = 0
        for f in as_completed(futures):
            if self._cancel_requested:
                self.cancelled.emit()
                return None
            results[futures[f]] = f.result()
            completed += 1
            self.progress.emit(
                completed,
                num_frames - 1,
                f"Registered {completed}/{num_frames - 1} frames...",
            )

        # Assemble the (num_frames, R, C) tmats array
        sample = next(iter(results.values()))
        tmats = np.zeros((num_frames,) + sample.shape)
        tmats[0] = np.eye(sample.shape[0], sample.shape[1])
        for idx, tmat in results.items():
            tmats[idx] = tmat

        return tmats

    def _transform_parallel(self, executor, stack, tmats, transform_const, num_frames):
        """Apply per-frame transformations in parallel."""
        self.progress.emit(0, num_frames, "Transforming frames (parallel)...")

        futures = {}
        for i in range(num_frames):
            if self._cancel_requested:
                self.cancelled.emit()
                return None
            f = executor.submit(_transform_frame, stack[i], tmats[i], transform_const)
            futures[f] = i

        results = [None] * num_frames
        completed = 0
        for f in as_completed(futures):
            if self._cancel_requested:
                self.cancelled.emit()
                return None
            idx = futures[f]
            results[idx] = f.result()
            completed += 1
            self.progress.emit(
                completed,
                num_frames,
                f"Transformed {completed}/{num_frames} frames...",
            )

        return np.stack(results)

    def _register_sequential(self, sr, stack, transform_const, num_frames):
        """Register every frame against a fixed reference, one at a time (with progress)."""
        if self._reference == "mean":
            ref_frame = np.mean(stack.astype(np.float32), axis=0)
        else:  # 'first'
            ref_frame = stack[0]

        if num_frames <= 1:
            tmat = sr.register(ref_frame, ref_frame)  # get shape
            tmats = np.zeros((num_frames,) + tmat.shape)
            tmats[0] = np.eye(tmat.shape[0], tmat.shape[1])
            return tmats

        # Identity for frame 0; register frames 1..n-1
        first_tmat = sr.register(ref_frame, stack[1])
        tmats = np.zeros((num_frames,) + first_tmat.shape)
        tmats[0] = np.eye(first_tmat.shape[0], first_tmat.shape[1])
        tmats[1] = first_tmat

        for i in range(2, num_frames):
            if self._check_cancel("registration"):
                return None
            self.progress.emit(
                i,
                num_frames - 1,
                f"Registering frame {i + 1}/{num_frames}...",
            )
            tmats[i] = sr.register(ref_frame, stack[i])

        return tmats

    def _transform_sequential(self, sr, stack, tmats, transform_const, num_frames):
        """Apply per-frame transformations one at a time (with progress)."""
        results = []
        for i in range(num_frames):
            if self._check_cancel("transformation"):
                return None
            self.progress.emit(
                i + 1,
                num_frames,
                f"Transforming frame {i + 1}/{num_frames}...",
            )
            result = sr.transform(stack[i], tmat=tmats[i])
            np.clip(result, 0, 65535, out=result)
            results.append(result.astype(np.uint16))

        return np.stack(results)
