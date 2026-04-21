"""Tests for GifGenerator — GIF creation, frame management, and optimize copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.gif_generator import GifGenerator


def test_generate_gif_creates_file(tmp_path: Path) -> None:
    out = str(tmp_path / "out.gif")
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    result = GifGenerator().generate_gif(frames, out, fps=5)
    assert result == out
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_generate_gif_fps_clamped_to_one(tmp_path: Path) -> None:
    out = str(tmp_path / "out.gif")
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    GifGenerator().generate_gif(frames, out, fps=0)
    assert Path(out).exists()


def test_add_frame_appends() -> None:
    gen = GifGenerator(fps=15)
    assert len(gen.frames) == 0
    gen.add_frame(np.zeros((2, 2), dtype=np.uint8))
    gen.add_frame(np.ones((2, 2), dtype=np.uint8))
    assert len(gen.frames) == 2


def test_init_stores_fps() -> None:
    gen = GifGenerator(fps=24)
    assert gen.fps == 24


def test_optimize_gif_copies_content(tmp_path: Path) -> None:
    src = tmp_path / "src.gif"
    dst = tmp_path / "dst.gif"
    src.write_bytes(b"GIFDATA123")
    GifGenerator().optimize_gif(str(src), str(dst))
    assert dst.read_bytes() == b"GIFDATA123"
