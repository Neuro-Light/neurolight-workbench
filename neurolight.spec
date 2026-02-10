# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Neurolight. Reproducible: run from repo root with
#   uv run pyinstaller neurolight.spec
# or
#   pyinstaller neurolight.spec
# after installing the project and PyInstaller.

from pathlib import Path

# Repo root (PyInstaller runs with cwd = directory containing this spec)
REPO_ROOT = Path.cwd()

# Entry script; imports use package layout under src/
script = REPO_ROOT / 'src' / 'main.py'
assert script.exists(), f"Entry script not found: {script}"

# Bundle assets if present (icons, etc.)
assets_dir = REPO_ROOT / 'assets'
datas = []
if assets_dir.is_dir():
    for item in assets_dir.rglob('*'):
        if item.is_file() and item.suffix.lower() in ('.png', '.ico', '.svg', '.qrc'):
            rel = item.relative_to(REPO_ROOT)
            datas.append((str(item), str(rel.parent)))

a = Analysis(
    [str(script)],
    pathex=[str(REPO_ROOT / 'src')],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='neurolight',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
