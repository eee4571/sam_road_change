"""Windows 中文路径兼容层，由本项目内的 Python 子进程自动加载。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_gis_data() -> None:
    """Use the GIS databases actually bundled in this portable runtime."""
    runtime_root = Path(__file__).resolve().parent.parent / "runtime"
    site_packages = runtime_root / "env" / "samroad_env" / "Lib" / "site-packages"
    # Rasterio/GDAL in this bundle expects the newer database layout shipped
    # with rasterio. Pyproj can consume it too; the reverse combination fails.
    proj_data = site_packages / "rasterio" / "proj_data"
    gdal_data = site_packages / "rasterio" / "gdal_data"
    if (proj_data / "proj.db").is_file():
        os.environ["PROJ_DATA"] = str(proj_data)
        os.environ["PROJ_LIB"] = str(proj_data)
    if gdal_data.is_dir():
        os.environ["GDAL_DATA"] = str(gdal_data)


def _patch_opencv() -> None:
    if sys.platform != "win32":
        return
    try:
        import cv2
        import numpy as np
    except ImportError:
        return

    original_imread = cv2.imread
    original_imwrite = cv2.imwrite

    def imread(filename, flags=cv2.IMREAD_COLOR):
        path = os.fspath(filename)
        try:
            with open(path, "rb") as stream:
                encoded = np.frombuffer(stream.read(), dtype=np.uint8)
            if encoded.size:
                image = cv2.imdecode(encoded, flags)
                if image is not None:
                    return image
        except (OSError, ValueError):
            pass
        return original_imread(path, flags)

    def imwrite(filename, image, params=None):
        path = os.fspath(filename)
        extension = os.path.splitext(path)[1] or ".png"
        encode_params = [] if params is None else params
        try:
            ok, encoded = cv2.imencode(extension, image, encode_params)
            if ok:
                with open(path, "wb") as stream:
                    stream.write(encoded.tobytes())
                return True
        except (OSError, ValueError, cv2.error):
            pass
        return original_imwrite(path, image, encode_params)

    cv2.imread = imread
    cv2.imwrite = imwrite


_configure_gis_data()
_patch_opencv()
