from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageEnhance, ImageFilter
from shapely.geometry import box
import geopandas as gpd


def degrade(rgb: np.ndarray, *, scale: float, blur: float, brightness: float, contrast: float,
            gamma: float, channel_scale: tuple[float, float, float], noise: float, seed: int) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    if scale < 1.0:
        reduced = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
        image = reduced.resize(image.size, Image.Resampling.BILINEAR)
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur))
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    values = np.asarray(image, dtype=np.float32) / 255.0
    values = np.power(np.clip(values, 0, 1), gamma)
    values *= np.asarray(channel_scale, dtype=np.float32)[None, None, :]
    if noise > 0:
        values += np.random.default_rng(seed).normal(0, noise, values.shape)
    return np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)


def write_rgb(path: Path, rgb: np.ndarray, profile: dict, mask: np.ndarray) -> None:
    target = dict(profile)
    target.update(driver="GTiff", count=3, dtype="uint8", compress="deflate", nodata=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **target) as dataset:
        dataset.write(np.moveaxis(rgb, 2, 0))
        dataset.write_mask(mask)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create a geometry-preserving sensor-style A/B image pair.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale", type=float, default=0.55)
    parser.add_argument("--blur", type=float, default=1.2)
    parser.add_argument("--brightness", type=float, default=0.88)
    parser.add_argument("--contrast", type=float, default=0.78)
    parser.add_argument("--gamma", type=float, default=1.18)
    parser.add_argument("--channel-scale", type=float, nargs=3, default=(0.92, 1.03, 1.08))
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--change", choices=("none", "added", "removed"), default="none")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.input) as source:
        data = source.read()
        if data.shape[0] < 3:
            data = np.repeat(data[:1], 3, axis=0)
        rgb = np.moveaxis(data[:3], 0, 2)
        if rgb.dtype != np.uint8:
            finite = rgb[np.isfinite(rgb)]
            low, high = (np.percentile(finite, (2, 98)) if finite.size else (0, 1))
            rgb = np.clip((rgb - low) / max(float(high - low), 1e-9) * 255, 0, 255).astype(np.uint8)
        mask = source.dataset_mask()
        profile = source.profile
        transform, crs = source.transform, source.crs
    before = rgb.copy()
    after = degrade(
        rgb, scale=args.scale, blur=args.blur, brightness=args.brightness,
        contrast=args.contrast, gamma=args.gamma, channel_scale=tuple(args.channel_scale),
        noise=args.noise, seed=args.seed,
    )
    height, width = rgb.shape[:2]
    row0, row1 = int(height * 0.46), max(int(height * 0.46) + 5, int(height * 0.49))
    col0, col1 = int(width * 0.18), int(width * 0.82)
    truth_path = None
    if args.change != "none":
        if args.change == "added":
            local = np.median(after[max(0, row0 - 20):row0, col0:col1], axis=(0, 1))
            road_color = np.clip(local * 1.35 + 25, 0, 255).astype(np.uint8)
            after[row0:row1, col0:col1] = road_color
        else:
            local = np.median(before[max(0, row0 - 20):row0, col0:col1], axis=(0, 1))
            before[row0:row1, col0:col1] = np.clip(local, 0, 255).astype(np.uint8)
        left, top = transform * (col0, row0)
        right, bottom = transform * (col1, row1)
        truth = gpd.GeoDataFrame(
            {"change_typ": [args.change]}, geometry=[box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))], crs=crs,
        )
        truth_path = output / f"truth_{args.change}.shp"
        truth.to_file(truth_path, driver="ESRI Shapefile", encoding="UTF-8")
    write_rgb(output / "A_original.tif", before, profile, mask)
    write_rgb(output / "B_degraded.tif", after, profile, mask)
    print(f"before={output / 'A_original.tif'}")
    print(f"after={output / 'B_degraded.tif'}")
    if truth_path:
        print(f"truth={truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
