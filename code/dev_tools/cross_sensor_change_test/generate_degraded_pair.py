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


def paint_controlled_road(rgb: np.ndarray, row0: int, row1: int, col0: int, col1: int) -> np.ndarray:
    result = rgb.copy()
    neighborhood = result[max(0, row0 - 24):max(1, row0 - 4), col0:col1]
    local = np.median(neighborhood, axis=(0, 1)) if neighborhood.size else np.asarray([128] * 3)
    road_color = np.clip(local * 0.45 + 42, 0, 255).astype(np.uint8)
    result[row0:row1, col0:col1] = road_color
    center = (row0 + row1) // 2
    result[max(row0, center - 1):min(row1, center + 2), col0:col1] = np.clip(
        road_color.astype(np.int16) + 55, 0, 255
    ).astype(np.uint8)
    return result


def erase_controlled_corridor(rgb: np.ndarray, row0: int, row1: int, col0: int, col1: int) -> np.ndarray:
    result = rgb.copy()
    top = result[max(0, row0 - 24):max(1, row0 - 4), col0:col1]
    bottom = result[min(result.shape[0] - 1, row1 + 4):min(result.shape[0], row1 + 24), col0:col1]
    samples = [part.reshape(-1, 3) for part in (top, bottom) if part.size]
    fill = np.median(np.concatenate(samples, axis=0), axis=0) if samples else np.asarray([128] * 3)
    result[row0:row1, col0:col1] = np.clip(fill, 0, 255).astype(np.uint8)
    return result


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
    height, width = rgb.shape[:2]
    row0, row1 = int(height * 0.46), max(int(height * 0.46) + 5, int(height * 0.49))
    col0, col1 = int(width * 0.18), int(width * 0.82)
    truth_path = None
    before_source = rgb.copy()
    after_source = rgb.copy()
    if args.change != "none":
        # Both periods start from the same explicitly cleared corridor, so the
        # synthetic direction never depends on whatever happened to exist at a
        # fixed location in the source image.
        before_source = erase_controlled_corridor(before_source, row0, row1, col0, col1)
        after_source = erase_controlled_corridor(after_source, row0, row1, col0, col1)
        if args.change == "added":
            after_source = paint_controlled_road(after_source, row0, row1, col0, col1)
        else:
            before_source = paint_controlled_road(before_source, row0, row1, col0, col1)
        left, top = transform * (col0, row0)
        right, bottom = transform * (col1, row1)
        truth = gpd.GeoDataFrame(
            {"change_typ": [args.change]}, geometry=[box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))], crs=crs,
        )
        truth_path = output / f"truth_{args.change}.shp"
        truth.to_file(truth_path, driver="ESRI Shapefile", encoding="UTF-8")
    before = before_source
    after = degrade(
        after_source, scale=args.scale, blur=args.blur, brightness=args.brightness,
        contrast=args.contrast, gamma=args.gamma, channel_scale=tuple(args.channel_scale),
        noise=args.noise, seed=args.seed,
    )
    write_rgb(output / "A_original.tif", before, profile, mask)
    write_rgb(output / "B_degraded.tif", after, profile, mask)
    print(f"before={output / 'A_original.tif'}")
    print(f"after={output / 'B_degraded.tif'}")
    if truth_path:
        print(f"truth={truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
