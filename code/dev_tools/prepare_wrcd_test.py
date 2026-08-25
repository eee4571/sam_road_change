from __future__ import annotations

"""Prepare the three-period WRCD validation data layout used by SamRoadChange."""

import argparse
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import shapes as raster_shapes
from rasterio.transform import from_origin
from shapely.geometry import Polygon, shape as shapely_shape


IMAGE_FILES = {
    "2012": "2012image.tif",
    "2014": "2014image.tif",
    "2016": "2016image.tif",
}
ROAD_FILES = {
    "2012": "2012label.tif",
    "2014": "2014label.tif",
    "2016": "2016label.tif",
}
CHANGE_FILES = {
    ("2012", "2014"): "change_label_2012to2014.tif",
    ("2014", "2016"): "change_label_2014to2016.tif",
}
EXPECTED_FILES = tuple(IMAGE_FILES.values()) + tuple(ROAD_FILES.values()) + tuple(CHANGE_FILES.values())
LABEL_FILES = frozenset(ROAD_FILES.values()) | frozenset(CHANGE_FILES.values())

DEFAULT_TILE_SIZE = 4096
DEFAULT_PIXEL_SIZE_METRES = 1.14
DEFAULT_CRS = CRS.from_epsg(3857)


@dataclass(frozen=True)
class RasterInfo:
    path: Path
    width: int
    height: int
    count: int
    transform: Affine
    crs: CRS | None

    @property
    def has_valid_georeference(self) -> bool:
        coefficients = tuple(self.transform)[:6]
        determinant = self.transform.a * self.transform.e - self.transform.b * self.transform.d
        return (
            self.crs is not None
            and all(math.isfinite(value) for value in coefficients)
            and math.isfinite(determinant)
            and not math.isclose(determinant, 0.0, abs_tol=1e-15)
        )


def _read_raster_info(path: Path) -> RasterInfo:
    try:
        with rasterio.open(path) as dataset:
            return RasterInfo(
                path=path,
                width=dataset.width,
                height=dataset.height,
                count=dataset.count,
                transform=dataset.transform,
                crs=dataset.crs,
            )
    except rasterio.errors.RasterioIOError as exc:
        raise ValueError(f"无法读取 TIFF：{path}\n{exc}") from exc


def _same_transform(left: Affine, right: Affine) -> bool:
    return bool(np.allclose(tuple(left)[:6], tuple(right)[:6], rtol=0.0, atol=1e-9))


def _same_georeference(info: RasterInfo, transform: Affine, crs: CRS) -> bool:
    return (
        info.has_valid_georeference
        and info.crs == crs
        and _same_transform(info.transform, transform)
    )


def _copy_with_georeference(
    source_path: Path,
    output_path: Path,
    transform: Affine,
    crs: CRS,
    *,
    width: int,
    height: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_path) as source:
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            width=width,
            height=height,
            transform=transform,
            crs=crs,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(output_path, "w", **profile) as target:
            for _block_index, window in source.block_windows(1):
                target.write(source.read(window=window), window=window)
                target.write_mask(source.dataset_mask(window=window), window=window)
            target.update_tags(**source.tags())
            for band in range(1, source.count + 1):
                target.update_tags(band, **source.tags(band))
            try:
                target.colorinterp = source.colorinterp
            except ValueError:
                # Some unusual TIFF color interpretations cannot be copied by GDAL.
                pass


def _binary_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        raster = dataset.read(1)
    return raster > 0


def _truth_polygons(
    before_road_path: Path,
    after_road_path: Path,
    change_path: Path,
    transform: Affine,
) -> list[Polygon]:
    before_road = _binary_mask(before_road_path)
    after_road = _binary_mask(after_road_path)
    np.logical_xor(before_road, after_road, out=before_road)
    del after_road
    changed = _binary_mask(change_path)
    np.logical_and(changed, before_road, out=changed)
    truth = changed
    del before_road
    if not truth.any():
        return []

    polygons: list[Polygon] = []
    for geometry, value in raster_shapes(
        truth.view(np.uint8), mask=truth, connectivity=8, transform=transform,
    ):
        if not value:
            continue
        polygon = shapely_shape(geometry)
        polygons.extend(_polygon_parts(polygon))
    return polygons


def _polygon_parts(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry] if geometry.area > 0 else []
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        parts: list[Polygon] = []
        for child in geometry.geoms:
            parts.extend(_polygon_parts(child))
        return parts
    return []


def _pixel_area_polygon(
    transform: Affine,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> Polygon:
    return Polygon(
        [
            transform * (col_start, row_start),
            transform * (col_stop, row_start),
            transform * (col_stop, row_stop),
            transform * (col_start, row_stop),
        ]
    )


def _write_polygon_shapefile(
    path: Path,
    polygons: Iterable[Polygon],
    crs: CRS,
    *,
    property_name: str,
    property_type: str,
    property_values: Iterable[int | str],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    polygon_list = list(polygons)
    value_list = list(property_values)
    if len(polygon_list) != len(value_list):
        raise ValueError(f"写入 {path} 时几何与属性数量不一致。")
    dtype = "int32" if property_type.startswith("int") else "object"
    frame = gpd.GeoDataFrame(
        {property_name: pd.Series(value_list, dtype=dtype)},
        geometry=gpd.GeoSeries(polygon_list, crs=crs),
        crs=crs,
    )
    frame.to_file(path, driver="ESRI Shapefile", encoding="UTF-8", engine="pyogrio")
    return len(frame)


def _clip_polygons(polygons: Iterable[Polygon], boundary: Polygon) -> list[Polygon]:
    boundary_bounds = boundary.bounds
    clipped: list[Polygon] = []
    for polygon in polygons:
        left, bottom, right, top = polygon.bounds
        if (
            right <= boundary_bounds[0]
            or top <= boundary_bounds[1]
            or left >= boundary_bounds[2]
            or bottom >= boundary_bounds[3]
        ):
            continue
        clipped.extend(_polygon_parts(polygon.intersection(boundary)))
    return clipped


def prepare_wrcd_test(source: Path, output: Path, *, tile_size: int = DEFAULT_TILE_SIZE) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if tile_size <= 0:
        raise ValueError("--tile-size 必须大于 0。")
    if not source.is_dir():
        raise ValueError(f"输入目录不存在：{source}")

    missing = [name for name in EXPECTED_FILES if not (source / name).is_file()]
    if missing:
        formatted = "\n".join(f"  - {name}" for name in missing)
        raise ValueError(f"输入目录缺少以下 WRCD 文件：\n{formatted}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录必须不存在或为空，避免覆盖已有数据：{output}")

    infos = {name: _read_raster_info(source / name) for name in EXPECTED_FILES}
    reference = infos[IMAGE_FILES["2012"]]
    if reference.count < 1:
        raise ValueError(f"影像没有可读取波段：{reference.path}")
    for filename in IMAGE_FILES.values():
        info = infos[filename]
        if info.count < 1:
            raise ValueError(f"TIFF 没有可读取波段：{info.path}")
        if (info.width, info.height) != (reference.width, reference.height):
            raise ValueError(
                "三期影像必须位于同一像素网格；"
                f"{info.path.name} 为 {info.width}x{info.height}，"
                f"预期 {reference.width}x{reference.height}。"
            )
    for filename in LABEL_FILES:
        info = infos[filename]
        if info.count < 1:
            raise ValueError(f"TIFF 没有可读取波段：{info.path}")
        # The published WRCD labels are one row and one column smaller than the
        # images.  They share the upper-left pixel grid, so pad only the absent
        # right/bottom edge with background instead of resampling the labels.
        if info.width not in {reference.width, reference.width - 1} or info.height not in {
            reference.height,
            reference.height - 1,
        }:
            raise ValueError(
                "标签必须与影像同尺寸，或仅在右侧/底部各少 1 pixel；"
                f"{info.path.name} 为 {info.width}x{info.height}，"
                f"影像为 {reference.width}x{reference.height}。"
            )

    georeferenced_images = [infos[name] for name in IMAGE_FILES.values() if infos[name].has_valid_georeference]
    used_fallback = not georeferenced_images
    if georeferenced_images:
        canonical_transform = georeferenced_images[0].transform
        canonical_crs = georeferenced_images[0].crs
        assert canonical_crs is not None
        for info in georeferenced_images[1:]:
            if not _same_georeference(info, canonical_transform, canonical_crs):
                raise ValueError(
                    "三期影像已有的 CRS/transform 不一致，无法在不重采样的情况下安全准备数据："
                    f"{georeferenced_images[0].path.name} 与 {info.path.name}"
                )
    else:
        canonical_transform = from_origin(
            0.0,
            reference.height * DEFAULT_PIXEL_SIZE_METRES,
            DEFAULT_PIXEL_SIZE_METRES,
            DEFAULT_PIXEL_SIZE_METRES,
        )
        canonical_crs = DEFAULT_CRS

    output.mkdir(parents=True, exist_ok=True)
    prepared_paths: dict[str, Path] = {}
    copied_files: list[str] = []
    copy_root = output / "_prepared_tiffs"
    for name, info in infos.items():
        same_size = (info.width, info.height) == (reference.width, reference.height)
        if same_size and _same_georeference(info, canonical_transform, canonical_crs):
            prepared_paths[name] = info.path
            continue
        prepared = copy_root / name
        _copy_with_georeference(
            info.path,
            prepared,
            canonical_transform,
            canonical_crs,
            width=reference.width,
            height=reference.height,
        )
        prepared_paths[name] = prepared.resolve()
        copied_files.append(name)

    truth_by_pair: dict[tuple[str, str], list[Polygon]] = {}
    for before, after in CHANGE_FILES:
        truth_by_pair[(before, after)] = _truth_polygons(
            prepared_paths[ROAD_FILES[before]],
            prepared_paths[ROAD_FILES[after]],
            prepared_paths[CHANGE_FILES[(before, after)]],
            canonical_transform,
        )

    row_starts = range(0, reference.height, tile_size)
    col_starts = range(0, reference.width, tile_size)
    area_count = math.ceil(reference.height / tile_size) * math.ceil(reference.width / tile_size)
    area_digits = max(2, len(str(area_count)))
    area_summaries = []
    area_number = 0
    for row_start in row_starts:
        for col_start in col_starts:
            area_number += 1
            area_id = f"area_{area_number:0{area_digits}d}"
            area_root = output / area_id
            boundary = _pixel_area_polygon(
                canonical_transform,
                row_start,
                min(row_start + tile_size, reference.height),
                col_start,
                min(col_start + tile_size, reference.width),
            )
            _write_polygon_shapefile(
                area_root / "01_验证区" / "boundary.shp",
                [boundary],
                canonical_crs,
                property_name="area_id",
                property_type="str:32",
                property_values=[area_id],
            )

            imagery_dir = area_root / "02_影像"
            imagery_dir.mkdir(parents=True, exist_ok=True)
            for period, filename in IMAGE_FILES.items():
                (imagery_dir / f"{period}.txt").write_text(
                    f"{prepared_paths[filename]}\n", encoding="utf-8",
                )

            truth_counts = {}
            truth_dir = area_root / "03_变化真值"
            for pair, polygons in truth_by_pair.items():
                clipped = _clip_polygons(polygons, boundary)
                truth_name = f"{pair[0]}_to_{pair[1]}.shp"
                truth_counts[f"{pair[0]}_to_{pair[1]}"] = _write_polygon_shapefile(
                    truth_dir / truth_name,
                    clipped,
                    canonical_crs,
                    property_name="id",
                    property_type="int",
                    property_values=range(1, len(clipped) + 1),
                )
            area_summaries.append({"area": area_id, "truth_features": truth_counts})

    return {
        "source": str(source),
        "output": str(output),
        "width": reference.width,
        "height": reference.height,
        "tile_size": tile_size,
        "area_count": area_count,
        "crs": canonical_crs.to_string(),
        "transform": tuple(canonical_transform)[:6],
        "fallback_georeference": used_fallback,
        "copied_tiffs": copied_files,
        "areas": area_summaries,
    }


def launch_gui() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("WRCD 测试数据快速准备")
    root.geometry("760x250")
    root.minsize(640, 230)

    source_var = tk.StringVar()
    output_var = tk.StringVar()
    status_var = tk.StringVar(value="选择包含 8 个 WRCD TIFF 的输入目录和一个空输出目录。")

    container = ttk.Frame(root, padding=18)
    container.pack(fill="both", expand=True)
    container.columnconfigure(1, weight=1)

    ttk.Label(container, text="输入目录").grid(row=0, column=0, padx=(0, 10), pady=8, sticky="w")
    source_entry = ttk.Entry(container, textvariable=source_var)
    source_entry.grid(row=0, column=1, pady=8, sticky="ew")

    def choose_source() -> None:
        selected = filedialog.askdirectory(title="选择 WRCD 输入目录", mustexist=True)
        if selected:
            source_var.set(selected)

    ttk.Button(container, text="浏览…", command=choose_source).grid(row=0, column=2, padx=(10, 0), pady=8)

    ttk.Label(container, text="输出目录").grid(row=1, column=0, padx=(0, 10), pady=8, sticky="w")
    output_entry = ttk.Entry(container, textvariable=output_var)
    output_entry.grid(row=1, column=1, pady=8, sticky="ew")

    def choose_output() -> None:
        selected = filedialog.askdirectory(title="选择空输出目录", mustexist=False)
        if selected:
            output_var.set(selected)

    ttk.Button(container, text="浏览…", command=choose_output).grid(row=1, column=2, padx=(10, 0), pady=8)

    progress = ttk.Progressbar(container, mode="indeterminate")
    progress.grid(row=2, column=0, columnspan=3, pady=(14, 8), sticky="ew")
    status_label = ttk.Label(container, textvariable=status_var, wraplength=700)
    status_label.grid(row=3, column=0, columnspan=3, pady=(0, 10), sticky="w")

    def set_running(running: bool) -> None:
        state = "disabled" if running else "normal"
        source_entry.configure(state=state)
        output_entry.configure(state=state)
        run_button.configure(state=state)
        if running:
            progress.start(12)
        else:
            progress.stop()

    def finish_success(summary: dict) -> None:
        set_running(False)
        status_var.set(f"完成：已生成 {summary['area_count']} 个验证区。")
        messagebox.showinfo(
            "准备完成",
            f"已生成 {summary['area_count']} 个验证区。\n\n"
            f"现在可在 GUI 中连接：\n{summary['output']}",
            parent=root,
        )

    def finish_error(message: str) -> None:
        set_running(False)
        status_var.set(f"失败：{message}")
        messagebox.showerror("准备失败", message, parent=root)

    def start() -> None:
        source_text = source_var.get().strip()
        output_text = output_var.get().strip()
        if not source_text or not output_text:
            messagebox.showwarning("路径不完整", "请同时指定输入目录和输出目录。", parent=root)
            return
        set_running(True)
        status_var.set("正在检查 TIFF、统一地理参考并生成验证区，请稍候…")

        def worker() -> None:
            try:
                summary = prepare_wrcd_test(Path(source_text), Path(output_text))
            except Exception as exc:  # GUI boundary: display all preparation failures.
                root.after(0, finish_error, str(exc))
            else:
                root.after(0, finish_success, summary)

        threading.Thread(target=worker, name="prepare-wrcd-test", daemon=True).start()

    run_button = ttk.Button(container, text="开始生成", command=start)
    run_button.grid(row=4, column=0, columnspan=3, pady=(2, 0))
    source_entry.focus_set()
    root.mainloop()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="将 WRCD 三期影像和标签整理为 SamRoadChange 可直接连接的验证项目。",
    )
    parser.add_argument("--source", help="包含 8 个 WRCD TIFF 的目录")
    parser.add_argument("--output", help="要生成的 SamRoadChange 验证项目目录")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"非重叠验证区边长（pixel，默认 {DEFAULT_TILE_SIZE}）",
    )
    args = parser.parse_args(argv)
    if not args.source and not args.output:
        return launch_gui()
    if not args.source or not args.output:
        parser.error("命令行运行时必须同时提供 --source 和 --output；不带参数运行则打开界面。")
    try:
        summary = prepare_wrcd_test(Path(args.source), Path(args.output), tile_size=args.tile_size)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"准备失败：{exc}\n")

    print(f"完成：{summary['area_count']} 个验证区 -> {summary['output']}")
    print(f"CRS：{summary['crs']}")
    print(f"transform：{summary['transform']}")
    if summary["copied_tiffs"]:
        print(f"已复制并统一地理参考的 TIFF：{len(summary['copied_tiffs'])} 个")
    else:
        print("原始 TIFF 地理参考一致，TXT 直接引用原始三期影像。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
