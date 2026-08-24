import argparse
import os
from datetime import datetime
from pathlib import Path

from input_catalog import read_path_list

import cv2
import numpy as np
import torch
try:
    import rasterio
except ImportError:
    rasterio = None

from networks.sam_multi_lora import (
    build_sam_vit_b_adapter_linknet_multi_lora,
    resize_model_pos_embed,
)


ROOT_DIR = Path(__file__).resolve().parent
RUNTIME_MODELS_ROOT = Path(
    os.environ.get("SAMROAD_MODELS_ROOT", ROOT_DIR.parents[2] / "runtime" / "models")
).expanduser()
DEFAULT_SAM_PRETRAINED_PATH = RUNTIME_MODELS_ROOT / "sam_molra" / "sam_vit_b_01ec64.pth"
DEFAULT_WEIGHT_PATH = RUNTIME_MODELS_ROOT / "sam_molra" / "adapter.th"
DEFAULT_INPUT_ROOT = ROOT_DIR / "data" / "infer_input"
DEFAULT_INPUT_DIR = DEFAULT_INPUT_ROOT / "img"
DEFAULT_INPUT_TXT_DIR = DEFAULT_INPUT_ROOT / "txts"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "runs" / "inference"
IMAGE_EXTENSIONS = (".tif", ".tiff", ".img", ".png", ".jpg", ".jpeg", ".bmp")
RASTER_EXTENSIONS = (".tif", ".tiff", ".img")


def resolve_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for inference, but no CUDA device is available.")
    return torch.device(device_name)


def get_model_holder(model):
    return model.module if hasattr(model, "module") else model


def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and state:
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if all(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        elif all(k.startswith("model.") for k in state):
            state = {k[len("model."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def to_uint8_image(image):
    image = image.astype(np.float32, copy=False)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))
    if max_val <= 1.0 and min_val >= 0.0:
        image = image * 255.0
    elif max_val > 255.0 or min_val < 0.0 or image.dtype != np.uint8:
        denom = max(1e-6, max_val - min_val)
        image = (image - min_val) / denom * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def read_raster_image(path):
    if rasterio is None:
        raise RuntimeError("rasterio is required to read remote-sensing formats such as .IMG.")

    with rasterio.open(str(path)) as ds:
        band_count = max(1, min(3, ds.count))
        bands = ds.read(list(range(1, band_count + 1)))
        if bands.shape[0] == 1:
            bands = np.repeat(bands, 3, axis=0)
        elif bands.shape[0] == 2:
            bands = np.concatenate([bands, bands[-1:]], axis=0)
        else:
            bands = bands[:3]

        rgb = np.transpose(bands, (1, 2, 0))
        bgr = to_uint8_image(rgb)[:, :, ::-1].copy()
        profile = ds.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )
        return bgr, profile


def read_image(path):
    if Path(path).suffix.lower() in RASTER_EXTENSIONS and rasterio is not None:
        try:
            return read_raster_image(path)
        except Exception as exc:
            print(f"Warning: rasterio could not read {path}; falling back to OpenCV. {exc}")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.dtype != np.uint8:
            img = to_uint8_image(img)
        return img, None

    return read_raster_image(path)


def write_mask_outputs(output_dir, image_path, mask, raster_profile=None, group_name=None):
    png_path = unique_output_path(output_dir, image_path, group_name, suffix=".png")
    cv2.imwrite(str(png_path), mask)

    tif_path = None
    if raster_profile is not None and rasterio is not None:
        tif_path = unique_output_path(output_dir, image_path, group_name, suffix=".tif")
        profile = raster_profile.copy()
        profile.update(
            width=mask.shape[1],
            height=mask.shape[0],
            count=1,
            dtype="uint8",
            nodata=0,
        )
        with rasterio.open(str(tif_path), "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)

    return png_path, tif_path


def resolve_listed_path(raw_path, txt_path):
    path = Path(raw_path.strip().strip('"'))
    if path.is_absolute():
        return path

    candidates = [
        txt_path.parent / path,
        ROOT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_txt_images(txt_path):
    listing = read_path_list(txt_path, search_roots=(ROOT_DIR,))
    return [entry.path for entry in listing.entries]


def collect_txt_files(input_txt, input_txt_dir):
    txt_files = []
    if input_txt:
        txt_files.append(Path(input_txt))
    if input_txt_dir:
        txt_dir = Path(input_txt_dir)
        if txt_dir.is_dir():
            txt_files.extend(sorted(txt_dir.glob("*.txt")))
        elif input_txt:
            pass
        else:
            raise FileNotFoundError(f"Txt directory not found: {txt_dir}")

    missing = [str(path) for path in txt_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Txt file not found: " + ", ".join(missing))
    return txt_files


def list_input_images(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return []
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def has_txt_files(input_txt_dir):
    txt_dir = Path(input_txt_dir)
    return txt_dir.is_dir() and any(path.is_file() for path in txt_dir.glob("*.txt"))


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def collect_image_records(input_dir, input_txt=None, input_txt_dir=None, input_mode="auto"):
    if input_mode not in {"auto", "img", "txt"}:
        raise ValueError("--input_mode must be one of: auto, img, txt")

    if input_txt:
        input_mode = "txt"

    if input_mode == "auto":
        if list_input_images(input_dir):
            input_mode = "img"
        elif input_txt_dir and has_txt_files(input_txt_dir):
            input_mode = "txt"
        else:
            raise FileNotFoundError(
                "No inference inputs found.\n"
                f"Put images under: {input_dir}\n"
                f"Or put txt batch files under: {input_txt_dir}"
            )

    if input_mode == "img":
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input image directory not found: {input_dir}")
        files = list_input_images(input_dir)
        if not files:
            exts = ", ".join(IMAGE_EXTENSIONS)
            raise FileNotFoundError(f"No image files ({exts}) found in {input_dir}")
        return [(path, None) for path in files]

    txt_files = collect_txt_files(input_txt, input_txt_dir)
    records = []
    for txt_path in txt_files:
        for image_path in read_txt_images(txt_path):
            records.append((image_path, txt_path.stem))
    if not records:
        raise FileNotFoundError(f"No image paths were found in txt file(s) under {input_txt_dir}.")
    return records


def resolve_output_dir(output_root, output_dir):
    if output_dir:
        return resolve_project_path(output_dir)
    output_root = resolve_project_path(output_root)
    return output_root / f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def unique_output_path(output_dir, image_path, group_name=None, suffix=".png"):
    target_dir = output_dir / group_name if group_name else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    base = f"{image_path.stem}_mask{suffix}"
    out_path = target_dir / base
    index = 2
    while out_path.exists():
        out_path = target_dir / f"{image_path.stem}_mask_{index}{suffix}"
        index += 1
    return out_path


def infer_tile(img_bgr, model, device, tile=1024, overlap=256, threshold=0.5, return_probability=False):
    h, w = img_bgr.shape[:2]
    step = tile - overlap
    if tile <= 0:
        raise ValueError("--tile must be positive.")
    if overlap < 0 or overlap >= tile:
        raise ValueError("--overlap must be >= 0 and smaller than --tile.")

    prob = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)

    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] != max(h - tile, 0):
        ys.append(max(h - tile, 0))
    if xs[-1] != max(w - tile, 0):
        xs.append(max(w - tile, 0))

    for y0 in ys:
        for x0 in xs:
            patch = img_bgr[y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape[:2]
            canvas = np.zeros((tile, tile, 3), np.uint8)
            canvas[:ph, :pw] = patch

            x = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0 * 3.2 - 1.6
            x = torch.from_numpy(x).unsqueeze(0).to(device)

            with torch.no_grad():
                y = model(x).squeeze().detach().cpu().numpy()

            y = y[:ph, :pw]
            prob[y0:y0 + ph, x0:x0 + pw] += y
            cnt[y0:y0 + ph, x0:x0 + pw] += 1.0

    prob /= np.maximum(cnt, 1e-6)
    if return_probability:
        return np.clip(prob, 0.0, 1.0)
    return (prob > threshold).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--input_mode", type=str, choices=["auto", "img", "txt"], default="auto")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--input_txt", type=str, default="")
    parser.add_argument("--input_txt_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--weight_path", type=str, default=str(DEFAULT_WEIGHT_PATH))
    parser.add_argument("--SAM_pretrained_path", type=str, default=str(DEFAULT_SAM_PRETRAINED_PATH))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument(
        "--max_image_megapixels",
        type=float,
        default=100.0,
        help="Fail fast above this single-image size to avoid exhausting RAM. Set 0 to disable.",
    )
    args = parser.parse_args()

    input_root = resolve_project_path(args.input_root)
    input_dir = resolve_project_path(args.input_dir) if args.input_dir else input_root / "img"
    input_txt_dir = resolve_project_path(args.input_txt_dir) if args.input_txt_dir else input_root / "txts"
    output_dir = resolve_output_dir(args.output_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.SAM_pretrained_path).is_file():
        raise FileNotFoundError(f"SAM pretrained weight not found: {args.SAM_pretrained_path}")
    if not Path(args.weight_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weight_path}")

    device = resolve_device(args.device)

    model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet_multi_lora(
        args.SAM_pretrained_path,
        image_size=args.image_size,
    )
    model = model.to(device)
    load_checkpoint(model, args.weight_path)

    holder = get_model_holder(model)
    holder.enc = resize_model_pos_embed(
        holder.enc,
        img_size=args.tile,
        encoder_global_attn_indexes=encoder_global_attn_indexes,
    )
    model.eval()

    records = collect_image_records(
        input_dir=input_dir,
        input_txt=args.input_txt,
        input_txt_dir=input_txt_dir,
        input_mode=args.input_mode,
    )

    print(f"Found {len(records)} image file(s)")
    print(f"Input root: {input_root}")
    print(f"Input mode: {args.input_mode}")
    print(f"Output dir: {output_dir}")
    print(f"Tile: {args.tile}, overlap: {args.overlap}, threshold: {args.threshold}")
    for idx, (path, group_name) in enumerate(records, 1):
        img, raster_profile = read_image(path)
        image_megapixels = img.shape[0] * img.shape[1] / 1_000_000.0
        if args.max_image_megapixels > 0 and image_megapixels > args.max_image_megapixels:
            raise RuntimeError(
                f"Input image is {img.shape[1]}x{img.shape[0]} ({image_megapixels:.1f} MP), "
                f"above the safety limit of {args.max_image_megapixels:g} MP. "
                "Split it into smaller tiles, or set --max_image_megapixels 0 only if sufficient RAM is available."
            )
        probability = infer_tile(
            img,
            model,
            device,
            tile=args.tile,
            overlap=args.overlap,
            threshold=args.threshold,
            return_probability=True,
        )
        mask = (probability > args.threshold).astype(np.uint8) * 255
        png_path, tif_path = write_mask_outputs(output_dir, path, mask, raster_profile, group_name)
        probability_path = png_path.with_name(f"{png_path.stem}_probability.png")
        cv2.imwrite(str(probability_path), np.clip(probability * 255.0, 0, 255).astype(np.uint8))
        if tif_path:
            print(f"[{idx}/{len(records)}] {path} -> {png_path}, {probability_path}, {tif_path}")
        else:
            print(f"[{idx}/{len(records)}] {path} -> {png_path}, {probability_path}")


if __name__ == "__main__":
    main()
