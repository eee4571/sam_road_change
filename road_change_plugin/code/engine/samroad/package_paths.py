import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT.parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[2]
RUNTIME_ROOT = WORKSPACE_ROOT / "runtime"
MODELS_ROOT = Path(os.environ.get("SAMROAD_MODELS_ROOT", RUNTIME_ROOT / "models")).expanduser()
DATA_ROOT = PACKAGE_ROOT / "data"
WEIGHTS_ROOT = PACKAGE_ROOT / "weights"
RUNS_ROOT = PACKAGE_ROOT / "runs"

TRAIN_INPUT_ROOT = DATA_ROOT / "train_input"
INFER_INPUT_ROOT = DATA_ROOT / "infer_input"
DATASETS_ROOT = DATA_ROOT / "datasets"
CUSTOM_DATASET_ROOT = DATASETS_ROOT / "custom_cityscale"

SAM_CKPT_PATH = MODELS_ROOT / "samroad" / "sam_vit_b_01ec64.pth"
TRAIN_CKPT_ROOT = WEIGHTS_ROOT / "trained_ckpts"

TRAIN_RUNS_ROOT = RUNS_ROOT / "training"
INFER_RUNS_ROOT = RUNS_ROOT / "inference"


def resolve_path(path, base=PROJECT_ROOT):
    path = Path(path)
    if path.is_absolute():
        return path
    return base / path


def existing_path(*candidates):
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])
