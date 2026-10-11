"""Build the Windows x64 UI bundle without reading or modifying backend runtime.

Run with a development Python: python road_change_plugin/tools/build_portable.py
Only the build machine needs internet and Git. No pip/venv is used in the bundle.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.14.3"
QT_VERSION = "6.11.2"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
PYTHON_SHA256 = "ad4961a479dedbeb7c7d113253f8db1b1935586b73c27488712beec4f2c894e6"
WHEELS = {
    "PySide6_Essentials": "c8a29def77032773a30879f7f24415b5395ad08592d147c170824ef4c735dfc1",
    "shiboken6": "6ab0eba1c904455df621f9a6df3ca2bb896bab8670572d2bc4e37804ae91f19a",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, target, digest):
    if target.is_file() and sha256(target) == digest:
        return
    print(f"Downloading {target.name}", flush=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as out:
        shutil.copyfileobj(source, out)
    if sha256(partial) != digest:
        raise ValueError(f"Download checksum mismatch: {target.name}")
    partial.replace(target)


def prepare_gui(cache):
    gui = ROOT / "gui_env"
    identity = {"python": PYTHON_VERSION, "PySide6_Essentials": QT_VERSION,
                "shiboken6": QT_VERSION, "platform": "win_amd64"}
    marker = gui / "bundle.json"
    if gui.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise RuntimeError("gui_env already exists with a different/incomplete build; preserve it before rebuilding.")
        return gui
    python_zip = cache / Path(PYTHON_URL).name
    download(PYTHON_URL, python_zip, PYTHON_SHA256)
    # Staging and its automatic cleanup are confined to this build directory.
    with tempfile.TemporaryDirectory(prefix="gui-", dir=cache.parent) as tmp:
        staging = Path(tmp).resolve()
        assert staging.is_relative_to((ROOT / "build").resolve())
        with zipfile.ZipFile(python_zip) as archive:
            archive.extractall(staging)
        site = staging / "Lib/site-packages"
        site.mkdir(parents=True)
        for package, digest in WHEELS.items():
            with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{QT_VERSION}/json", timeout=60) as response:
                metadata = json.load(response)
            wheels = [item for item in metadata["urls"]
                      if item["filename"].endswith("-cp310-abi3-win_amd64.whl")]
            if len(wheels) != 1 or wheels[0]["digests"]["sha256"] != digest:
                raise ValueError(f"Unexpected wheel identity: {package}")
            wheel = cache / wheels[0]["filename"]
            download(wheels[0]["url"], wheel, digest)
            with zipfile.ZipFile(wheel) as archive:
                if any(".data/" in name for name in archive.namelist()):
                    raise ValueError("Wheel requires installation layout handling")
                archive.extractall(site)
        # Relative paths make the environment relocatable. Do not enable site:
        # user packages, registry, PYTHONPATH and .pth startup hooks stay disabled.
        (staging / "python314._pth").write_text(
            "python314.zip\n.\nLib/site-packages\n..\n", encoding="utf-8")
        (staging / "bundle.json").write_text(json.dumps(identity, indent=2) + "\n")
        shutil.copytree(staging, gui)
    return gui


def source_files():
    # Git is the source inventory, never enumerate runtime/env or runtime/model.
    result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "."],
                            cwd=ROOT, check=True, stdout=subprocess.PIPE)
    for name in sorted(set(result.stdout.decode("utf-8").split("\0"))):
        path = Path(name)
        if not name or path.name == "AGENTS.md":
            continue
        if path.parts[0] in {"code", "plugin", "experiments"}:
            yield path
        elif name in {"standalone.py", "start_plugin.bat", "plugin.json", "README.md",
                      "PORTABLE.md", "tools/build_portable.py", "resources/production_config.json",
                      "resources/backend_snapshot.json", "resources/chevron_down.svg",
                      "runtime/config/samroad_inference.yaml"}:
            yield path


def trim_gui(gui):
    # This application uses QWidget, not QML. The wheel also contains developer
    # headers and QML build objects whose paths exceed Windows Explorer's limit
    # after relocation. Retain runtime DLLs, platform/image plugins and licenses.
    for name in ("qml", "include", "typesystems", "glue"):
        target = (gui / "Lib/site-packages/PySide6" / name).resolve()
        if not target.is_relative_to(gui.resolve()):
            raise ValueError(f"Refusing cleanup outside gui_env: {target}")
        if target.is_dir():
            shutil.rmtree(target)


def clean_environment():
    env = os.environ.copy()
    for key in list(env):
        if key.upper().startswith(("PYTHON", "QT_")):
            del env[key]
    # Simulate a machine with no Python, Conda or Qt on PATH.
    windows = Path(env["SYSTEMROOT"])
    env["PATH"] = os.pathsep.join(map(str, (windows / "System32", windows)))
    return env


def check_bundle(root):
    env = clean_environment()
    python = root / "gui_env/python.exe"
    probe = (
        "import pathlib,sys,PySide6; "
        "r=pathlib.Path(sys.executable).resolve().parents[1]; "
        "assert pathlib.Path(PySide6.__file__).is_relative_to(r/'gui_env'); "
        "assert all(pathlib.Path(p).is_relative_to(r) for p in sys.path); "
        "from plugin.runner import Runner; "
        "assert Runner().python_path()==r/'runtime/env/samroad_env/python.exe'; "
        "assert not {'torch','numpy','rasterio','tkinter'}.intersection(sys.modules); "
        "from PySide6.QtCore import QTimer; import standalone; "
        "original=standalone.create_plugin; "
        "standalone.create_plugin=lambda: (QTimer.singleShot(150, standalone.QApplication.instance().quit), original())[1]; "
        "raise SystemExit(standalone.main())"
    )
    # offscreen exercises the actual application event loop without a visible UI.
    subprocess.run([str(python), "-c", probe], cwd=root, env={**env, "QT_QPA_PLATFORM": "offscreen"},
                   check=True, timeout=30)
    # The BAT also constructs a QWidget using the native Windows Qt platform.
    subprocess.run([env["COMSPEC"], "/d", "/c", "start_plugin.bat", "--check"],
                   cwd=root, env=env, check=True, timeout=30)


def main():
    if os.name != "nt":
        raise SystemExit("Build and verify on Windows x64.")
    cache = ROOT / "build/downloads"
    cache.mkdir(parents=True, exist_ok=True)
    gui = prepare_gui(cache)
    trim_gui(gui)
    check_bundle(ROOT)
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    output = dist / "road_change_plugin-windows-x64-portable.zip"
    pending = output.with_suffix(".zip.partial")
    manifest = {}
    print("Packaging application and isolated gui_env (backend runtime excluded)", flush=True)
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        files = list(source_files()) + [p.relative_to(ROOT) for p in gui.rglob("*")
                                       if p.is_file() and "__pycache__" not in p.parts]
        for relative in sorted(files):
            path = ROOT / relative
            if path.is_symlink():
                raise ValueError(f"Refusing linked distribution file: {relative}")
            archive.write(path, "road_change_plugin/" + relative.as_posix())
            manifest[relative.as_posix()] = sha256(path)
        archive.writestr("road_change_plugin/portable_manifest.json",
                         json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    # Test what the recipient receives, after relocation, without any runtime.
    with tempfile.TemporaryDirectory(prefix="portable-中文 空格-", dir=ROOT / "build") as tmp:
        relocated = Path(tmp).resolve()
        assert relocated.is_relative_to((ROOT / "build").resolve())
        with zipfile.ZipFile(pending) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity check failed")
            archive.extractall(relocated)
        app = relocated / "road_change_plugin"
        assert not (app / "runtime/env").exists()
        assert not (app / "runtime/model").exists()
        for relative, digest in manifest.items():
            assert sha256(app / relative) == digest, relative
        snapshot = json.loads((app / "resources/backend_snapshot.json").read_text())
        for relative, digest in snapshot.items():
            assert sha256(app / relative) == digest, relative
        check_bundle(app)
    pending.replace(output)
    checksum = sha256(output)
    output.with_suffix(".zip.sha256").write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    print(f"Verified: {output}\nSize: {output.stat().st_size / 1024**2:.1f} MiB\nSHA256: {checksum}")


if __name__ == "__main__":
    main()
