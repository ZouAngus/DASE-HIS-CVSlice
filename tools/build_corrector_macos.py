"""Build natively on an Apple Silicon Mac, validate the app, then ZIP it."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = "CVSlice-SkeletonCorrector"


def run(args, **kw):
    return subprocess.run(args, cwd=ROOT, check=True, **kw)


def main():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("Run this build on macOS with a native Apple Silicon Python (arm64).")
    output = ROOT / "dist" / "macos-arm64"
    work = ROOT / "build" / "macos-arm64"
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--onedir", "--windowed", "--name", NAME,
         "--target-architecture", "arm64", "--osx-bundle-identifier",
         "org.cvslice.skeletoncorrector", "--distpath", str(output),
         "--workpath", str(work / "objects"), "--specpath", str(work),
         "--collect-all", "rtmlib", "--collect-all", "onnxruntime",
         "--exclude-module", "torch", str(ROOT / "corrector_main.py")])
    app = output / f"{NAME}.app"
    exe = app / "Contents" / "MacOS" / NAME
    arch = run(["lipo", "-archs", str(exe)], capture_output=True, text=True).stdout.strip()
    if arch != "arm64":
        raise RuntimeError(f"Unexpected app architecture: {arch}")
    run(["codesign", "--verify", "--deep", "--strict", str(app)])
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    report = output / "smoke-test.json"
    run([str(exe), "--smoke-test", str(report)], env=env, timeout=120)
    data = json.loads(report.read_text(encoding="utf-8"))
    if data.get("status") != "passed" or not data.get("pose_backend_imports"):
        raise RuntimeError("Packaged application smoke test failed")
    revision = run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    # ditto preserves the symlinks, executable modes and bundle layout needed
    # by PyInstaller on macOS. Do not ZIP this bundle on Windows.
    zip_path = output / f"{NAME}-macOS-arm64.zip"
    run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app), str(zip_path)])
    with zip_path.open("rb") as archive:
        sha = hashlib.file_digest(archive, "sha256").hexdigest()
    (output / "SHA256.txt").write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")
    (output / "build-info.json").write_text(json.dumps({
        "revision": revision, "architecture": arch, "build_os": platform.platform(),
        "python": sys.version, "signing": "ad-hoc (not Apple-notarized)",
        "zip_sha256": sha, "smoke_test": data,
    }, indent=2), encoding="utf-8")
    (output / "READ-ME-FIRST.txt").write_text(
        "CVSlice Skeleton Corrector / Apple Silicon\n\n"
        "Unzip on the Mac, then move CVSlice-SkeletonCorrector.app to Applications.\n"
        "No separate Python installation is needed. Use File > Open to select your dataset.\n"
        "Built and startup-tested on macOS 15; older macOS versions are unverified.\n"
        "The app is ad-hoc signed, not Apple-notarized. macOS may ask you to approve\n"
        "it in System Settings > Privacy & Security > Open Anyway.\n"
        "The optional pose model downloads weights when first used (network required).\n"
        "Videos, calibration and skeleton datasets are not included.\n"
        "An automated synthetic-data smoke test passed; real-data review is still advised.\n",
        encoding="utf-8")
    print(zip_path)


if __name__ == "__main__":
    main()
