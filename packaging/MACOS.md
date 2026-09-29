# Skeleton Corrector for Apple Silicon

This build produces a native arm64 `.app` for M-series Macs and a ZIP archive.
The Python interpreter and dependencies are inside the app; datasets are not.

Build **on macOS**, using an arm64 Python 3.12 environment:

```sh
python3.12 -m venv .venv-macos
source .venv-macos/bin/activate
python -m pip install -r packaging/requirements-macos.txt
python -m pip install --no-deps rtmlib==0.0.15
python tools/build_corrector_macos.py
```

Outputs: `dist/macos-arm64/CVSlice-SkeletonCorrector-macOS-arm64.zip`,
`build-info.json`, SHA256 checksum, a smoke-test report and screenshot.

The GitHub workflow uses the macos-15 ARM runner. A push to a
`codex/macos-arm64-package-*` branch triggers it; workflow_dispatch is also
provided once the workflow exists on the repository's default branch.
It runs isolated regression tests, then builds and launches the **frozen app**
against synthetic videos, CSV and SMPL PKL. This does not test model inference
or guarantee correctness on every real dataset.

The runner validates macOS 15. Older OS releases are not certified by this build;
individual bundled dependencies may require a newer OS than their source code.
The app uses ad-hoc signing and is not notarized by Apple; users may need to
approve it in System Settings > Privacy & Security. Formal notarization needs
an Apple Developer signing identity and credentials, which are not included.

PyInstaller is not a cross-compiler. Windows can prepare/submit this build but
cannot generate a valid macOS app on its own. The ZIP must be created on macOS
with `ditto` to preserve bundle symlinks and executable permissions.

Sources: [PyInstaller](https://pyinstaller.org/en/stable/index.html),
[macOS signing](https://pyinstaller.org/en/stable/feature-notes.html#macos-binary-code-signing),
[GitHub runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
