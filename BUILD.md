# analyzerl_parser BUILD Instructions

Build and packaging notes for `analyzerl_parser`

## Build the Rust binary

Run from the outer `analyzerl_parser` folder, where `analyzerl_boxcars/Cargo.toml` exists:

```powershell
cargo build --release --manifest-path .\analyzerl_boxcars\Cargo.toml
```

Linux release build in manylinux:

```powershell
docker run --rm `
  -v "${PWD}:/io" `
  -w /io `
  quay.io/pypa/manylinux2014_x86_64 `
  bash -lc "curl https://sh.rustup.rs -sSf | sh -s -- -y && source ~/.cargo/env && cargo build --release --manifest-path analyzerl_boxcars/Cargo.toml"
```

Windows binary output:

```text
analyzerl_boxcars\target\release\analyzerl_boxcars.exe
```

Linux binary output:

```text
analyzerl_boxcars/target/release/analyzerl_boxcars
```

## Build wheels

Install tools:

```powershell
python -m pip install -U pip build twine cibuildwheel maturin
```

Build a local wheel with the bundled CLI included:

```powershell
python -m maturin build --release
```

Built wheels land in:

```text
analyzerl_boxcars\target\wheels
```

Build Windows wheel with `cibuildwheel`:

```powershell
$env:CIBW_BUILD = "cp39-*"
$env:CIBW_SKIP = "pp* *-musllinux*"
$env:CIBW_ARCHS_WINDOWS = "AMD64"

python -m cibuildwheel --platform windows --output-dir dist
```

Build Linux wheel through Docker Desktop:

```powershell
$env:CIBW_BUILD = "cp39-*"
$env:CIBW_SKIP = "pp* *-musllinux*"
$env:CIBW_ARCHS_LINUX = "x86_64"

python -m cibuildwheel --platform linux --output-dir dist
```

Build source distribution:

```powershell
python -m build --sdist
```

Check artifacts:

```powershell
twine check dist/*
```

Upload to PyPI:

```powershell
twine upload dist/*
```

## Runtime expectations

After install, both of these should work without a separately installed system binary:

```powershell
analyzerl-boxcars --help
```

```python
from analyzerl_parser import parse_replay
```

In a source checkout, binary resolution still falls back to:

```text
analyzerl_boxcars/target/release/analyzerl_boxcars.exe
```

on Windows, or:

```text
analyzerl_boxcars/target/release/analyzerl_boxcars
```

on Linux.
