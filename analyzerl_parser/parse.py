import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Literal, Sequence, TypeAlias

ReplayPathInput: TypeAlias = str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
ReturnType: TypeAlias = Literal["export", "pandas", "polars"]
OutputType: TypeAlias = Literal["frames", "frames-only", "pbp"]
ExportFormat: TypeAlias = Literal["csv", "parquet"]
GpuMode: TypeAlias = Literal["auto", "cuda", "rocm"]

_DATA_DIR = Path.cwd()
_CSV_FLOAT_COLUMNS = (
    "history_seconds_since_kickoff",
    "history_weighted_event_count",
    "history_weighted_touch_count",
    "history_weighted_turnover_count",
    "history_weighted_pass_count",
    "history_weighted_shot_count",
    "history_weighted_goal_count",
    "history_weighted_save_count",
    "history_weighted_clear_count",
    "history_weighted_kickoff_count",
    "history_weighted_demo_count",
    "history_weighted_bump_count",
    "history_weighted_challenge_count",
    "history_weighted_entry_count",
    "history_weighted_exit_count",
    "history_weighted_retrieval_count",
)


def set_data_dir(path: str | os.PathLike[str]) -> Path:
    """Set the base directory used to resolve relative parser paths.

    Args:
        path: Base directory for relative replay and export paths.

    Returns:
        The resolved base directory.
    """
    global _DATA_DIR
    _DATA_DIR = Path(path).expanduser().resolve()
    return _DATA_DIR


def get_data_dir() -> Path:
    """Return the base directory used to resolve relative parser paths."""
    return _DATA_DIR


def _path(value: str | os.PathLike[str]) -> Path:
    value = Path(value).expanduser()
    return value if value.is_absolute() else (_DATA_DIR / value).resolve()


def _ensure_executable(path):
    path = Path(path)

    if sys.platform.startswith("win"):
        return path

    # Installed Linux wheels may need the bundled binary made executable.
    mode = path.stat().st_mode
    if mode & 0o111 == 0:
        try:
            os.chmod(path, mode | 0o755)
        except PermissionError:
            path = _copy_binary_to_user_cache(path)

    return path


def _user_cache_dir():
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "analyzerl_parser" / "bin"

    return Path.home() / ".cache" / "analyzerl_parser" / "bin"


def _copy_binary_to_user_cache(source):
    source = Path(source)
    cache_dir = _user_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / source.name

    # Fall back to a user-writable binary path when site-packages is read-only.
    should_copy = True
    if target.exists():
        source_stat = source.stat()
        target_stat = target.stat()
        should_copy = (
            source_stat.st_size != target_stat.st_size
            or int(source_stat.st_mtime) != int(target_stat.st_mtime)
        )

    if should_copy:
        shutil.copy2(source, target)

    mode = target.stat().st_mode
    if mode & 0o111 == 0:
        os.chmod(target, mode | 0o755)

    return target


def _boxcars_binary():
    exe_name = "analyzerl_boxcars.exe" if sys.platform.startswith("win") else "analyzerl_boxcars"

    package_file = Path(__file__).resolve()

    # Prefer packaged binaries, then local development builds, then PATH.
    candidates = [
        package_file.parent / "bin" / exe_name,
        package_file.parent.parent / "analyzerl_boxcars" / "target" / "release" / exe_name,
        package_file.parent / "analyzerl_boxcars" / "target" / "release" / exe_name,
        Path.cwd() / "analyzerl_parser" / "analyzerl_boxcars" / "target" / "release" / exe_name,
    ]

    system_binary = shutil.which(exe_name)
    if system_binary:
        candidates.append(Path(system_binary))

    for candidate in candidates:
        if candidate.exists():
            return str(_ensure_executable(candidate))

    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find analyzerl_boxcars binary. Searched:\n"
        f"{searched}"
    )


def _run_boxcars(command):
    try:
        subprocess.run(command, check=True)
        return
    except PermissionError:
        binary = Path(command[0])

        if sys.platform.startswith("win"):
            raise

        cached_binary = _copy_binary_to_user_cache(binary)
        subprocess.run([str(cached_binary), *command[1:]], check=True)


def _gpu_mode(gpu: GpuMode | str | None) -> str | None:
    if gpu is None:
        return None
    value = str(gpu).lower()
    if value == "auto":
        return "auto"
    if value in {"rocm", "amd", "radeon"}:
        return "rocm"
    if value in {"cuda", "nvidia"}:
        return "cuda"
    raise ValueError("gpu must be None, 'auto', 'cuda', or 'rocm'")


def _replay_files(replay_path: ReplayPathInput) -> list[Path]:
    if isinstance(replay_path, (list, tuple, set)):
        paths = []

        for item in replay_path:
            paths.extend(_replay_files(item))

        paths = sorted(set(paths))

        if not paths:
            raise FileNotFoundError("No .replay files found in input list")

        return paths

    replay_path = _path(replay_path)

    if replay_path.is_file():
        if replay_path.suffix.lower() != ".replay":
            raise ValueError(f"Not a .replay file: {replay_path}")
        return [replay_path]

    if replay_path.is_dir():
        paths = sorted(replay_path.rglob("*.replay"))

        if not paths:
            raise FileNotFoundError(f"No .replay files found in: {replay_path}")

        return paths

    raise FileNotFoundError(f"Replay path does not exist: {replay_path}")


def _prepare_replay_inputs(
    replay_path: ReplayPathInput,
    replay_files: Sequence[Path] | None = None,
) -> list[Path]:
    # Keep a single folder intact so the Rust parser can batch it efficiently.
    if not isinstance(replay_path, (list, tuple, set)):
        resolved = _path(replay_path)

        if resolved.is_dir():
            recursive_replays = list(replay_files) if replay_files is not None else _replay_files(resolved)
            direct_replays = [path for path in recursive_replays if path.parent == resolved]
            if len(direct_replays) == len(recursive_replays):
                return [resolved]
            return recursive_replays

        if resolved.is_file():
            if resolved.suffix.lower() != ".replay":
                raise ValueError(f"Not a .replay file: {resolved}")
            return [resolved]

    return list(replay_files) if replay_files is not None else _replay_files(replay_path)


def _apply_limit(replay_inputs: list[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return replay_inputs

    limit = int(limit)
    if limit <= 0:
        raise ValueError("limit must be a positive integer when provided")

    return replay_inputs[:limit]


def _output_config(
    output: OutputType | str,
    export_format: ExportFormat | str | None = None,
) -> dict[str, str]:
    output = str(output).lower()
    export_format = None if export_format is None else str(export_format).lower()

    if output in {"frames", "frames-only"}:
        export_format = export_format or "parquet"
        if export_format not in {"csv", "parquet"}:
            raise ValueError("export_format must be one of: 'csv', 'parquet'")
        config = {
            "command": "frames",
            "out_arg": "--out-frames",
            "glob": f"*_frames.{export_format}",
            "suffix": "_frames",
            "format": export_format,
        }
        if output == "frames-only":
            config["mode_arg"] = "--frames-only"
        return config

    if output == "pbp":
        export_format = export_format or "csv"
        if export_format not in {"csv", "parquet"}:
            raise ValueError("export_format must be one of: 'csv', 'parquet'")
        return {
            "command": "parse",
            "out_arg": "--out-pbp",
            "glob": f"*_pbp.{export_format}",
            "suffix": "_pbp",
            "format": export_format,
        }

    raise ValueError("output must be one of: 'frames', 'frames-only', 'pbp'")


def _export_path_config(
    export: str | os.PathLike[str],
    output: OutputType | str,
    export_format: ExportFormat | str | None,
    replay_count: int,
) -> tuple[Path, Path | None, dict[str, str]]:
    export_path = _path(export)
    suffix = export_path.suffix.lower()
    file_target = None

    if suffix in {".csv", ".parquet"}:
        inferred_format = suffix[1:]
        if export_format is not None and str(export_format).lower() != inferred_format:
            raise ValueError("export_format conflicts with the export path suffix")
        if replay_count != 1:
            raise ValueError("file export paths can only be used with exactly one replay")
        export_format = inferred_format
        file_target = export_path
        export_path = export_path.parent

    return export_path, file_target, _output_config(output, export_format)


def _expected_export_files(
    replay_files: Sequence[Path],
    export_path: Path,
    config: dict[str, str],
) -> list[Path]:
    return [
        export_path / f"{path.stem}{config['suffix']}.{config['format']}"
        for path in replay_files
    ]


def _actual_export_files(
    replay_files: Sequence[Path],
    export_path: Path,
    config: dict[str, str],
    file_target: Path | None = None,
) -> list[Path]:
    expected = [
        path
        for path in _expected_export_files(replay_files, export_path, config)
        if path.exists()
    ]

    if file_target is not None:
        if file_target.exists() and not expected:
            return [file_target]
        if not expected:
            return []

        source = expected[0]
        file_target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != file_target.resolve():
            source.replace(file_target)
        return [file_target]

    return expected


def _assert_export_files(export_files: Sequence[Path], expected_count: int) -> None:
    if len(export_files) != expected_count:
        preview = [str(path) for path in export_files[:20]]
        if len(export_files) > 20:
            preview.append(f"... {len(export_files) - 20} more")
        raise FileNotFoundError(
            f"Parser produced {len(export_files)} export files, expected {expected_count}. "
            f"Found: {preview}"
        )


def _read_export_files(
    export_files: Iterable[Path],
    return_type: ReturnType | str,
    export_format: ExportFormat | str,
    xg_model_path: str | os.PathLike[str] | None = None,
):
    return_type = str(return_type).lower()
    export_format = str(export_format).lower()

    if return_type == "polars":
        import polars as pl

        export_files = list(export_files)
        if not export_files:
            return pl.DataFrame()

        if export_format == "parquet":
            frame = pl.scan_parquet(export_files).collect()
        else:
            frame = pl.scan_csv(
                export_files,
                schema_overrides={
                    column: pl.Float64
                    for column in _CSV_FLOAT_COLUMNS
                },
            ).collect()

        if xg_model_path is not None:
            from .xg import apply_xg_to_pbp

            frame = apply_xg_to_pbp(frame, xg_model_path)

        return frame

    if return_type == "pandas":
        import pandas as pd

        if not export_files:
            return pd.DataFrame()

        if xg_model_path is not None:
            from .xg import apply_xg_to_pbp

            polars_frame = _read_export_files(
                export_files,
                "polars",
                export_format,
                xg_model_path=xg_model_path,
            )
            return polars_frame.to_pandas()

        reader = pd.read_parquet if export_format == "parquet" else pd.read_csv
        return pd.concat(
            [reader(path) for path in export_files],
            ignore_index=True,
        )

    raise ValueError("return_type must be one of: 'export', 'pandas', 'polars'")


def parse_replay(
    replay_path: ReplayPathInput = "data/replays",
    export: str | os.PathLike[str] = "data/frames",
    workers: int = 4,
    return_type: ReturnType = "export",
    output: OutputType = "frames",
    export_format: ExportFormat | None = None,
    force: bool = False,
    limit: int | None = None,
    xg_model_path: str | os.PathLike[str] | None = None,
    gpu: GpuMode | str | None = None,
    rotation_events: bool = True,
):
    """Parse one or more Rocket League replays with the bundled Rust CLI.

    Args:
        replay_path: A replay file, a folder of replay files, or a sequence of
            replay file and folder paths.
        export: Output folder for generated frame or play-by-play files.
        workers: Number of Rust parser worker threads to use.
        return_type: Whether to return export paths, a pandas DataFrame, or a
            Polars DataFrame.
        output: Export mode, either full analyzed ``frames`` or play-by-play
            ``pbp``.
        export_format: Output file format. Defaults to ``parquet`` for
            ``frames`` and ``csv`` for ``pbp``.
        force: Whether to overwrite existing exports.
        limit: Optional replay count limit when ``replay_path`` is a directory.
        xg_model_path: Optional saved xG model file or folder. When provided,
            parsed PBP or frame exports are scored and an ``xG`` column is
            added to shot and goal rows.
        gpu: Optional parser GPU mode. Python ``None`` is the default and uses
            CPU only; ``"auto"``, ``"cuda"``, and ``"rocm"`` request GPU.
        rotation_events: Whether to add ``rotation-filled``,
            ``rotation-cut``, and ``rotation-stalled`` event rows. Defaults
            to ``True``.

    Returns:
        A list of exported file paths, a pandas DataFrame, or a Polars
        DataFrame depending on ``return_type``.

    Raises:
        FileNotFoundError: If replay input paths or the parser binary cannot be
            found.
        ValueError: If the export configuration is invalid.
        subprocess.CalledProcessError: If the Rust parser process fails.
    """
    return_type = str(return_type).lower()

    if return_type not in {"export", "polars", "pandas"}:
        raise ValueError("return_type must be one of: 'export', 'pandas', 'polars'")

    if export is None:
        raise ValueError("export cannot be None")

    replay_files = _apply_limit(_replay_files(replay_path), limit)
    export_path, file_target, config = _export_path_config(
        export,
        output,
        export_format,
        len(replay_files),
    )
    export_path.mkdir(parents=True, exist_ok=True)
    replay_inputs = _prepare_replay_inputs(replay_path, replay_files)
    expected_count = len(replay_files)

    workers = max(int(workers or 1), 1)
    binary = _boxcars_binary()
    export_files = []

    command = [
        binary,
        config["command"],
        "--workers",
        str(workers),
    ]
    for replay_input in replay_inputs:
        command.extend(["--replays", str(replay_input)])

    command.extend([config["out_arg"], str(export_path)])
    command.extend(["--format", config["format"]])
    gpu_mode = _gpu_mode(gpu)
    if gpu_mode is not None:
        command.extend(["--gpu", gpu_mode])
    if "mode_arg" in config:
        command.append(config["mode_arg"])
    if not rotation_events:
        command.append("--no-rotation-events")

    if limit is not None:
        command.extend(["--limit", str(int(limit))])

    if force:
        command.append("--force")

    _run_boxcars(command)

    export_files = _actual_export_files(
        replay_files,
        export_path,
        config,
        file_target=file_target,
    )
    _assert_export_files(export_files, expected_count)

    if xg_model_path is not None:
        from .xg import apply_xg_to_file

        for export_file in export_files:
            apply_xg_to_file(export_file, xg_model_path, config["format"])

    if return_type == "export":
        return export_files

    return _read_export_files(
        export_files,
        return_type,
        config["format"],
        xg_model_path=None,
    )
