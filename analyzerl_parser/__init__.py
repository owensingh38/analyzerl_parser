"""Public Python API for the AnalyzeRL parser package."""

from os import PathLike
from typing import Any, Literal, Sequence, TypeAlias

ReplayPathInput: TypeAlias = str | PathLike[str] | Sequence[str | PathLike[str]]
StatsInput: TypeAlias = str | PathLike[str] | Sequence[str | PathLike[str]]
ReturnType: TypeAlias = Literal["export", "pandas", "polars"]
OutputType: TypeAlias = Literal["frames", "pbp"]
ExportFormat: TypeAlias = Literal["csv", "parquet"]
StatsReturnType: TypeAlias = Literal["polars", "pandas", "list"]
RenderMode: TypeAlias = Literal["2d", "3d"]

__version__ = "1.0.3"

def parse_replay(
    replay_path: ReplayPathInput = "data/replays",
    export: str | PathLike[str] = "data/frames",
    workers: int = 4,
    return_type: ReturnType = "export",
    output: OutputType = "frames",
    export_format: ExportFormat | None = None,
    force: bool = False,
    limit: int | None = None,
    xg_model_path: str | PathLike[str] | None = None,
):
    """Parse one or more replays into play-by-play or analyzed frame exports.

    Args:
        replay_path: Replay file path, replay folder path, or a sequence of
            replay file and folder paths.
        export: Output folder for generated exports.
        workers: Number of Rust parser worker threads to use.
        return_type: Whether to return export paths, pandas data, or Polars
            data.
        output: Export mode, either ``frames`` or ``pbp``.
        export_format: File format for the exports.
        force: Whether to overwrite existing exports.
        limit: Optional replay count limit when parsing a directory.
        xg_model_path: Optional saved xG model file or folder. When provided,
            parsed PBP or frame exports receive an ``xG`` column on shot and
            goal rows.

    Returns:
        Parser output in the format requested by ``return_type``.
    """
    from .parse import parse_replay as _parse_replay

    return _parse_replay(
        replay_path=replay_path,
        export=export,
        workers=workers,
        return_type=return_type,
        output=output,
        export_format=export_format,
        force=force,
        limit=limit,
        xg_model_path=xg_model_path,
    )


def calculate_stats(
    frames: StatsInput,
    return_type: StatsReturnType = "polars",
    export: str | PathLike[str] | None = None,
    group_by: Sequence[str] | str | None = None,
    rates: bool = False,
    workers: int = 4,
    parse_export: str | PathLike[str] = "data/frames",
    force: bool = False,
    limit: int | None = None,
    xg_model_path: str | PathLike[str] | None = None,
):
    """Aggregate per-player replay stats from replay, PBP, or frame data.

    Args:
        frames: Replay file, replay folder, one or more replay files, CSV or
            Parquet PBP/frame file, folder of CSV or Parquet PBP/frame files,
            or one or more CSV or Parquet PBP/frame files.
        return_type: Whether to return Polars, pandas, or list output.
        export: Optional destination path for the aggregated stats table.
        group_by: Output grouping columns. Defaults to ``["replay_id",
            "player_id"]``.
        rates: Whether to add per-five-minute and per-game rate columns.
        workers: Number of Rust parser workers to use when replay inputs need
            parsing.
        parse_export: Output folder for generated PBP Parquet files when
            replay inputs need parsing.
        force: Whether to overwrite existing parser exports for replay inputs.
        limit: Optional file or replay count limit.
        xg_model_path: Optional saved xG model file or folder. When provided,
            source rows are scored before expected-goal stats are aggregated.

    Returns:
        Aggregated replay stats in the requested format.
    """
    from .stats import calculate_stats as _calculate_stats

    return _calculate_stats(
        frames=frames,
        return_type=return_type,
        export=export,
        group_by=group_by,
        rates=rates,
        workers=workers,
        parse_export=parse_export,
        force=force,
        limit=limit,
        xg_model_path=xg_model_path,
    )


def animate_replay(
    replay_path: str | PathLike[str],
    event_window_frames: int = 45,
    event_types: str | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    parser_path: str | PathLike[str] | None = None,
    render_mode: RenderMode = "3d",
    export_path: str | PathLike[str] | None = None,
    view_elev: int = 28,
    view_azim: int = -64,
    xg_model_path: str | PathLike[str] | None = None,
):
    """Render an animated replay view from the parser's animation export.

    Args:
        replay_path: Replay file to animate.
        event_window_frames: Number of frames for visible event history.
        event_types: Optional comma-separated event type filter.
        start_frame: Optional first frame to render.
        end_frame: Optional last frame to render.
        parser_path: Optional explicit parser executable path.
        render_mode: ``2d`` or ``3d`` rendering mode.
        export_path: Optional path for GIF or MP4 export.
        view_elev: Default 3D camera elevation.
        view_azim: Default 3D camera azimuth.
        xg_model_path: Optional saved xG model file or folder. When provided,
            shot and goal event labels include ``xG``.

    Returns:
        An export path or interactive timer, depending on mode.
    """
    from .animate import animate_replay as _animate_replay

    return _animate_replay(
        replay_path=replay_path,
        event_window_frames=event_window_frames,
        event_types=event_types,
        start_frame=start_frame,
        end_frame=end_frame,
        parser_path=parser_path,
        render_mode=render_mode,
        export_path=export_path,
        view_elev=view_elev,
        view_azim=view_azim,
        xg_model_path=xg_model_path,
    )
