"""Public Python API for the AnalyzeRL parser package."""

from os import PathLike
from typing import Literal, Sequence, TypeAlias

ReplayPathInput: TypeAlias = str | PathLike[str] | Sequence[str | PathLike[str]]
StatsInput: TypeAlias = str | PathLike[str] | Sequence[str | PathLike[str]]
ReturnType: TypeAlias = Literal["export", "pandas", "polars"]
OutputType: TypeAlias = Literal["frames", "pbp"]
ExportFormat: TypeAlias = Literal["csv", "parquet"]
StatsReturnType: TypeAlias = Literal["polars", "pandas", "list"]
RenderMode: TypeAlias = Literal["2d", "3d"]
ExportMode: TypeAlias = Literal["fast"]

__version__ = "1.0.2"


def parse_replay(
    replay_path: ReplayPathInput = "data/replays",
    export: str | PathLike[str] = "data/frames",
    workers: int = 4,
    return_type: ReturnType = "export",
    output: OutputType = "frames",
    export_format: ExportFormat | None = None,
    force: bool = False,
    limit: int | None = None,
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
    )


def calculate_stats(
    frames: StatsInput,
    return_type: StatsReturnType = "polars",
    export: str | PathLike[str] | None = None,
    workers: int = 4,
    parse_export: str | PathLike[str] = "data/frames",
    force: bool = False,
    limit: int | None = None,
):
    """Aggregate per-player replay stats from replay, PBP, or frame data.

    Args:
        frames: Replay file, replay folder, one or more replay files, CSV or
            Parquet PBP/frame file, folder of CSV or Parquet PBP/frame files,
            or one or more CSV or Parquet PBP/frame files.
        return_type: Whether to return Polars, pandas, or list output.
        export: Optional destination path for the aggregated stats table.
        workers: Number of Rust parser workers to use when replay inputs need
            parsing.
        parse_export: Output folder for generated PBP Parquet files when
            replay inputs need parsing.
        force: Whether to overwrite existing parser exports for replay inputs.
        limit: Optional file or replay count limit.

    Returns:
        Aggregated replay stats in the requested format.
    """
    from .stats import calculate_stats as _calculate_stats

    return _calculate_stats(
        frames=frames,
        return_type=return_type,
        export=export,
        workers=workers,
        parse_export=parse_export,
        force=force,
        limit=limit,
    )


def animate_replay(
    replay_path: str | PathLike[str],
    frame_step: int = 2,
    interval: int = 33,
    event_window_frames: int = 45,
    event_types: str | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    parser_path: str | PathLike[str] | None = None,
    render_mode: RenderMode = "3d",
    export_path: str | PathLike[str] | None = None,
    export_fps: int = 30,
    export_mode: ExportMode = "fast",
    export_max_frames=None,
    view_elev: int = 28,
    view_azim: int = -64,
):
    """Render an animated replay view from the parser's animation export.

    Args:
        replay_path: Replay file to animate.
        frame_step: Frame downsampling step used for the animation export.
        interval: Playback interval in milliseconds.
        event_window_frames: Number of frames for visible event history.
        event_types: Optional comma-separated event type filter.
        start_frame: Optional first frame to render.
        end_frame: Optional last frame to render.
        parser_path: Optional explicit parser executable path.
        render_mode: ``2d`` or ``3d`` rendering mode.
        export_path: Optional path for GIF or MP4 export.
        export_fps: Export frame rate.
        export_mode: Export strategy name.
        export_max_frames: Optional frame cap for fast exports.
        view_elev: Default 3D camera elevation.
        view_azim: Default 3D camera azimuth.

    Returns:
        An export path or interactive timer, depending on mode.
    """
    from .animate import animate_replay as _animate_replay

    return _animate_replay(
        replay_path=replay_path,
        frame_step=frame_step,
        interval=interval,
        event_window_frames=event_window_frames,
        event_types=event_types,
        start_frame=start_frame,
        end_frame=end_frame,
        parser_path=parser_path,
        render_mode=render_mode,
        export_path=export_path,
        export_fps=export_fps,
        export_mode=export_mode,
        export_max_frames=export_max_frames,
        view_elev=view_elev,
        view_azim=view_azim,
    )
