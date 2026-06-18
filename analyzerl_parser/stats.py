"""Player-level replay stat aggregation for parsed AnalyzeRL data."""

from os import PathLike
from typing import Any, Literal, Sequence

import polars as pl

from pathlib import Path

DATA_SUFFIXES = {".csv", ".parquet"}
REPLAY_SUFFIX = ".replay"
CSV_NULL_VALUES = ["", "NA", "NaN", "None", "null"]

PLAYER_SLOTS = [
    "blue_player_1",
    "blue_player_2",
    "blue_player_3",
    "orange_player_1",
    "orange_player_2",
    "orange_player_3",
]


def requested_columns(has_xg: bool) -> list[str]:
    columns = [
        "game_id",
        "replay_id",
        "event_type",
        "event_team",
        "event_player_1_id",
        "event_player_1_name",
        "event_player_1_team",
        "event_player_2_id",
        "event_player_2_name",
        "event_player_2_team",
        "seconds_elapsed",
        "official_assist",
        "boost_pickup_amount",
        "boost_pickup_type",
        "off_demo",
        "off_kickoff",
        "off_challenge_win",
        "off_bump",
        "off_air_dribble",
        "off_ground_dribble",
        "off_flick",
        "pass_in_play",
        "aerialing",
        "air_dribble",
        "ground_dribble",
        "flick_shot",
        "rebound",
        "off_flip_reset",
        "off_wall",
        "off_ceiling",
        "blue_team_name",
        "orange_team_name",
    ]

    if has_xg:
        columns.append("xG")

    for slot in PLAYER_SLOTS:
        for field in [
            "id",
            "network_id",
            "name",
            "platform",
            "rank",
            "rank_tier",
            "pro_player",
            "mmr",
            "car_id",
            "car_name",
            "is_bot",
            "time_in_game",
        ]:
            columns.append(f"{slot}_{field}")

    return list(dict.fromkeys(columns))


def _is_path_like(value: Any) -> bool:
    return isinstance(value, str | PathLike)


def _is_path_input(value: Any) -> bool:
    if _is_path_like(value):
        return True

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return bool(value) and all(_is_path_like(item) for item in value)

    return False


def _path_items(value: Any) -> list[str | PathLike[str]]:
    if _is_path_like(value):
        return [value]

    return list(value)


def _resolve_input_path(value: str | PathLike[str]) -> Path:
    from .parse import _path

    return _path(value)


def _folder_data_files(folder: Path) -> list[Path]:
    files = []
    for suffix in sorted(DATA_SUFFIXES):
        files.extend(sorted(folder.rglob(f"*{suffix}")))

    return files


def _split_path_inputs(
    value: str | PathLike[str] | Sequence[str | PathLike[str]],
) -> tuple[list[Path], list[Path]]:
    #Split replay inputs from already-parsed PBP/frame inputs.
    replay_inputs: list[Path] = []
    tabular_inputs: list[Path] = []

    for item in _path_items(value):
        path = _resolve_input_path(item)

        if path.is_dir():
            #Replay folders are parsed first even if prior exports exist nearby.
            replay_files = sorted(path.rglob(f"*{REPLAY_SUFFIX}"))
            if replay_files:
                replay_inputs.append(path)
                continue

            data_files = _folder_data_files(path)
            if data_files:
                tabular_inputs.extend(data_files)
                continue

            raise FileNotFoundError(
                f"No replay, CSV, or Parquet files were found in {path}"
            )

        if not path.exists():
            raise FileNotFoundError(f"Stats input does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix == REPLAY_SUFFIX:
            replay_inputs.append(path)
            continue

        if suffix in DATA_SUFFIXES:
            tabular_inputs.append(path)
            continue

        raise ValueError(
            "stats path inputs must be replay, CSV, or Parquet files; "
            f"got {path}"
        )

    return replay_inputs, tabular_inputs


def _columns_for_file(path: Path) -> list[str] | None:
    try:
        if path.suffix.lower() == ".parquet":
            return pl.scan_parquet(path).collect_schema().names()

        #Read only the CSV header so large frame exports stay lazy.
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()

        return [column.strip() for column in header.rstrip("\r\n").split(",")]
    except Exception:
        return None


def _lazy_scan_files(paths: Sequence[Path]) -> pl.LazyFrame:
    #Group matching schemas so Polars can scan each batch efficiently.
    file_schemas: list[tuple[Path, tuple[str, ...]]] = []

    for path in paths:
        columns = _columns_for_file(path)
        if columns:
            file_schemas.append((path, tuple(columns)))

    if not file_schemas:
        raise ValueError("No readable CSV or Parquet stats inputs were found")

    has_xg = any("xG" in schema for _, schema in file_schemas)
    columns = requested_columns(has_xg)
    groups: dict[tuple[str, tuple[str, ...]], list[Path]] = {}

    for path, schema in file_schemas:
        groups.setdefault((path.suffix.lower(), schema), []).append(path)

    scans = []
    for (suffix, schema), group_paths in groups.items():
        if suffix == ".parquet":
            scan = pl.scan_parquet(group_paths)
        else:
            scan = pl.scan_csv(
                group_paths,
                infer_schema_length=10000,
                null_values=CSV_NULL_VALUES,
                low_memory=True,
                glob=False,
            )

        existing = set(schema)
        scans.append(
            scan.with_columns(
                [
                    pl.lit(None).alias(column)
                    for column in columns
                    if column not in existing
                ]
            ).select(columns)
        )

    return pl.concat(scans, how="vertical_relaxed")


def _parse_replay_inputs(
    replay_inputs: Sequence[Path],
    *,
    workers: int,
    parse_export: str | PathLike[str],
    force: bool,
    limit: int | None,
) -> list[Path]:
    from .parse import parse_replay

    if len(replay_inputs) == 1:
        replay_path: Path | list[Path] = replay_inputs[0]
    else:
        replay_path = list(replay_inputs)

    exported = parse_replay(
        replay_path=replay_path,
        export=parse_export,
        workers=workers,
        return_type="export",
        output="pbp",
        export_format="parquet",
        force=force,
        limit=limit,
    )

    return [Path(path) for path in exported]


def _lazy_frames_from_paths(
    value: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    workers: int,
    parse_export: str | PathLike[str],
    force: bool,
    limit: int | None,
) -> pl.LazyFrame:
    replay_inputs, tabular_inputs = _split_path_inputs(value)

    #Replay inputs become PBP parquet before entering the stats pipeline.
    if replay_inputs:
        tabular_inputs.extend(
            _parse_replay_inputs(
                replay_inputs,
                workers=workers,
                parse_export=parse_export,
                force=force,
                limit=limit,
            )
        )
    elif limit is not None:
        tabular_inputs = tabular_inputs[:limit]

    return _lazy_scan_files(tabular_inputs)


def _to_lazy_frames(frames: Any) -> pl.LazyFrame:
    if isinstance(frames, pl.LazyFrame):
        return frames

    if isinstance(frames, pl.DataFrame):
        return frames.lazy()

    try:
        import pandas as pd

        if isinstance(frames, pd.DataFrame):
            return pl.from_pandas(frames).lazy()
    except ImportError:
        pass

    if not isinstance(frames, list) or not frames:
        raise TypeError(
            "frames must be a Polars DataFrame, pandas DataFrame, "
            "list[dict], list[list[dict]], or list of DataFrames"
        )

    first = frames[0]

    if isinstance(first, dict):
        return pl.DataFrame(frames).lazy()

    if isinstance(first, list):
        return pl.concat(
            [pl.DataFrame(replay_rows) for replay_rows in frames],
            how="vertical_relaxed",
        ).lazy()

    if isinstance(first, pl.DataFrame):
        return pl.concat(frames, how="vertical_relaxed").lazy()

    try:
        import pandas as pd

        if isinstance(first, pd.DataFrame):
            return pl.concat(
                [pl.from_pandas(frame) for frame in frames],
                how="vertical_relaxed",
            ).lazy()
    except ImportError:
        pass

    raise TypeError(
        "frames must be a Polars DataFrame, pandas DataFrame, "
        "list[dict], list[list[dict]], or list of DataFrames"
    )


def _ensure_columns(rows, columns):
    existing = rows.collect_schema().names()

    return rows.with_columns(
        [
            pl.lit(None).alias(column)
            for column in columns
            if column not in existing
        ]
    ).select(columns)


def string_col(column):
    return pl.col(column).cast(pl.Utf8, strict=False).fill_null("")


def number_col(column):
    return pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0)


def flag_col(column):
    return string_col(column).str.to_lowercase().is_in(["true", "1", "yes"])


def count_if(condition, name):
    return condition.cast(pl.Int64).sum().alias(name)


def sum_if(condition, value, name):
    return pl.when(condition).then(value).otherwise(0.0).sum().alias(name)


def player_slot_frame(events, slot):
    team = "orange" if slot.startswith("orange") else "blue"
    team_name_col = "orange_team_name" if team == "orange" else "blue_team_name"

    return events.select(
        [
            pl.col("replay_id"),
            string_col(f"{slot}_id").alias("player_id"),
            string_col(f"{slot}_network_id").alias("network_id"),
            string_col(f"{slot}_name").alias("player_name"),
            pl.lit(team).alias("team"),
            string_col(team_name_col).alias("team_name"),
            string_col(f"{slot}_platform").alias("platform"),
            string_col(f"{slot}_rank").alias("rank"),
            string_col(f"{slot}_rank_tier").alias("rank_tier"),
            flag_col(f"{slot}_pro_player").alias("pro_player"),
            number_col(f"{slot}_mmr").alias("mmr"),
            string_col(f"{slot}_car_id").alias("car_id"),
            string_col(f"{slot}_car_name").alias("car_name"),
            flag_col(f"{slot}_is_bot").alias("is_bot"),
            number_col(f"{slot}_time_in_game").alias("time_in_game"),
            number_col("seconds_elapsed").alias("seconds_elapsed"),
        ]
    ).filter(pl.col("player_id") != "")


def player_frame(rows):
    players = pl.concat(
        [player_slot_frame(rows, slot) for slot in PLAYER_SLOTS],
        how="vertical_relaxed",
    )

    players = players.group_by(["replay_id", "player_id"]).agg(
        [
            pl.col("network_id").drop_nulls().first(),
            pl.col("player_name").drop_nulls().first(),
            pl.col("team").drop_nulls().first(),
            pl.col("team_name").drop_nulls().first(),
            pl.col("platform").drop_nulls().first(),
            pl.col("rank").drop_nulls().first(),
            pl.col("rank_tier").drop_nulls().first(),
            pl.col("pro_player").max(),
            pl.col("mmr").max(),
            pl.col("car_id").drop_nulls().first(),
            pl.col("car_name").drop_nulls().first(),
            pl.col("is_bot").max(),
            pl.col("time_in_game").max(),
            pl.col("seconds_elapsed").max().alias("_max_seconds_elapsed"),
        ]
    )

    return (
        players.with_columns(
            [
                pl.when(pl.col("time_in_game") > 0)
                .then(pl.col("time_in_game"))
                .otherwise(pl.col("_max_seconds_elapsed"))
                .alias("time_in_game")
            ]
        )
        .with_columns([(pl.col("time_in_game") / 60.0).alias("time_on_field")])
        .drop("_max_seconds_elapsed")
    )


def primary_event_stats(events, has_xg):
    shot = pl.col("event_type").is_in(["shot", "goal"])
    goal = pl.col("event_type") == "goal"

    expressions = [
        pl.len().alias("event_count"),
        count_if(shot, "shots"),
        count_if(goal, "goals"),
        count_if(pl.col("event_type") == "save", "saves"),
        count_if(pl.col("event_type") == "touch", "touches"),
        count_if(pl.col("event_type") == "pass", "passes"),
        count_if(pl.col("event_type") == "turnover", "turnovers"),
        count_if(pl.col("event_type") == "challenge", "challenge_wins"),
        count_if(pl.col("event_type") == "kickoff", "kickoff_wins"),
        count_if(pl.col("event_type") == "demo", "demos"),
        count_if(pl.col("event_type") == "bump", "bumps"),
        count_if(pl.col("event_type") == "entry", "entries"),
        count_if(pl.col("event_type") == "exit", "exits"),
        count_if(pl.col("event_type") == "retrieval", "retrievals"),
        count_if(pl.col("event_type") == "air_dribble", "air_dribbles"),
        count_if(pl.col("event_type") == "ground_dribble", "ground_dribbles"),
        count_if(pl.col("event_type") == "flick", "flicks"),
        count_if(pl.col("event_type") == "flip_reset", "flip_resets"),
        count_if(pl.col("event_type") == "boost-pickup", "boost_pickups"),
        count_if(
            (pl.col("event_type") == "boost-pickup")
            & (string_col("boost_pickup_type") == "small"),
            "small_boost_pickups",
        ),
        count_if(
            (pl.col("event_type") == "boost-pickup")
            & (string_col("boost_pickup_type") == "big"),
            "big_boost_pickups",
        ),
        count_if(
            (pl.col("event_type") == "boost-pickup")
            & (string_col("boost_pickup_type") == "reset"),
            "boost_resets",
        ),
        sum_if(
            pl.col("event_type") == "boost-pickup",
            number_col("boost_pickup_amount"),
            "boost_collected",
        ),
        count_if(shot & flag_col("off_demo"), "off_demo_shots"),
        count_if(shot & flag_col("off_kickoff"), "off_kickoff_shots"),
        count_if(shot & flag_col("off_challenge_win"), "off_challenge_win_shots"),
        count_if(shot & flag_col("off_bump"), "off_bump_shots"),
        count_if(shot & flag_col("off_air_dribble"), "off_air_dribble_shots"),
        count_if(shot & flag_col("off_ground_dribble"), "off_ground_dribble_shots"),
        count_if(shot & flag_col("off_flick"), "off_flick_shots"),
        count_if(shot & flag_col("pass_in_play"), "pass_in_play_shots"),
        count_if(shot & flag_col("aerialing"), "aerial_shots"),
        count_if(shot & flag_col("air_dribble"), "air_dribble_shots"),
        count_if(shot & flag_col("ground_dribble"), "ground_dribble_shots"),
        count_if(shot & flag_col("flick_shot"), "flick_shots"),
        count_if(shot & flag_col("rebound"), "rebound_shots"),
        count_if(shot & flag_col("off_flip_reset"), "off_flip_reset_shots"),
        count_if(shot & flag_col("off_wall"), "off_wall_shots"),
        count_if(shot & flag_col("off_ceiling"), "off_ceiling_shots"),
    ]

    if has_xg:
        xg = number_col("xG")
        expressions.extend(
            [
                sum_if(shot, xg, "expected_goals"),
                sum_if(goal, xg, "expected_goals_on_goals"),
                sum_if(shot & ~goal, xg, "expected_goals_on_misses"),
            ]
        )

    return (
        events.filter(string_col("event_player_1_id") != "")
        .group_by(
            [
                "replay_id",
                string_col("event_player_1_id").alias("player_id"),
            ]
        )
        .agg(expressions)
    )


def secondary_event_stats(events, has_xg):
    assist = (pl.col("event_type") == "goal") & (
        flag_col("official_assist") | (string_col("event_player_2_id") != "")
    )
    shot = pl.col("event_type").is_in(["shot", "goal"])

    expressions = [
        count_if(assist, "assists"),
        count_if(shot, "shot_assists"),
        count_if(pl.col("event_type") == "pass", "passes_received"),
        count_if(pl.col("event_type") == "demo", "demos_taken"),
        count_if(pl.col("event_type") == "bump", "bumps_taken"),
        count_if(pl.col("event_type") == "challenge", "challenge_losses"),
        count_if(pl.col("event_type") == "kickoff", "kickoff_losses"),
    ]

    if has_xg:
        expressions.append(sum_if(shot, number_col("xG"), "expected_assists"))

    return (
        events.filter(string_col("event_player_2_id") != "")
        .group_by(
            [
                "replay_id",
                string_col("event_player_2_id").alias("player_id"),
            ]
        )
        .agg(expressions)
    )


def team_event_stats(events, has_xg):
    shot = pl.col("event_type").is_in(["shot", "goal"])
    goal = pl.col("event_type") == "goal"

    expressions = [
        count_if(shot, "shots_for"),
        count_if(goal, "goals_for"),
        count_if(pl.col("event_type") == "demo", "demos_for"),
        count_if(pl.col("event_type") == "entry", "entries_for"),
        count_if(pl.col("event_type") == "exit", "exits_for"),
    ]

    if has_xg:
        expressions.append(sum_if(shot, number_col("xG"), "expected_goals_for"))

    return (
        events.filter(string_col("event_team").is_in(["blue", "orange"]))
        .group_by(
            [
                "replay_id",
                pl.col("event_team").alias("team"),
            ]
        )
        .agg(expressions)
    )


def calculate_player_replay_stats(
    frames: Any,
    *,
    workers: int = 4,
    parse_export: str | PathLike[str] = "data/frames",
    force: bool = False,
    limit: int | None = None,
) -> pl.DataFrame:
    """Build per-player replay stats from parsed play-by-play or frame data.

    Args:
        frames: Replay, CSV, Parquet, folder, or parsed replay data in a
            supported Polars, pandas, or list-based tabular shape.
        workers: Number of Rust parser workers to use when replay inputs need
            parsing.
        parse_export: Output folder for generated PBP Parquet files when
            replay inputs need parsing.
        force: Whether to overwrite existing parser exports for replay inputs.
        limit: Optional file or replay count limit.

    Returns:
        A Polars DataFrame containing one row per player per replay.
    """
    if _is_path_input(frames):
        rows = _lazy_frames_from_paths(
            frames,
            workers=workers,
            parse_export=parse_export,
            force=force,
            limit=limit,
        )
    else:
        rows = _to_lazy_frames(frames)

    schema = rows.collect_schema().names()

    has_xg = "xG" in schema
    columns = requested_columns(has_xg)
    rows = _ensure_columns(rows, columns)

    rows = rows.with_columns(
        [
            pl.when(string_col("replay_id") != "")
            .then(string_col("replay_id"))
            .otherwise(string_col("game_id"))
            .alias("replay_id"),
            string_col("event_type").alias("event_type"),
            string_col("event_team").alias("event_team"),
        ]
    )

    events = rows.filter(string_col("event_type") != "")

    players = player_frame(rows)
    primary = primary_event_stats(events, has_xg)
    secondary = secondary_event_stats(events, has_xg)
    team_for = team_event_stats(events, has_xg)

    team_against = team_for.rename(
        {
            "team": "opponent_team",
            **{
                column: column.replace("_for", "_against")
                for column in team_for.collect_schema().names()
                if column.endswith("_for")
            },
        }
    )

    stats = players.with_columns(
        [
            pl.when(pl.col("team") == "blue")
            .then(pl.lit("orange"))
            .otherwise(pl.lit("blue"))
            .alias("opponent_team")
        ]
    ).join(primary, on=["replay_id", "player_id"], how="left")

    stats = stats.join(secondary, on=["replay_id", "player_id"], how="left")
    stats = stats.join(team_for, on=["replay_id", "team"], how="left")
    stats = stats.join(team_against, on=["replay_id", "opponent_team"], how="left").drop(
        "opponent_team"
    )

    identity_columns = [
        "replay_id",
        "player_id",
        "network_id",
        "player_name",
        "team",
        "team_name",
        "platform",
        "rank",
        "rank_tier",
        "pro_player",
        "mmr",
        "car_id",
        "car_name",
        "is_bot",
        "time_in_game",
        "time_on_field",
    ]

    count_columns = [
        column
        for column in stats.collect_schema().names()
        if column not in identity_columns
    ]

    stats = stats.with_columns([pl.col(column).fill_null(0) for column in count_columns])

    if has_xg:
        stats = stats.with_columns(
            [
                pl.when(pl.col("shots") > 0)
                .then(pl.col("expected_goals") / pl.col("shots"))
                .otherwise(0.0)
                .alias("expected_goals_per_shot"),
                (pl.col("goals") - pl.col("expected_goals")).alias(
                    "goals_minus_expected_goals"
                ),
                (pl.col("goals_for") - pl.col("expected_goals_for")).alias(
                    "goals_for_minus_expected_goals_for"
                ),
                (pl.col("goals_against") - pl.col("expected_goals_against")).alias(
                    "goals_against_minus_expected_goals_against"
                ),
            ]
        )

    return stats.sort(["replay_id", "team", "player_name", "player_id"]).collect()


def calculate_stats(
    frames: Any,
    return_type: Literal["polars", "pandas", "list"] = "polars",
    export: str | PathLike[str] | None = None,
    workers: int = 4,
    parse_export: str | PathLike[str] = "data/frames",
    force: bool = False,
    limit: int | None = None,
):
    """Aggregate replay stats and optionally export them.

    Args:
        frames: Replay file, replay folder, one or more replay files, CSV or
            Parquet PBP/frame file, folder of CSV or Parquet PBP/frame files,
            one or more CSV or Parquet PBP/frame files, or parsed replay data
            in a supported tabular format.
        return_type: Whether to return Polars, pandas, or list output.
        export: Optional output path for CSV or Parquet export.
        workers: Number of Rust parser workers to use when replay inputs need
            parsing.
        parse_export: Output folder for generated PBP Parquet files when
            replay inputs need parsing.
        force: Whether to overwrite existing parser exports for replay inputs.
        limit: Optional file or replay count limit.

    Returns:
        Aggregated stats in the requested return format.
    """
    stats = calculate_player_replay_stats(
        frames,
        workers=workers,
        parse_export=parse_export,
        force=force,
        limit=limit,
    )

    if export is not None:
        export = Path(export)
        export.parent.mkdir(parents=True, exist_ok=True)

        if export.suffix == ".parquet":
            stats.write_parquet(export)
        else:
            stats.write_csv(export)

    if return_type == "polars":
        return stats

    if return_type == "pandas":
        return stats.to_pandas()

    if return_type == "list":
        return stats.to_dicts()

    raise ValueError("return_type must be one of: 'polars', 'pandas', 'list'")
