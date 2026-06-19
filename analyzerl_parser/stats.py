"""Player-level replay stat aggregation for parsed AnalyzeRL data."""

from os import PathLike
from typing import Any, Literal, Sequence

import polars as pl

from pathlib import Path

DATA_SUFFIXES = {".csv", ".parquet"}
REPLAY_SUFFIX = ".replay"
CSV_NULL_VALUES = ["", "NA", "NaN", "None", "null"]
DEFAULT_GROUP_BY = ["replay_id", "player_id"]
STATS_FILE_BATCH_SIZE = 250

PLAYER_SLOTS = [
    "blue_player_1",
    "blue_player_2",
    "blue_player_3",
    "orange_player_1",
    "orange_player_2",
    "orange_player_3",
]

IDENTITY_COLUMNS = [
    "replay_id",
    "player_id",
    "actor_id",
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
    "games_played",
]

DERIVED_XG_COLUMNS = [
    "expected_goals_per_shot",
    "goals_minus_expected_goals",
    "goals_for_minus_expected_goals_for",
    "goals_against_minus_expected_goals_against",
]

TOUCH_EVENT_TYPES = [
    "touch",
    "pass",
    "turnover",
    "challenge",
    "kickoff",
    "entry",
    "exit",
    "retrieval",
    "air-dribble",
    "air_dribble",
    "ground-dribble",
    "ground_dribble",
    "flick",
    "flip-reset",
    "shot",
    "goal",
    "save",
]


def requested_columns(
    has_xg: bool,
    extra_columns: Sequence[str] | None = None,
) -> list[str]:
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
        "frame_number",
        "seconds_elapsed",
        "delta",
        "event_length",
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
        "off_double_tap",
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
            "actor_id",
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
            "pos_x",
        ]:
            columns.append(f"{slot}_{field}")

    if extra_columns:
        columns.extend(extra_columns)

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


def _lazy_scan_files(
    paths: Sequence[Path],
    *,
    extra_columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    #Group matching schemas so Polars can scan each batch efficiently.
    file_schemas: list[tuple[Path, tuple[str, ...]]] = []

    for path in paths:
        columns = _columns_for_file(path)
        if columns:
            file_schemas.append((path, tuple(columns)))

    if not file_schemas:
        raise ValueError("No readable CSV or Parquet stats inputs were found")

    has_xg = any("xG" in schema for _, schema in file_schemas)
    columns = requested_columns(has_xg, extra_columns=extra_columns)
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
        present_columns = [column for column in columns if column in existing]
        scans.append(
            scan.select(present_columns).with_columns(
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
    extra_columns: Sequence[str] | None = None,
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

    return _lazy_scan_files(tabular_inputs, extra_columns=extra_columns)


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


def time_in_game_seconds_expr() -> pl.Expr:
    time_value = number_col("time_in_game")
    max_elapsed = number_col("_max_seconds_elapsed")

    return (
        pl.when(time_value > 0)
        .then(pl.when(time_value > 60.0).then(time_value).otherwise(time_value * 60.0))
        .otherwise(max_elapsed)
    )


def add_frame_delta_seconds(rows: pl.LazyFrame) -> pl.LazyFrame:
    return (
        rows.sort(["replay_id", "seconds_elapsed", "frame_number"])
        .with_columns(
            [
                (
                    pl.col("seconds_elapsed")
                    .rank(method="ordinal")
                    .over(["replay_id", "seconds_elapsed"])
                    == 1
                ).alias("_stats_first_time_row")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("_stats_first_time_row") & (number_col("delta") > 0))
                .then(number_col("delta"))
                .when(pl.col("_stats_first_time_row"))
                .then(
                    number_col("seconds_elapsed")
                    .diff()
                    .over("replay_id")
                    .fill_null(0.0)
                )
                .otherwise(0.0)
                .clip(0.0, 1.0)
                .alias("_stats_frame_delta_seconds")
            ]
        )
        .drop("_stats_first_time_row")
    )


def player_slot_rows(events, slot, *, include_inactive: bool = False):
    team = "orange" if slot.startswith("orange") else "blue"
    team_name_col = "orange_team_name" if team == "orange" else "blue_team_name"

    rows = (
        events.select(
            [
                pl.col("replay_id"),
                string_col(f"{slot}_id").alias("player_id"),
                string_col(f"{slot}_actor_id").alias("actor_id"),
                string_col(f"{slot}_network_id").alias("network_id"),
                string_col(f"{slot}_name").alias("player_name"),
                pl.lit(team).alias("team"),
                string_col("event_type").alias("event_type"),
                string_col("event_team").alias("event_team"),
                string_col("event_player_1_id").alias("event_player_1_id"),
                string_col("event_player_2_id").alias("event_player_2_id"),
                number_col("xG").alias("xG"),
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
                number_col("frame_number").alias("frame_number"),
                pl.col(f"{slot}_pos_x")
                .cast(pl.Float64, strict=False)
                .is_not_null()
                .alias("_has_position"),
                number_col("seconds_elapsed").alias("seconds_elapsed"),
                number_col("event_length").alias("event_length"),
                number_col("_stats_frame_delta_seconds").alias("frame_delta_seconds"),
                (string_col("event_type") != "").alias("_is_event_row"),
            ]
        )
        .with_columns(
            [
                pl.when(
                    (pl.col("event_player_1_id") == pl.col("player_id"))
                    & (pl.col("event_type").is_in(["game-join", "respawn"]))
                )
                .then(1)
                .when(
                    (pl.col("event_player_1_id") == pl.col("player_id"))
                    & (pl.col("event_type") == "game-leave")
                )
                .then(0)
                .when(
                    (pl.col("event_player_2_id") == pl.col("player_id"))
                    & (pl.col("event_type") == "demo")
                )
                .then(0)
                .otherwise(None)
                .alias("_presence_state"),
                pl.when(pl.col("event_type") == "game-join")
                .then(0)
                .when(pl.col("event_type") == "respawn")
                .then(0)
                .when(pl.col("event_type") == "game-leave")
                .then(2)
                .when(pl.col("event_type") == "demo")
                .then(2)
                .otherwise(1)
                .alias("_presence_sort"),
            ]
        )
        .sort(
            [
                "replay_id",
                "player_id",
                "frame_number",
                "seconds_elapsed",
                "_presence_sort",
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("car_id") != "")
                    | pl.col("is_bot")
                    | (pl.col("time_in_game") > 0)
                ).alias("_legacy_active_evidence")
            ]
        )
        .with_columns(
            [
                pl.col("_has_position")
                .max()
                .over(["replay_id", "player_id"])
                .alias("_slot_has_position"),
                pl.col("event_type")
                .is_in(["game-join", "game-leave", "respawn"])
                .max()
                .over("replay_id")
                .alias("_replay_has_presence_events"),
                pl.col("_presence_state")
                .is_not_null()
                .max()
                .over(["replay_id", "player_id"])
                .alias("_slot_has_presence_events"),
                pl.col("_presence_state")
                .forward_fill()
                .fill_null(0)
                .over(["replay_id", "player_id"])
                .alias("_presence_active_after"),
                pl.col("seconds_elapsed")
                .max()
                .over("replay_id")
                .alias("_replay_end_seconds"),
            ]
        )
        .with_columns(
            [
                pl.col("_presence_active_after")
                .shift(1)
                .fill_null(0)
                .over(["replay_id", "player_id"])
                .alias("_presence_active_before")
            ]
        )
        .with_columns(
            [
                (pl.col("car_id") != "")
                .alias("_has_static_car")
            ]
        )
        .with_columns(
            [
                (
                    pl.when(pl.col("_slot_has_presence_events"))
                    .then(pl.col("_presence_active_after") > 0)
                    .when(pl.col("_replay_has_presence_events"))
                    .then(False)
                    .otherwise(
                        pl.col("_has_position")
                        | (
                            ~pl.col("_slot_has_position")
                            & pl.col("_legacy_active_evidence")
                        )
                    )
                ).alias("_active_on_row"),
                (
                    pl.when(pl.col("_slot_has_presence_events"))
                    .then(
                        pl.when(pl.col("_presence_state") == 0)
                        .then(pl.col("_presence_active_before") > 0)
                        .otherwise(pl.col("_presence_active_after") > 0)
                    )
                    .when(pl.col("_replay_has_presence_events"))
                    .then(False)
                    .otherwise(
                        pl.col("_has_position")
                        | (
                            ~pl.col("_slot_has_position")
                            & pl.col("_legacy_active_evidence")
                        )
                    )
                ).alias("_active_for_event_context"),
            ]
        )
    )

    if include_inactive:
        return rows.filter(pl.col("player_id") != "")

    return rows.filter((pl.col("player_id") != "") & pl.col("_active_on_row"))


def player_slot_frame(events, slot):
    rows = player_slot_rows(events, slot, include_inactive=True)
    active_rows = rows.filter(pl.col("_active_on_row"))

    presence_time = (
        rows.filter(pl.col("_slot_has_presence_events") & pl.col("_presence_state").is_not_null())
        .with_columns(
            [
                pl.col("seconds_elapsed")
                .shift(-1)
                .over(["replay_id", "player_id"])
                .alias("_next_presence_seconds"),
                (pl.col("seconds_elapsed") + pl.col("event_length"))
                .alias("_row_end_seconds"),
            ]
        )
        .with_columns(
            [
                (
                    pl.coalesce(
                        [
                            "_next_presence_seconds",
                            pl.max_horizontal(
                                "_replay_end_seconds",
                                "_row_end_seconds",
                            ),
                        ]
                    )
                    - pl.col("seconds_elapsed")
                )
                .clip(0.0, 36000.0)
                .alias("_presence_interval_seconds")
            ]
        )
        .group_by(["replay_id", "player_id"])
        .agg(
            [
                pl.when(pl.col("_presence_state") == 1)
                .then(pl.col("_presence_interval_seconds"))
                .otherwise(0.0)
                .sum()
                .alias("_presence_time_seconds")
            ]
        )
    )

    return (
        active_rows
        .group_by(["replay_id", "player_id"])
        .agg(
            [
                pl.col("actor_id").drop_nulls().first(),
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
                pl.col("_has_position").max(),
                pl.col("_has_static_car").max(),
                pl.col("_slot_has_presence_events").max(),
                pl.col("time_in_game").max(),
                pl.col("seconds_elapsed").max().alias("_max_seconds_elapsed"),
                pl.col("event_length").sum().alias("_event_length_seconds"),
                pl.col("frame_delta_seconds").sum().alias("_frame_time_seconds"),
                pl.len().alias("_row_count"),
                pl.col("_is_event_row").sum().alias("_event_row_count"),
            ]
        )
        .join(presence_time, on=["replay_id", "player_id"], how="left")
        .with_columns(number_col("_presence_time_seconds").alias("_presence_time_seconds"))
    )


def event_presence_player_frame(events):
    primary_presence = events.select(
        [
            pl.col("replay_id"),
            string_col("event_player_1_id").alias("player_id"),
            string_col("event_player_1_name").alias("player_name"),
            string_col("event_player_1_team").alias("team"),
            number_col("frame_number").alias("frame_number"),
            number_col("seconds_elapsed").alias("seconds_elapsed"),
            number_col("seconds_elapsed")
            .max()
            .over("replay_id")
            .alias("_replay_end_seconds"),
            number_col("event_length").alias("event_length"),
            pl.when(pl.col("event_type").is_in(["game-join", "respawn"]))
            .then(1)
            .when(pl.col("event_type") == "game-leave")
            .then(0)
            .otherwise(None)
            .alias("_presence_state"),
            pl.when(pl.col("event_type") == "game-leave")
            .then(2)
            .otherwise(0)
            .alias("_presence_sort"),
        ]
    )
    demo_presence = events.select(
        [
            pl.col("replay_id"),
            string_col("event_player_2_id").alias("player_id"),
            string_col("event_player_2_name").alias("player_name"),
            string_col("event_player_2_team").alias("team"),
            number_col("frame_number").alias("frame_number"),
            number_col("seconds_elapsed").alias("seconds_elapsed"),
            number_col("seconds_elapsed")
            .max()
            .over("replay_id")
            .alias("_replay_end_seconds"),
            number_col("event_length").alias("event_length"),
            pl.when(pl.col("event_type") == "demo")
            .then(0)
            .otherwise(None)
            .alias("_presence_state"),
            pl.lit(2).alias("_presence_sort"),
        ]
    )
    presence_rows = (
        pl.concat([primary_presence, demo_presence], how="vertical_relaxed")
        .filter((pl.col("player_id") != "") & pl.col("_presence_state").is_not_null())
        .sort(
            [
                "replay_id",
                "player_id",
                "frame_number",
                "seconds_elapsed",
                "_presence_sort",
            ]
        )
        .with_columns(
            [
                pl.col("seconds_elapsed")
                .shift(-1)
                .over(["replay_id", "player_id"])
                .alias("_next_presence_seconds"),
                (pl.col("seconds_elapsed") + pl.col("event_length")).alias(
                    "_row_end_seconds"
                ),
            ]
        )
        .with_columns(
            [
                (
                    pl.coalesce(
                        [
                            "_next_presence_seconds",
                            pl.max_horizontal("_replay_end_seconds", "_row_end_seconds"),
                        ]
                    )
                    - pl.col("seconds_elapsed")
                )
                .clip(0.0, 36000.0)
                .alias("_presence_interval_seconds")
            ]
        )
    )

    return presence_rows.group_by(["replay_id", "player_id"]).agg(
        [
            pl.lit(None).alias("actor_id"),
            pl.lit(None).alias("network_id"),
            pl.col("player_name").drop_nulls().first(),
            pl.col("team").drop_nulls().first(),
            pl.lit(None).alias("team_name"),
            pl.lit(None).alias("platform"),
            pl.lit(None).alias("rank"),
            pl.lit(None).alias("rank_tier"),
            pl.lit(False).alias("pro_player"),
            pl.lit(0.0).alias("mmr"),
            pl.lit(None).alias("car_id"),
            pl.lit(None).alias("car_name"),
            pl.lit(False).alias("is_bot"),
            pl.lit(False).alias("_has_position"),
            pl.lit(False).alias("_has_static_car"),
            pl.lit(True).alias("_slot_has_presence_events"),
            pl.lit(0.0).alias("time_in_game"),
            pl.col("seconds_elapsed").max().alias("_max_seconds_elapsed"),
            pl.lit(0.0).alias("_event_length_seconds"),
            pl.lit(0.0).alias("_frame_time_seconds"),
            pl.len().alias("_row_count"),
            pl.len().alias("_event_row_count"),
            pl.when(pl.col("_presence_state") == 1)
            .then(pl.col("_presence_interval_seconds"))
            .otherwise(0.0)
            .sum()
            .alias("_presence_time_seconds"),
        ]
    )


def player_frame(rows):
    players = pl.concat(
        [player_slot_frame(rows, slot) for slot in PLAYER_SLOTS]
        + [event_presence_player_frame(rows)],
        how="vertical_relaxed",
    ).filter(pl.col("player_id") != "")

    players = players.group_by(["replay_id", "player_id"]).agg(
        [
            pl.col("network_id").drop_nulls().first(),
            pl.col("actor_id").drop_nulls().first(),
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
            pl.col("_has_position").max(),
            pl.col("_has_static_car").max(),
            pl.col("_slot_has_presence_events").max(),
            pl.col("_presence_time_seconds").max(),
            pl.col("time_in_game").max(),
            pl.col("_max_seconds_elapsed").max(),
            pl.col("_event_length_seconds").max(),
            pl.col("_frame_time_seconds").max(),
            pl.col("_row_count").max(),
            pl.col("_event_row_count").max(),
        ]
    )

    return (
        players.with_columns(
            [
                time_in_game_seconds_expr().alias("_fallback_time_on_field_seconds")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("_slot_has_presence_events") & (pl.col("_presence_time_seconds") > 0))
                .then(pl.col("_presence_time_seconds"))
                .when(
                    (pl.col("_frame_time_seconds") > 0)
                    & (
                        (pl.col("_event_row_count") == 0)
                        | (pl.col("_row_count") > pl.col("_event_row_count") * 2)
                    )
                )
                .then(pl.col("_frame_time_seconds"))
                .when(pl.col("_event_length_seconds") > 0)
                .then(pl.col("_event_length_seconds"))
                .otherwise(pl.col("_fallback_time_on_field_seconds"))
                .alias("_time_on_field_seconds")
            ]
        )
        .with_columns(
            [
                pl.col("_time_on_field_seconds").alias("time_in_game"),
                (pl.col("_time_on_field_seconds") / 60.0).alias("time_on_field"),
            ]
        )
        .drop(
            [
                "_max_seconds_elapsed",
                "_event_length_seconds",
                "_frame_time_seconds",
                "_row_count",
                "_event_row_count",
                "_has_position",
                "_has_static_car",
                "_slot_has_presence_events",
                "_presence_time_seconds",
                "_time_on_field_seconds",
                "_fallback_time_on_field_seconds",
            ]
        )
    )


def primary_event_stats(events, has_xg):
    shot = pl.col("event_type").is_in(["shot", "goal"])
    goal = pl.col("event_type") == "goal"
    shot_breakdowns = [
        ("off_demo", flag_col("off_demo")),
        ("off_kickoff", flag_col("off_kickoff")),
        ("off_challenge_win", flag_col("off_challenge_win")),
        ("off_bump", flag_col("off_bump")),
        ("off_air_dribble", flag_col("off_air_dribble")),
        ("off_ground_dribble", flag_col("off_ground_dribble")),
        ("off_flick", flag_col("off_flick")),
        ("off_double_tap", flag_col("off_double_tap")),
        ("pass_in_play", flag_col("pass_in_play")),
        ("aerial", flag_col("aerialing")),
        ("air_dribble", flag_col("air_dribble")),
        ("ground_dribble", flag_col("ground_dribble")),
        ("flick", flag_col("flick_shot")),
        ("rebound", flag_col("rebound")),
        ("off_flip_reset", flag_col("off_flip_reset")),
        ("off_wall", flag_col("off_wall")),
        ("off_ceiling", flag_col("off_ceiling")),
    ]

    expressions = [
        count_if(shot, "shots"),
        count_if(goal, "goals"),
        count_if(pl.col("event_type") == "save", "saves"),
        count_if(pl.col("event_type").is_in(TOUCH_EVENT_TYPES), "touches"),
        count_if(pl.col("event_type") == "pass", "passes"),
        count_if(pl.col("event_type") == "turnover", "turnovers"),
        count_if(pl.col("event_type") == "challenge", "challenge_wins"),
        count_if(pl.col("event_type") == "kickoff", "kickoff_wins"),
        count_if(pl.col("event_type") == "shadow", "shadows"),
        count_if(pl.col("event_type") == "press", "presses"),
        count_if(pl.col("event_type") == "demo", "demos_applied"),
        count_if(pl.col("event_type") == "bump", "bumps"),
        count_if(pl.col("event_type") == "entry", "entries"),
        count_if(pl.col("event_type") == "exit", "exits"),
        count_if(pl.col("event_type") == "retrieval", "retrievals"),
        count_if(pl.col("event_type").is_in(["air-dribble", "air_dribble"]), "air_dribbles"),
        count_if(pl.col("event_type").is_in(["ground-dribble", "ground_dribble"]), "ground_dribbles"),
        count_if(pl.col("event_type") == "flick", "flicks"),
        count_if(pl.col("event_type") == "flip-reset", "flip_resets"),
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
        sum_if(
            pl.col("event_type") == "boost-pickup",
            number_col("boost_pickup_amount"),
            "boost_collected",
        ),
    ]
    expressions.extend(
        count_if(shot & condition, f"{name}_shots")
        for name, condition in shot_breakdowns
    )

    if has_xg:
        xg = number_col("xG")
        expressions.extend(
            [
                sum_if(shot, xg, "expected_goals"),
            ]
        )
        expressions.extend(
            sum_if(shot & condition, xg, f"{name}_expected_goals")
            for name, condition in shot_breakdowns
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
        count_if(pl.col("event_type") == "shadow", "shadows_taken"),
        count_if(pl.col("event_type") == "press", "presses_taken"),
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


def active_player_team_event_stats(rows, has_xg):
    active_events = pl.concat(
        [
            player_slot_rows(rows, slot, include_inactive=True)
            for slot in PLAYER_SLOTS
        ],
        how="vertical_relaxed",
    ).filter(
        (pl.col("player_id") != "")
        & string_col("event_team").is_in(["blue", "orange"])
    )

    shot = pl.col("event_type").is_in(["shot", "goal"])
    goal = pl.col("event_type") == "goal"
    same_team = pl.col("event_team") == pl.col("team")
    opposing_team = pl.col("event_team") != pl.col("team")
    active = pl.col("_active_for_event_context")

    expressions = [
        count_if(active & shot & same_team, "shots_for"),
        count_if(active & shot & opposing_team, "shots_against"),
        count_if(active & goal & same_team, "goals_for"),
        count_if(active & goal & opposing_team, "goals_against"),
        count_if(
            active & (pl.col("event_type") == "demo") & same_team,
            "demos_applied_for",
        ),
        count_if(
            active & (pl.col("event_type") == "demo") & opposing_team,
            "demos_taken_against",
        ),
        count_if(active & (pl.col("event_type") == "entry") & same_team, "entries_for"),
        count_if(active & (pl.col("event_type") == "entry") & opposing_team, "entries_against"),
        count_if(active & (pl.col("event_type") == "exit") & same_team, "exits_for"),
        count_if(active & (pl.col("event_type") == "exit") & opposing_team, "exits_against"),
    ]

    if has_xg:
        xg = number_col("xG")
        expressions.extend(
            [
                sum_if(active & shot & same_team, xg, "expected_goals_for"),
                sum_if(active & shot & opposing_team, xg, "expected_goals_against"),
            ]
        )

    return active_events.group_by(["replay_id", "player_id"]).agg(expressions)


def normalize_group_by(group_by: Sequence[str] | str | None) -> list[str]:
    if group_by is None:
        return list(DEFAULT_GROUP_BY)

    if isinstance(group_by, str):
        columns = [group_by]
    else:
        columns = list(group_by)

    columns = [str(column) for column in columns]
    if not columns:
        raise ValueError("group_by must contain at least one column")

    return list(dict.fromkeys(columns))


def recompute_xg_derived_columns(stats: pl.DataFrame, has_xg: bool) -> pl.DataFrame:
    if not has_xg:
        return stats

    columns = (
        stats.collect_schema().names()
        if isinstance(stats, pl.LazyFrame)
        else stats.columns
    )
    required = {
        "shots",
        "goals",
        "expected_goals",
        "goals_for",
        "expected_goals_for",
        "goals_against",
        "expected_goals_against",
    }
    if not required.issubset(columns):
        return stats

    return stats.with_columns(
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


def aggregate_stats(
    stats: pl.DataFrame | pl.LazyFrame,
    *,
    group_by: Sequence[str] | str | None,
    has_xg: bool,
) -> pl.DataFrame | pl.LazyFrame:
    group_columns = normalize_group_by(group_by)
    columns = (
        stats.collect_schema().names()
        if isinstance(stats, pl.LazyFrame)
        else stats.columns
    )
    missing = [column for column in group_columns if column not in columns]
    if missing:
        raise ValueError(f"group_by columns are not present in stats output: {missing}")

    metric_columns = [
        column
        for column in columns
        if column not in IDENTITY_COLUMNS and column not in DERIVED_XG_COLUMNS
    ]
    aggregations = [
        pl.col(column).sum().alias(column)
        for column in metric_columns
        if column not in group_columns and column != "games_played"
    ]
    if "games_played" in columns and "games_played" not in group_columns:
        if "replay_id" in columns:
            aggregations.append(pl.col("replay_id").n_unique().alias("games_played"))
        else:
            aggregations.append(pl.col("games_played").sum().alias("games_played"))

    if "time_in_game" not in group_columns:
        aggregations.append(pl.col("time_in_game").sum().alias("time_in_game"))
    if "time_on_field" not in group_columns:
        aggregations.append(pl.col("time_on_field").sum().alias("time_on_field"))

    for column in IDENTITY_COLUMNS:
        if column in group_columns or column in {
            "time_in_game",
            "time_on_field",
            "games_played",
        }:
            continue
        if column not in columns:
            continue
        aggregations.append(pl.col(column).drop_nulls().first().alias(column))

    aggregated = stats.group_by(group_columns).agg(aggregations)
    aggregated_columns = (
        aggregated.collect_schema().names()
        if isinstance(aggregated, pl.LazyFrame)
        else aggregated.columns
    )

    ordered_columns = list(group_columns)
    ordered_columns.extend(
        column
        for column in IDENTITY_COLUMNS
        if column in aggregated_columns and column not in ordered_columns
    )
    ordered_columns.extend(
        column
        for column in aggregated_columns
        if column not in ordered_columns
    )

    aggregated = aggregated.select(ordered_columns)
    return recompute_xg_derived_columns(aggregated, has_xg)


def add_rate_stats(stats: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    schema = stats.collect_schema() if isinstance(stats, pl.LazyFrame) else stats.schema
    columns = schema.names() if isinstance(stats, pl.LazyFrame) else stats.columns
    rate_source_columns = [
        column
        for column in columns
        if column not in IDENTITY_COLUMNS
        and column not in DERIVED_XG_COLUMNS
        and column != "games_played"
        and schema[column].is_numeric()
    ]

    return stats.with_columns(
        [
            pl.when(pl.col("time_on_field") > 0)
            .then(pl.col(column) * 5.0 / pl.col("time_on_field"))
            .otherwise(0.0)
            .alias(f"{column}_per_five")
            for column in rate_source_columns
        ]
        + [
            pl.when(pl.col("games_played") > 0)
            .then(pl.col(column) / pl.col("games_played"))
            .otherwise(0.0)
            .alias(f"{column}_per_game")
            for column in rate_source_columns
            if "games_played" in columns
        ]
    )


def finish_stats(
    stats: pl.DataFrame | pl.LazyFrame,
    *,
    group_by: Sequence[str] | str | None,
    rates: bool,
    has_xg: bool,
) -> pl.DataFrame:
    stats = aggregate_stats(stats, group_by=group_by, has_xg=has_xg)

    if rates:
        stats = add_rate_stats(stats)

    stats = stats.collect() if isinstance(stats, pl.LazyFrame) else stats
    sort_columns = [
        column
        for column in ["replay_id", "team", "player_name", "player_id"]
        if column in stats.columns
    ]
    return stats.sort(sort_columns) if sort_columns else stats


def calculate_stats_from_lazy_rows(
    rows: pl.LazyFrame,
    *,
    group_by: Sequence[str] | str | None,
    rates: bool,
    xg_model_path: str | PathLike[str] | None,
    xg_columns: Sequence[str] | None,
) -> pl.DataFrame:
    if xg_model_path is not None:
        from .xg import apply_xg_to_pbp

        rows = apply_xg_to_pbp(rows, xg_model_path).lazy()

    schema = rows.collect_schema().names()

    has_xg = "xG" in schema
    columns = requested_columns(has_xg, extra_columns=xg_columns)
    rows = _ensure_columns(rows, columns)
    if not has_xg:
        rows = rows.with_columns(pl.lit(None, dtype=pl.Float64).alias("xG"))

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
    has_presence_events = (
        events.select(
            pl.col("event_type")
            .is_in(["game-join", "game-leave", "respawn"])
            .any()
            .alias("has_presence_events")
        )
        .collect()
        .item()
    )

    if has_presence_events:
        rows = rows.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("_stats_frame_delta_seconds")
        )
        events = rows.filter(string_col("event_type") != "")
        player_source = events
    else:
        rows = add_frame_delta_seconds(rows)
        events = rows.filter(string_col("event_type") != "")
        player_source = rows

    players = player_frame(player_source)
    primary = primary_event_stats(events, has_xg)
    secondary = secondary_event_stats(events, has_xg)
    active_team = active_player_team_event_stats(player_source, has_xg)

    stats = players.with_columns(
        [
            pl.lit(1, dtype=pl.Int64).alias("games_played"),
        ]
    ).join(primary, on=["replay_id", "player_id"], how="left")

    stats = stats.join(secondary, on=["replay_id", "player_id"], how="left")
    stats = stats.join(active_team, on=["replay_id", "player_id"], how="left")

    count_columns = [
        column
        for column in stats.collect_schema().names()
        if column not in IDENTITY_COLUMNS
    ]

    stats = stats.with_columns([pl.col(column).fill_null(0) for column in count_columns])
    stats = recompute_xg_derived_columns(stats, has_xg)
    return finish_stats(stats, group_by=group_by, rates=rates, has_xg=has_xg)


def calculate_stats_from_path_batches(
    frames: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    group_by: Sequence[str] | str | None,
    rates: bool,
    workers: int,
    parse_export: str | PathLike[str],
    force: bool,
    limit: int | None,
    xg_model_path: str | PathLike[str] | None,
    xg_columns: Sequence[str] | None,
) -> pl.DataFrame | None:
    replay_inputs, tabular_inputs = _split_path_inputs(frames)

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

    if len(tabular_inputs) <= STATS_FILE_BATCH_SIZE:
        return None

    partials = []
    for start in range(0, len(tabular_inputs), STATS_FILE_BATCH_SIZE):
        batch = tabular_inputs[start : start + STATS_FILE_BATCH_SIZE]
        rows = _lazy_scan_files(batch, extra_columns=xg_columns)
        partials.append(
            calculate_stats_from_lazy_rows(
                rows,
                group_by=DEFAULT_GROUP_BY,
                rates=False,
                xg_model_path=xg_model_path,
                xg_columns=xg_columns,
            )
        )

    combined = pl.concat(partials, how="vertical_relaxed").lazy()
    has_xg = any("expected_goals" in partial.columns for partial in partials)
    return finish_stats(combined, group_by=group_by, rates=rates, has_xg=has_xg)


def calculate_player_replay_stats(
    frames: Any,
    *,
    group_by: Sequence[str] | str | None = None,
    rates: bool = False,
    workers: int = 4,
    parse_export: str | PathLike[str] = "data/frames",
    force: bool = False,
    limit: int | None = None,
    xg_model_path: str | PathLike[str] | None = None,
) -> pl.DataFrame:
    """Build per-player replay stats from parsed play-by-play or frame data.

    Args:
        frames: Replay, CSV, Parquet, folder, or parsed replay data in a
            supported Polars, pandas, or list-based tabular shape.
        group_by: Output grouping columns. Defaults to one row per replay and
            player.
        rates: Whether to add per-five-minute and per-game rate columns.
        workers: Number of Rust parser workers to use when replay inputs need
            parsing.
        parse_export: Output folder for generated PBP Parquet files when
            replay inputs need parsing.
        force: Whether to overwrite existing parser exports for replay inputs.
        limit: Optional file or replay count limit.
        xg_model_path: Optional saved xG model file or folder. When provided,
            source rows are scored before stats are aggregated.

    Returns:
        A Polars DataFrame containing one row per player per replay.
    """
    xg_columns = None
    if xg_model_path is not None:
        from .xg import xg_model_source_columns

        xg_columns = xg_model_source_columns()

    if _is_path_input(frames):
        batched = calculate_stats_from_path_batches(
            frames,
            group_by=group_by,
            rates=rates,
            workers=workers,
            parse_export=parse_export,
            force=force,
            limit=limit,
            xg_model_path=xg_model_path,
            xg_columns=xg_columns,
        )
        if batched is not None:
            return batched

    if _is_path_input(frames):
        rows = _lazy_frames_from_paths(
            frames,
            workers=workers,
            parse_export=parse_export,
            force=force,
            limit=limit,
            extra_columns=xg_columns,
        )
    else:
        rows = _to_lazy_frames(frames)

    return calculate_stats_from_lazy_rows(
        rows,
        group_by=group_by,
        rates=rates,
        xg_model_path=xg_model_path,
        xg_columns=xg_columns,
    )


def calculate_stats(
    frames: Any,
    return_type: Literal["polars", "pandas", "list"] = "polars",
    export: str | PathLike[str] | None = None,
    group_by: Sequence[str] | str | None = None,
    rates: bool = False,
    workers: int = 4,
    parse_export: str | PathLike[str] = "data/frames",
    force: bool = False,
    limit: int | None = None,
    xg_model_path: str | PathLike[str] | None = None,
):
    """Aggregate replay stats and optionally export them.

    Args:
        frames: Replay file, replay folder, one or more replay files, CSV or
            Parquet PBP/frame file, folder of CSV or Parquet PBP/frame files,
            one or more CSV or Parquet PBP/frame files, or parsed replay data
            in a supported tabular format.
        return_type: Whether to return Polars, pandas, or list output.
        export: Optional output path for CSV or Parquet export.
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
        Aggregated stats in the requested return format.
    """
    stats = calculate_player_replay_stats(
        frames,
        group_by=group_by,
        rates=rates,
        workers=workers,
        parse_export=parse_export,
        force=force,
        limit=limit,
        xg_model_path=xg_model_path,
    )

    if export is not None:
        export = _resolve_input_path(export)
        export.parent.mkdir(parents=True, exist_ok=True)

        suffix = export.suffix.lower()
        if suffix == ".parquet":
            if export.exists():
                export.unlink()
            stats.write_parquet(export)
        elif suffix == ".csv":
            if export.exists():
                export.unlink()
            stats.write_csv(export)
        else:
            raise ValueError("export path must end in .csv or .parquet")

    if return_type == "polars":
        return stats

    if return_type == "pandas":
        return stats.to_pandas()

    if return_type == "list":
        return stats.to_dicts()

    raise ValueError("return_type must be one of: 'polars', 'pandas', 'list'")
