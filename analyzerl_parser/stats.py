import gc

import polars as pl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from typing import Any, Literal, Sequence

DATA_SUFFIXES = {".csv", ".parquet"}
REPLAY_SUFFIX = ".replay"
CSV_NULL_VALUES = ["", "NA", "NaN", "None", "null"]
DEFAULT_GROUP_BY = ["replay_id", "player_id"]
STATS_FILE_BATCH_SIZE = 32
STATS_BATCH_BYTES = 128 * 1024 * 1024
STATS_SINGLE_SCAN_FILE_LIMIT = 1
STATS_MAX_PARALLEL_BATCHES = 4
FIELD_THIRD_Y = 5120.0 / 3.0
SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
CROSSBAR_HEIGHT = 642.775
GROUND_HEIGHT = 20.0
CEILING_Z = 2044.0
DOUBLE_COMMIT_BALL_DISTANCE = 1100.0
DOUBLE_COMMIT_TEAMMATE_DISTANCE = 1300.0

PLAYER_SLOTS = [
    "blue_player_1",
    "blue_player_2",
    "blue_player_3",
    "blue_player_4",
    "orange_player_1",
    "orange_player_2",
    "orange_player_3",
    "orange_player_4",
]
FRAME_ONLY_TIME_COLUMNS = [
    "time_behind_ball",
    "time_in_front_of_ball",
    "time_closest_to_ball",
    "time_furthest_from_ball",
    "time_first_man",
    "time_second_man",
    "time_third_man",
    "time_fourth_man",
    "time_side_wall",
    "time_offensive_back_wall",
    "time_defensive_back_wall",
    "time_ceiling",
]
FRAME_ONLY_EVENT_TIME_COLUMNS = [
    "time_air_dribble",
    "time_ground_dribble",
]
FRAME_ONLY_VALUE_COLUMNS = [
    "rotation_tasks",
    "rotations_filled",
    "rotations_cut",
    "rotations_stalled",
    "rotations_fill_small_boost",
    "rotations_fill_big_boost",
    "rotations_fill_boost_collected",
    "avg_rotations_fill_small_boost",
    "avg_rotations_fill_big_boost",
    "time_rotations_fill",
    "time_rotations_cut",
    "time_rotations_stalled",
    "avg_time_rotations_fill",
    "avg_time_rotations_cut",
    "avg_time_rotations_stalled",
    "avg_reaction_time",
    "reaction_time_total",
    "reaction_time_count",
    "avg_distance_to_teammates",
    "avg_distance_to_teammate_closest_to_ball",
    "avg_distance_to_teammate_further_from_ball",
    "avg_distance_from_ball",
    "avg_distance_from_offensive_net",
    "avg_distance_from_defensive_net",
    "avg_angle_to_teammates",
    "avg_angle_to_teammate_closest_to_ball",
    "avg_angle_to_teammate_further_from_ball",
    "avg_angle_from_ball",
    "avg_angle_from_offensive_net",
    "avg_angle_from_defensive_net",
]

FRAME_TIME_COLUMNS = [
    "time_zero_boost",
    "time_full_boost",
    "time_zero_to_quarter_boost",
    "time_quarter_to_half_boost",
    "time_half_to_three_quarters_boost",
    "time_three_quarters_to_full_boost",
    "time_offensive_third",
    "time_neutral_third",
    "time_defensive_third",
    "time_low_air",
    "time_high_air",
    "time_ground",
    "time_possession",
    "time_holding_powerslide",
    "time_holding_boost",
    "time_wasting_boost",
    *FRAME_ONLY_TIME_COLUMNS,
]

FRAME_VALUE_COLUMNS = [
    "distance_traveled",
    "avg_speed",
    "max_speed",
    "powerslide_presses",
    "boost_amount_wasted",
    *FRAME_ONLY_VALUE_COLUMNS,
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
    "decal_id",
    "wheels_id",
    "boost_id",
    "antenna_id",
    "topper_id",
    "engine_audio_id",
    "trail_id",
    "goal_explosion_id",
    "primary_paint_finish_id",
    "accent_paint_finish_id",
    "camera_fov",
    "camera_height",
    "camera_angle",
    "camera_distance",
    "camera_stiffness",
    "camera_swivel",
    "camera_transition",
    "is_bot",
    "time_in_game",
    "time_on_field",
    "games_played",
]

DERIVED_XG_COLUMNS = [
    "expected_goals_per_shot",
    "expected_goals_per_shot_attempt",
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
    "missed-shot",
    "missed-pass",
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
        "event_player_3_id",
        "event_player_3_name",
        "event_player_3_team",
        "frame_number",
        "frame_has_event",
        "seconds_elapsed",
        "delta",
        "event_length",
        "event_duration",
        "official_shot",
        "official_goal",
        "official_assist",
        "official_save",
        "official_shot_count",
        "official_goal_count",
        "official_assist_count",
        "official_save_count",
        "controlled",
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
        "off_pass",
        "off_fake",
        "off_whiff",
        "off_rotation_cut",
        "aerialing",
        "air_dribble",
        "ground_dribble",
        "flick_shot",
        "rebound",
        "double_tap",
        "flip_reset",
        "off_flip_reset",
        "off_wall",
        "off_ceiling",
        "ball_pos_x",
        "ball_pos_y",
        "ball_pos_z",
        "ball_vel_x",
        "ball_vel_y",
        "ball_vel_z",
        "ball_speed_from_last_event",
        "ball_angle_from_last_event",
        "ball_distance_from_last_event",
        "distance_to_goal",
        "team_size",
        "previous_hit_frame_number",
        "next_hit_frame_number",
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
            "decal_id",
            "wheels_id",
            "boost_id",
            "antenna_id",
            "topper_id",
            "engine_audio_id",
            "trail_id",
            "goal_explosion_id",
            "primary_paint_finish_id",
            "accent_paint_finish_id",
            "camera_fov",
            "camera_height",
            "camera_angle",
            "camera_distance",
            "camera_stiffness",
            "camera_swivel",
            "camera_transition",
            "is_bot",
            "score",
            "time_in_game",
            "pos_x",
            "pos_y",
            "pos_z",
            "vel_x",
            "vel_y",
            "vel_z",
            "boost",
            "boost_active",
            "throttle",
            "steer",
            "handbrake",
            "dodge_active",
            "jump_active",
            "double_jump_active",
            "flipped",
            "supersonic",
            "distance_to_ball",
            "angle_to_ball",
            "rotation_role",
            "distance_to_own_net",
            "angle_to_own_net",
            "distance_to_opp_net",
            "angle_to_opp_net",
        ]:
            columns.append(f"{slot}_{field}")
        for target in PLAYER_SLOTS:
            if target != slot:
                columns.append(f"{slot}_distance_to_{target}")

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

    frame_stems = {
        path.stem.removesuffix("_frames")
        for path in files
        if path.stem.endswith("_frames")
    }
    return [
        path
        for path in files
        if not (
            path.stem.endswith("_pbp")
            and path.stem.removesuffix("_pbp") in frame_stems
        )
    ]


def _split_path_inputs(
    value: str | PathLike[str] | Sequence[str | PathLike[str]],
) -> tuple[list[Path], list[Path]]:
    # Split replay inputs from already-parsed PBP/frame inputs.
    replay_inputs: list[Path] = []
    tabular_inputs: list[Path] = []

    for item in _path_items(value):
        path = _resolve_input_path(item)

        if path.is_dir():
            # Replay folders are parsed first even if prior exports exist nearby.
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

        # Read only the CSV header so large frame exports stay lazy.
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
    if paths and all(path.suffix.lower() == ".parquet" for path in paths):
        schema = tuple(_columns_for_file(paths[0]) or [])
        if schema:
            has_xg = "xG" in schema
            columns = requested_columns(has_xg, extra_columns=extra_columns)
            existing = set(schema)
            present_columns = [column for column in columns if column in existing]
            return (
                pl.scan_parquet(list(paths))
                .select(present_columns)
                .with_columns(
                    [
                        pl.lit(None).alias(column)
                        for column in columns
                        if column not in existing
                    ]
                )
                .select(columns)
            )

    # Group matching schemas so Polars can scan each batch efficiently.
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
    gpu: str | None,
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
        output="frames",
        export_format="parquet",
        force=force,
        limit=limit,
        gpu=gpu,
    )

    return [Path(path) for path in exported]


def _native_stats_export(
    replay_inputs: Sequence[Path],
    export: str | PathLike[str],
    *,
    workers: int,
    force: bool,
    limit: int | None,
    gpu: str | None,
) -> Path:
    from .parse import _boxcars_binary, _gpu_mode, _run_boxcars

    export_path = _resolve_input_path(export)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        _boxcars_binary(),
        "stats",
        "--workers",
        str(max(int(workers or 1), 1)),
        "--out-stats",
        str(export_path),
        "--format",
        "csv",
    ]
    gpu_mode = _gpu_mode(gpu)
    if gpu_mode is not None:
        command.extend(["--gpu", gpu_mode])
    for replay_input in replay_inputs:
        command.extend(["--replays", str(replay_input)])
    if limit is not None:
        command.extend(["--limit", str(int(limit))])
    if force:
        command.append("--force")

    _run_boxcars(command)
    return export_path


def _lazy_frames_from_paths(
    value: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    workers: int,
    parse_export: str | PathLike[str],
    force: bool,
    limit: int | None,
    gpu: str | None,
    extra_columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    replay_inputs, tabular_inputs = _split_path_inputs(value)

    # Replay inputs become PBP parquet before entering the stats pipeline.
    if replay_inputs:
        tabular_inputs.extend(
            _parse_replay_inputs(
                replay_inputs,
                workers=workers,
                parse_export=parse_export,
                force=force,
                limit=limit,
                gpu=gpu,
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


def shot_event_expr() -> pl.Expr:
    event_type = string_col("event_type")
    return event_type.is_in(["shot", "goal"])


def shot_attempt_expr() -> pl.Expr:
    event_type = string_col("event_type")
    return event_type.is_in(["shot", "goal", "missed-shot"])


def goal_event_expr() -> pl.Expr:
    return string_col("event_type") == "goal"


def touch_event_expr() -> pl.Expr:
    non_goal_touch_types = [value for value in TOUCH_EVENT_TYPES if value != "goal"]
    return string_col("event_type").is_in(non_goal_touch_types) | goal_event_expr()


def boost_pad_expr() -> pl.Expr:
    """Return the rows representing actual small or big boost-pad pickups."""
    return (string_col("event_type") == "boost-pickup") & string_col(
        "boost_pickup_type"
    ).is_in(["small", "big"])


def scaled_boost_pickup_amount_expr() -> pl.Expr:
    """Return boost-pad value in the standard 0-100 boost scale."""
    pickup_type = string_col("boost_pickup_type")
    return (
        pl.when(pickup_type == "small")
        .then(12.0)
        .when(pickup_type == "big")
        .then(100.0)
        .otherwise(0.0)
    )


def event_player_position_y_expr() -> pl.Expr:
    """Find the primary event player's y position from the matching frame slot."""
    player_id = string_col("event_player_1_id")
    return pl.coalesce(
        [
            pl.when(player_id == string_col(f"{slot}_id"))
            .then(pl.col(f"{slot}_pos_y").cast(pl.Float64, strict=False))
            .otherwise(None)
            for slot in PLAYER_SLOTS
        ]
    )


def event_player_position_x_expr() -> pl.Expr:
    """Find the primary event player's x position from the matching frame slot."""
    player_id = string_col("event_player_1_id")
    return pl.coalesce(
        [
            pl.when(player_id == string_col(f"{slot}_id"))
            .then(pl.col(f"{slot}_pos_x").cast(pl.Float64, strict=False))
            .otherwise(None)
            for slot in PLAYER_SLOTS
        ]
    )


def event_player_position_z_expr() -> pl.Expr:
    """Find the primary event player's z position from the matching frame slot."""
    player_id = string_col("event_player_1_id")
    return pl.coalesce(
        [
            pl.when(player_id == string_col(f"{slot}_id"))
            .then(pl.col(f"{slot}_pos_z").cast(pl.Float64, strict=False))
            .otherwise(None)
            for slot in PLAYER_SLOTS
        ]
    )


def event_player_boost_expr() -> pl.Expr:
    """Find the primary event player's boost amount from the matching frame slot."""
    player_id = string_col("event_player_1_id")
    return pl.coalesce(
        [
            pl.when(player_id == string_col(f"{slot}_id"))
            .then(pl.col(f"{slot}_boost").cast(pl.Float64, strict=False))
            .otherwise(None)
            for slot in PLAYER_SLOTS
        ]
    ).fill_null(0.0)


def boost_zone_exprs() -> tuple[pl.Expr, pl.Expr]:
    """Return stolen and protected pad conditions from team-relative thirds."""
    team = string_col("event_team")
    position_y = event_player_position_y_expr()
    stolen = ((team == "blue") & (position_y > FIELD_THIRD_Y)) | (
        (team == "orange") & (position_y < -FIELD_THIRD_Y)
    )
    protected = ((team == "blue") & (position_y < -FIELD_THIRD_Y)) | (
        (team == "orange") & (position_y > FIELD_THIRD_Y)
    )
    return stolen, protected


def event_player_slot_condition(slot: str) -> pl.Expr:
    return string_col("event_player_1_id") == string_col(f"{slot}_id")


def event_player_role_condition(role_index: int) -> pl.Expr:
    conditions = []
    for slot in PLAYER_SLOTS:
        team_slots = [
            other for other in PLAYER_SLOTS if other != slot and other[:4] == slot[:4]
        ]
        closer_teammates = sum(
            pl.when(
                (string_col(f"{other_slot}_id") != "")
                & (
                    number_col(f"{other_slot}_distance_to_ball")
                    < number_col(f"{slot}_distance_to_ball")
                )
            )
            .then(1)
            .otherwise(0)
            for other_slot in team_slots
        )
        conditions.append(
            event_player_slot_condition(slot) & ((closer_teammates + 1) == role_index)
        )

    return pl.any_horizontal(conditions)


def event_location_breakdowns() -> list[tuple[str, pl.Expr]]:
    ball_z = number_col("ball_pos_z")
    team = string_col("event_team")
    player_x = event_player_position_x_expr()
    player_y = event_player_position_y_expr()
    player_z = event_player_position_z_expr()
    near_side_wall = (SIDE_WALL_X - player_x.abs()).abs() <= 900.0
    near_back_wall = (BACK_WALL_Y - player_y.abs()).abs() <= 900.0
    on_or_off_wall = (player_z > GROUND_HEIGHT) & (near_side_wall | near_back_wall)
    side_wall = on_or_off_wall & near_side_wall
    offensive_back_wall = on_or_off_wall & (
        ((team == "blue") & ((player_y - BACK_WALL_Y).abs() <= 900.0))
        | ((team == "orange") & ((player_y + BACK_WALL_Y).abs() <= 900.0))
    )
    defensive_back_wall = on_or_off_wall & (
        ((team == "blue") & ((player_y + BACK_WALL_Y).abs() <= 900.0))
        | ((team == "orange") & ((player_y - BACK_WALL_Y).abs() <= 900.0))
    )

    return [
        ("first_man", event_player_role_condition(1)),
        ("second_man", event_player_role_condition(2)),
        ("third_man", event_player_role_condition(3)),
        ("fourth_man", event_player_role_condition(4)),
        ("offensive_back_wall", offensive_back_wall),
        ("defensive_back_wall", defensive_back_wall),
        ("side_wall", side_wall),
        ("ground", ball_z <= GROUND_HEIGHT),
        ("air", (ball_z > GROUND_HEIGHT) & (ball_z < CEILING_Z)),
        ("ceiling", (CEILING_Z - ball_z) <= 350.0),
        ("double_tap", flag_col("double_tap")),
    ]


def event_player_team_slots() -> list[tuple[str, list[str]]]:
    return [
        (
            slot,
            [
                other
                for other in PLAYER_SLOTS
                if other != slot and other[:4] == slot[:4]
            ],
        )
        for slot in PLAYER_SLOTS
    ]


def inferred_double_commit_expr() -> pl.Expr:
    same_play_commit = pl.any_horizontal(
        [
            event_player_slot_condition(slot)
            & pl.any_horizontal(
                [
                    (string_col(f"{teammate}_id") != "")
                    & (
                        number_col(f"{teammate}_distance_to_ball")
                        <= DOUBLE_COMMIT_BALL_DISTANCE
                    )
                    & (
                        number_col(f"{slot}_distance_to_{teammate}")
                        <= DOUBLE_COMMIT_TEAMMATE_DISTANCE
                    )
                    for teammate in teammates
                ]
            )
            for slot, teammates in event_player_team_slots()
            if teammates
        ]
    )

    return (string_col("event_type") == "double-commit") | (
        touch_event_expr() & same_play_commit
    )


def teammate_bump_expr() -> pl.Expr:
    return (
        (string_col("event_type") == "bump")
        & (string_col("event_player_1_team") != "")
        & (string_col("event_player_1_team") == string_col("event_player_2_team"))
    )


def opponent_bump_expr() -> pl.Expr:
    return (string_col("event_type") == "bump") & ~teammate_bump_expr()


def count_if(condition, name):
    return condition.cast(pl.Int64).sum().alias(name)


def sum_if(condition, value, name):
    return pl.when(condition).then(value).otherwise(0.0).sum().alias(name)


def avg_if(condition, value, name):
    return (
        pl.when(condition)
        .then(value)
        .otherwise(None)
        .mean()
        .fill_null(0.0)
        .alias(name)
    )


def max_if(condition, value, name):
    return (
        pl.when(condition)
        .then(value)
        .otherwise(None)
        .max()
        .fill_null(0.0)
        .alias(name)
    )


def official_stat_count(stat_type, fallback_condition, name, condition=None):
    """Prefer replay-recorded stat totals while supporting older parsed inputs."""
    official = flag_col(f"official_{stat_type}")
    recorded_count = number_col(f"official_{stat_type}_count")
    scope = pl.lit(True) if condition is None else condition
    scoped_official = official & scope
    recorded_total = (
        pl.when(scoped_official)
        .then(pl.when(recorded_count > 0).then(recorded_count).otherwise(1.0))
        .otherwise(0.0)
        .sum()
    )
    fallback_total = (fallback_condition & scope).cast(pl.Int64).sum()

    return (
        pl.when(official.any())
        .then(recorded_total)
        .otherwise(fallback_total)
        .cast(pl.Int64)
        .alias(name)
    )


def time_in_game_seconds_expr() -> pl.Expr:
    time_value = number_col("time_in_game")
    max_elapsed = number_col("_max_seconds_elapsed")

    return (
        pl.when(time_value > 0)
        .then(pl.when(time_value > 60.0).then(time_value).otherwise(time_value * 60.0))
        .otherwise(max_elapsed)
    )


def add_frame_delta_seconds(rows: pl.LazyFrame) -> pl.LazyFrame:
    frame_row = string_col("event_type") == ""
    event_row = (string_col("event_type") != "") & (string_col("event_player_1_id") != "")
    seconds = number_col("seconds_elapsed")
    previous_seconds = seconds.shift(1).over("replay_id")
    timestamp_delta = (seconds - previous_seconds).fill_null(0.0).clip(0.0, 1.0)
    new_timestamp = previous_seconds.is_null() | (seconds != previous_seconds)

    return rows.with_columns(
        [
            pl.when(frame_row & new_timestamp)
            .then(timestamp_delta)
            .otherwise(0.0)
            .alias("_stats_frame_delta_seconds"),
            pl.when(event_row)
            .then(
                pl.when(number_col("event_duration") > 0)
                .then(number_col("event_duration"))
                .otherwise(1.0 / 30.0)
            )
            .otherwise(0.0)
            .alias("_stats_event_frame_delta_seconds"),
        ]
    )


def player_slot_rows(
    events,
    slot,
    *,
    include_inactive: bool = False,
    player_slots: Sequence[str] = PLAYER_SLOTS,
):
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
                flag_col("official_shot").alias("official_shot"),
                flag_col("official_goal").alias("official_goal"),
                flag_col("official_save").alias("official_save"),
                number_col("official_shot_count").alias("official_shot_count"),
                number_col("official_goal_count").alias("official_goal_count"),
                number_col("official_save_count").alias("official_save_count"),
                flag_col("controlled").alias("controlled"),
                number_col("xG").alias("xG"),
                number_col("team_size").alias("team_size"),
                string_col(team_name_col).alias("team_name"),
                string_col(f"{slot}_platform").alias("platform"),
                string_col(f"{slot}_rank").alias("rank"),
                string_col(f"{slot}_rank_tier").alias("rank_tier"),
                flag_col(f"{slot}_pro_player").alias("pro_player"),
                number_col(f"{slot}_mmr").alias("mmr"),
                string_col(f"{slot}_car_id").alias("car_id"),
                string_col(f"{slot}_car_name").alias("car_name"),
                string_col(f"{slot}_decal_id").alias("decal_id"),
                string_col(f"{slot}_wheels_id").alias("wheels_id"),
                string_col(f"{slot}_boost_id").alias("boost_id"),
                string_col(f"{slot}_antenna_id").alias("antenna_id"),
                string_col(f"{slot}_topper_id").alias("topper_id"),
                string_col(f"{slot}_engine_audio_id").alias("engine_audio_id"),
                string_col(f"{slot}_trail_id").alias("trail_id"),
                string_col(f"{slot}_goal_explosion_id").alias("goal_explosion_id"),
                string_col(f"{slot}_primary_paint_finish_id").alias(
                    "primary_paint_finish_id"
                ),
                string_col(f"{slot}_accent_paint_finish_id").alias(
                    "accent_paint_finish_id"
                ),
                number_col(f"{slot}_camera_fov").alias("camera_fov"),
                number_col(f"{slot}_camera_height").alias("camera_height"),
                number_col(f"{slot}_camera_angle").alias("camera_angle"),
                number_col(f"{slot}_camera_distance").alias("camera_distance"),
                number_col(f"{slot}_camera_stiffness").alias("camera_stiffness"),
                number_col(f"{slot}_camera_swivel").alias("camera_swivel"),
                number_col(f"{slot}_camera_transition").alias("camera_transition"),
                flag_col(f"{slot}_is_bot").alias("is_bot"),
                number_col(f"{slot}_score").alias("score"),
                number_col(f"{slot}_time_in_game").alias("time_in_game"),
                number_col("frame_number").alias("frame_number"),
                pl.col(f"{slot}_pos_x")
                .cast(pl.Float64, strict=False)
                .is_not_null()
                .alias("_has_position"),
                number_col("seconds_elapsed").alias("seconds_elapsed"),
                number_col("event_length").alias("event_length"),
                number_col("event_duration").alias("event_duration"),
                number_col("_stats_frame_delta_seconds").alias("frame_delta_seconds"),
                number_col(f"{slot}_boost").alias("boost"),
                flag_col(f"{slot}_boost_active").alias("boost_active"),
                number_col(f"{slot}_throttle").alias("throttle"),
                number_col(f"{slot}_steer").alias("steer"),
                flag_col(f"{slot}_handbrake").alias("handbrake"),
                flag_col(f"{slot}_dodge_active").alias("dodge_active"),
                flag_col(f"{slot}_jump_active").alias("jump_active"),
                flag_col(f"{slot}_double_jump_active").alias("double_jump_active"),
                flag_col(f"{slot}_flipped").alias("flipped"),
                flag_col(f"{slot}_supersonic").alias("supersonic"),
                number_col(f"{slot}_pos_x").alias("pos_x"),
                number_col(f"{slot}_pos_y").alias("pos_y"),
                number_col(f"{slot}_pos_z").alias("pos_z"),
                number_col(f"{slot}_vel_x").alias("vel_x"),
                number_col(f"{slot}_vel_y").alias("vel_y"),
                number_col(f"{slot}_vel_z").alias("vel_z"),
                number_col(f"{slot}_distance_to_ball").alias("distance_to_ball"),
                number_col(f"{slot}_angle_to_ball").alias("angle_to_ball"),
                number_col(f"{slot}_rotation_role").alias("rotation_role"),
                number_col(f"{slot}_distance_to_own_net").alias("distance_to_own_net"),
                number_col(f"{slot}_angle_to_own_net").alias("angle_to_own_net"),
                number_col(f"{slot}_distance_to_opp_net").alias("distance_to_opp_net"),
                number_col(f"{slot}_angle_to_opp_net").alias("angle_to_opp_net"),
                number_col("ball_pos_x").alias("ball_pos_x"),
                number_col("ball_pos_y").alias("ball_pos_y"),
                number_col("ball_pos_z").alias("ball_pos_z"),
                number_col("ball_vel_x").alias("ball_vel_x"),
                number_col("ball_vel_y").alias("ball_vel_y"),
                number_col("ball_vel_z").alias("ball_vel_z"),
                pl.min_horizontal(
                    [
                        pl.col(f"{other_slot}_distance_to_ball")
                        .cast(pl.Float64, strict=False)
                        for other_slot in player_slots
                    ]
                ).alias("nearest_distance_to_ball"),
                pl.max_horizontal(
                    [
                        pl.col(f"{other_slot}_distance_to_ball")
                        .cast(pl.Float64, strict=False)
                        for other_slot in player_slots
                    ]
                ).alias("furthest_distance_to_ball"),
                (string_col("event_type") != "").alias("_is_event_row"),
            ]
            + [
                string_col(f"{other_slot}_id").alias(f"{other_slot}_id")
                for other_slot in player_slots
            ]
            + [
                (
                    pl.lit(0.0)
                    if other_slot == slot
                    else number_col(f"{slot}_distance_to_{other_slot}")
                ).alias(f"distance_to_{other_slot}")
                for other_slot in player_slots
            ]
            + [
                number_col(f"{other_slot}_distance_to_ball").alias(
                    f"{other_slot}_distance_to_ball"
                )
                for other_slot in player_slots
            ]
            + [
                number_col(f"{other_slot}_pos_x").alias(f"{other_slot}_pos_x")
                for other_slot in player_slots
            ]
            + [
                number_col(f"{other_slot}_pos_y").alias(f"{other_slot}_pos_y")
                for other_slot in player_slots
            ]
            + [
                (
                    pl.lit(0.0)
                    if other_slot == slot
                    else pl.arctan2(
                        number_col(f"{other_slot}_pos_y") - number_col(f"{slot}_pos_y"),
                        number_col(f"{other_slot}_pos_x") - number_col(f"{slot}_pos_x"),
                    )
                ).alias(f"angle_to_{other_slot}")
                for other_slot in player_slots
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


def player_slot_frame(events, slot, player_slots: Sequence[str] = PLAYER_SLOTS):
    rows = player_slot_rows(
        events,
        slot,
        include_inactive=True,
        player_slots=player_slots,
    )
    active_rows = (
        rows.filter(pl.col("_active_on_row"))
        .with_columns(
            [
                (
                    pl.col("vel_x").pow(2)
                    + pl.col("vel_y").pow(2)
                    + pl.col("vel_z").pow(2)
                )
                .sqrt()
                .alias("_speed"),
                (
                    (pl.col("pos_x") - pl.col("pos_x").shift(1)).pow(2)
                    + (pl.col("pos_y") - pl.col("pos_y").shift(1)).pow(2)
                    + (pl.col("pos_z") - pl.col("pos_z").shift(1)).pow(2)
                )
                .sqrt()
                .fill_null(0.0)
                .alias("_distance_traveled"),
                (
                    pl.col("handbrake")
                    & ~pl.col("handbrake").shift(1).fill_null(False)
                ).alias("_powerslide_pressed"),
                (pl.col("boost").shift(1) - pl.col("boost"))
                .clip(lower_bound=0.0)
                .fill_null(0.0)
                .alias("_boost_used"),
                (pl.col("boost") - pl.col("boost").shift(1))
                .clip(lower_bound=0.0)
                .fill_null(0.0)
                .alias("_boost_gained"),
                (
                    (pl.col("ball_vel_x") - pl.col("ball_vel_x").shift(1)).pow(2)
                    + (pl.col("ball_vel_y") - pl.col("ball_vel_y").shift(1)).pow(2)
                    + (pl.col("ball_vel_z") - pl.col("ball_vel_z").shift(1)).pow(2)
                )
                .sqrt()
                .fill_null(0.0)
                .alias("_ball_velocity_delta"),
                (
                    (pl.col("boost_active") != pl.col("boost_active").shift(1))
                    | (pl.col("handbrake") != pl.col("handbrake").shift(1))
                    | (pl.col("dodge_active") != pl.col("dodge_active").shift(1))
                    | (pl.col("jump_active") != pl.col("jump_active").shift(1))
                    | (pl.col("double_jump_active") != pl.col("double_jump_active").shift(1))
                    | (pl.col("flipped") != pl.col("flipped").shift(1))
                    | ((pl.col("throttle") - pl.col("throttle").shift(1)).abs() >= 0.25)
                    | ((pl.col("steer") - pl.col("steer").shift(1)).abs() >= 0.25)
                )
                .fill_null(False)
                .alias("_input_changed"),
            ]
        )
    )
    boost = pl.col("boost")
    position_y = pl.col("pos_y")
    position_z = pl.col("pos_z")
    frame_seconds = pl.col("frame_delta_seconds")
    team = "orange" if slot.startswith("orange") else "blue"
    same_team_slots = [
        other for other in player_slots if other != slot and other.startswith(team)
    ]
    offensive_third = (
        position_y < -FIELD_THIRD_Y
        if team == "orange"
        else position_y > FIELD_THIRD_Y
    )
    defensive_third = (
        position_y > FIELD_THIRD_Y
        if team == "orange"
        else position_y < -FIELD_THIRD_Y
    )
    neutral_third = position_y.abs() <= FIELD_THIRD_Y
    possession = (
        pl.col("nearest_distance_to_ball").is_not_null()
        & (
            (pl.col("distance_to_ball") - pl.col("nearest_distance_to_ball")).abs()
            <= 0.001
        )
    )
    furthest_from_ball = (
        pl.col("furthest_distance_to_ball").is_not_null()
        & (
            (pl.col("distance_to_ball") - pl.col("furthest_distance_to_ball")).abs()
            <= 0.001
        )
    )
    behind_ball = (
        position_y < pl.col("ball_pos_y")
        if team == "blue"
        else position_y > pl.col("ball_pos_y")
    )
    first_man_index = (
        pl.when(pl.col("rotation_role") > 0)
        .then(pl.col("rotation_role"))
        .otherwise(
            sum(
                pl.when(
                    (string_col(f"{other_slot}_id") != "")
                    & (
                        number_col(f"{other_slot}_distance_to_ball")
                        < pl.col("distance_to_ball")
                    )
                )
                .then(1)
                .otherwise(0)
                for other_slot in same_team_slots
            )
            + 1
        )
    )
    side_wall = (SIDE_WALL_X - pl.col("pos_x").abs()).abs() <= 350.0
    offensive_back_wall = (
        (position_y + 5120.0).abs() <= 350.0
        if team == "orange"
        else (position_y - 5120.0).abs() <= 350.0
    )
    defensive_back_wall = (
        (position_y - 5120.0).abs() <= 350.0
        if team == "orange"
        else (position_y + 5120.0).abs() <= 350.0
    )
    ceiling = (2044.0 - position_z) <= 350.0
    active_rows = (
        active_rows.with_columns(
            first_man_index.cast(pl.Int64, strict=False).alias("_rotation_role")
        )
        .with_columns(
            [
                pl.col("_rotation_role")
                .shift(1)
                .over(["replay_id", "player_id"])
                .alias("_previous_rotation_role"),
                (
                    pl.col("_rotation_role")
                    != pl.col("_rotation_role").shift(1).over(["replay_id", "player_id"])
                )
                .fill_null(False)
                .alias("_rotation_role_changed"),
            ]
        )
        .with_columns(
            pl.col("_rotation_role_changed")
            .cast(pl.Int64)
            .cum_sum()
            .over(["replay_id", "player_id"])
            .alias("_rotation_run_id")
        )
        .with_columns(
            [
                pl.col("frame_delta_seconds")
                .sum()
                .over(["replay_id", "player_id", "_rotation_run_id"])
                .alias("_rotation_run_seconds"),
                pl.col("_boost_gained")
                .sum()
                .over(["replay_id", "player_id", "_rotation_run_id"])
                .alias("_rotation_run_boost_gained"),
                pl.when((pl.col("_boost_gained") > 0.0) & (pl.col("_boost_gained") < 50.0))
                .then(1)
                .otherwise(0)
                .sum()
                .over(["replay_id", "player_id", "_rotation_run_id"])
                .alias("_rotation_run_small_boost"),
                pl.when(pl.col("_boost_gained") >= 50.0)
                .then(1)
                .otherwise(0)
                .sum()
                .over(["replay_id", "player_id", "_rotation_run_id"])
                .alias("_rotation_run_big_boost"),
                pl.col("frame_delta_seconds")
                .cum_sum()
                .over(["replay_id", "player_id", "_rotation_run_id"])
                .alias("_rotation_run_elapsed_seconds"),
                (
                    (pl.col("_ball_velocity_delta") >= 500.0)
                    & (
                        pl.col("_ball_velocity_delta")
                        .shift(1)
                        .over(["replay_id", "player_id"])
                        .fill_null(0.0)
                        < 500.0
                    )
                ).alias("_ball_movement_started"),
            ]
        )
        .with_columns(
            [
                pl.col("_ball_movement_started")
                .cast(pl.Int64)
                .cum_sum()
                .over(["replay_id", "player_id"])
                .alias("_ball_movement_id"),
                pl.when(pl.col("_ball_movement_started"))
                .then(pl.col("frame_number") / 30.0)
                .otherwise(None)
                .forward_fill()
                .over(["replay_id", "player_id"])
                .alias("_last_ball_movement_seconds"),
                pl.col("_rotation_run_seconds")
                .shift(1)
                .over(["replay_id", "player_id"])
                .fill_null(0.0)
                .alias("_previous_rotation_run_seconds"),
                pl.col("_rotation_run_boost_gained")
                .shift(1)
                .over(["replay_id", "player_id"])
                .fill_null(0.0)
                .alias("_previous_rotation_run_boost_gained"),
                pl.col("_rotation_run_small_boost")
                .shift(1)
                .over(["replay_id", "player_id"])
                .fill_null(0)
                .alias("_previous_rotation_run_small_boost"),
                pl.col("_rotation_run_big_boost")
                .shift(1)
                .over(["replay_id", "player_id"])
                .fill_null(0)
                .alias("_previous_rotation_run_big_boost"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("_input_changed")
                    & pl.col("_last_ball_movement_seconds").is_not_null()
                    & (
                        ((pl.col("frame_number") / 30.0) - pl.col("_last_ball_movement_seconds"))
                        .is_between(0.05, 1.5, closed="both")
                    )
                ).alias("_reaction_observed"),
                (
                    (pl.col("frame_number") / 30.0) - pl.col("_last_ball_movement_seconds")
                ).alias("_reaction_time_seconds"),
            ]
        )
        .with_columns(
            (
                pl.col("_reaction_observed")
                & (
                    pl.col("_reaction_observed")
                    .cast(pl.Int64)
                    .cum_sum()
                    .over(["replay_id", "player_id", "_ball_movement_id"])
                    == 1
                )
            ).alias("_reaction_observed")
        )
    )
    rotation_role = pl.col("_rotation_role")
    previous_rotation_role = pl.col("_previous_rotation_role")
    team_size = number_col("team_size").cast(pl.Int64, strict=False)
    valid_rotation_transition = (
        (team_size > 1)
        & pl.col("_rotation_role_changed")
        & (pl.col("_previous_rotation_run_seconds") >= 0.5)
        & rotation_role.is_not_null()
        & previous_rotation_role.is_not_null()
        & (rotation_role != previous_rotation_role)
    )
    rotation_filled = valid_rotation_transition & (
        (
            (previous_rotation_role > 1)
            & (rotation_role == previous_rotation_role - 1)
        )
        | ((previous_rotation_role == 1) & (rotation_role == team_size))
    )
    rotation_cut = valid_rotation_transition & (
        ((previous_rotation_role - rotation_role) > 1)
        | (
            (previous_rotation_role == 1)
            & (rotation_role > 1)
            & (rotation_role != team_size)
        )
    )
    rotation_stalled = (
        (team_size > 1)
        & (rotation_role == 1)
        & (pl.col("_rotation_run_elapsed_seconds") >= 1.5)
        & ((pl.col("_rotation_run_elapsed_seconds") - frame_seconds) < 1.5)
    )
    rotation_task = rotation_filled | rotation_cut | rotation_stalled

    def frame_time_seconds(condition: pl.Expr, name: str) -> pl.Expr:
        return (
            pl.when(condition)
            .then(frame_seconds)
            .otherwise(0.0)
            .sum()
            .alias(name)
        )

    frame_time_stats = [
        frame_time_seconds(boost <= 0.0, "time_zero_boost"),
        frame_time_seconds(boost >= 100.0, "time_full_boost"),
        frame_time_seconds(
            (boost > 0.0) & (boost < 25.0),
            "time_zero_to_quarter_boost",
        ),
        frame_time_seconds(
            (boost >= 25.0) & (boost < 50.0),
            "time_quarter_to_half_boost",
        ),
        frame_time_seconds(
            (boost >= 50.0) & (boost < 75.0),
            "time_half_to_three_quarters_boost",
        ),
        frame_time_seconds(
            (boost >= 75.0) & (boost < 100.0),
            "time_three_quarters_to_full_boost",
        ),
        frame_time_seconds(offensive_third, "time_offensive_third"),
        frame_time_seconds(neutral_third, "time_neutral_third"),
        frame_time_seconds(defensive_third, "time_defensive_third"),
        frame_time_seconds(
            (position_z > GROUND_HEIGHT) & (position_z < CROSSBAR_HEIGHT),
            "time_low_air",
        ),
        frame_time_seconds(position_z >= CROSSBAR_HEIGHT, "time_high_air"),
        frame_time_seconds(position_z <= GROUND_HEIGHT, "time_ground"),
        frame_time_seconds(possession, "time_possession"),
        frame_time_seconds(pl.col("handbrake"), "time_holding_powerslide"),
        frame_time_seconds(pl.col("boost_active"), "time_holding_boost"),
        frame_time_seconds(
            pl.col("boost_active") & pl.col("supersonic"),
            "time_wasting_boost",
        ),
        frame_time_seconds(behind_ball, "time_behind_ball"),
        frame_time_seconds(~behind_ball, "time_in_front_of_ball"),
        frame_time_seconds(possession, "time_closest_to_ball"),
        frame_time_seconds(furthest_from_ball, "time_furthest_from_ball"),
        frame_time_seconds(first_man_index == 1, "time_first_man"),
        frame_time_seconds(first_man_index == 2, "time_second_man"),
        frame_time_seconds(first_man_index == 3, "time_third_man"),
        frame_time_seconds(first_man_index == 4, "time_fourth_man"),
        frame_time_seconds(side_wall, "time_side_wall"),
        frame_time_seconds(offensive_back_wall, "time_offensive_back_wall"),
        frame_time_seconds(defensive_back_wall, "time_defensive_back_wall"),
        frame_time_seconds(ceiling, "time_ceiling"),
    ]
    total_frame_seconds = pl.col("frame_delta_seconds").sum()
    teammate_distance_exprs = [
        pl.when(string_col(f"{other_slot}_id") != "")
        .then(pl.col(f"distance_to_{other_slot}"))
        .otherwise(None)
        for other_slot in same_team_slots
    ]
    teammate_angles = [
        pl.when(string_col(f"{other_slot}_id") != "")
        .then(
            pl.arctan2(
                pl.col(f"{other_slot}_pos_y") - pl.col("pos_y"),
                pl.col(f"{other_slot}_pos_x") - pl.col("pos_x"),
            )
        )
        .otherwise(None)
        for other_slot in same_team_slots
    ]
    closest_teammate_distance = pl.min_horizontal(
        [
            pl.when(string_col(f"{other_slot}_id") != "")
            .then(pl.col(f"{other_slot}_distance_to_ball"))
            .otherwise(None)
            for other_slot in same_team_slots
        ]
    )
    furthest_teammate_distance = pl.max_horizontal(
        [
            pl.when(string_col(f"{other_slot}_id") != "")
            .then(pl.col(f"{other_slot}_distance_to_ball"))
            .otherwise(None)
            for other_slot in same_team_slots
        ]
    )

    def time_weighted_average(value: pl.Expr, name: str) -> pl.Expr:
        return (
            pl.when(total_frame_seconds > 0)
            .then((value * pl.col("frame_delta_seconds")).sum() / total_frame_seconds)
            .otherwise(0.0)
            .alias(name)
        )

    def teammate_value_for_rank(target_distance: pl.Expr, field: str) -> pl.Expr:
        return pl.coalesce(
            [
                pl.when(
                    (string_col(f"{other_slot}_id") != "")
                    & (
                        (
                            pl.col(f"{other_slot}_distance_to_ball")
                            - target_distance
                        ).abs()
                        <= 0.001
                    )
                )
                .then(pl.col(field.format(slot=other_slot)))
                .otherwise(None)
                for other_slot in same_team_slots
            ]
        )

    frame_value_stats = [
        pl.col("_distance_traveled").sum().alias("distance_traveled"),
        pl.when(total_frame_seconds > 0)
        .then(
            (pl.col("_speed") * pl.col("frame_delta_seconds")).sum()
            / total_frame_seconds
        )
        .otherwise(0.0)
        .alias("avg_speed"),
        pl.col("_speed").max().fill_null(0.0).alias("max_speed"),
        pl.col("_powerslide_pressed").sum().alias("powerslide_presses"),
        pl.when(pl.col("boost_active") & pl.col("supersonic"))
        .then(pl.col("_boost_used"))
        .otherwise(0.0)
        .sum()
        .alias("boost_amount_wasted"),
        pl.when(rotation_task).then(1).otherwise(0).sum().alias("rotation_tasks"),
        pl.when(rotation_filled).then(1).otherwise(0).sum().alias("rotations_filled"),
        pl.when(rotation_cut).then(1).otherwise(0).sum().alias("rotations_cut"),
        pl.when(rotation_stalled).then(1).otherwise(0).sum().alias("rotations_stalled"),
        pl.when(rotation_filled)
        .then(pl.col("_previous_rotation_run_small_boost"))
        .otherwise(0)
        .sum()
        .alias("rotations_fill_small_boost"),
        pl.when(rotation_filled)
        .then(pl.col("_previous_rotation_run_big_boost"))
        .otherwise(0)
        .sum()
        .alias("rotations_fill_big_boost"),
        pl.when(rotation_filled)
        .then(pl.col("_previous_rotation_run_boost_gained"))
        .otherwise(0.0)
        .sum()
        .alias("rotations_fill_boost_collected"),
        pl.when(pl.when(rotation_filled).then(1).otherwise(0).sum() > 0)
        .then(
            pl.when(rotation_filled)
            .then(pl.col("_previous_rotation_run_small_boost"))
            .otherwise(0)
            .sum()
            / pl.when(rotation_filled).then(1).otherwise(0).sum()
        )
        .otherwise(0.0)
        .alias("avg_rotations_fill_small_boost"),
        pl.when(pl.when(rotation_filled).then(1).otherwise(0).sum() > 0)
        .then(
            pl.when(rotation_filled)
            .then(pl.col("_previous_rotation_run_big_boost"))
            .otherwise(0)
            .sum()
            / pl.when(rotation_filled).then(1).otherwise(0).sum()
        )
        .otherwise(0.0)
        .alias("avg_rotations_fill_big_boost"),
        pl.when(rotation_filled)
        .then(pl.col("_previous_rotation_run_seconds"))
        .otherwise(0.0)
        .sum()
        .alias("time_rotations_fill"),
        pl.when(rotation_cut)
        .then(pl.col("_previous_rotation_run_seconds"))
        .otherwise(0.0)
        .sum()
        .alias("time_rotations_cut"),
        pl.when(rotation_stalled)
        .then(pl.col("_rotation_run_elapsed_seconds"))
        .otherwise(0.0)
        .sum()
        .alias("time_rotations_stalled"),
        pl.when(pl.when(rotation_filled).then(1).otherwise(0).sum() > 0)
        .then(
            pl.when(rotation_filled)
            .then(pl.col("_previous_rotation_run_seconds"))
            .otherwise(0.0)
            .sum()
            / pl.when(rotation_filled).then(1).otherwise(0).sum()
        )
        .otherwise(0.0)
        .alias("avg_time_rotations_fill"),
        pl.when(pl.when(rotation_cut).then(1).otherwise(0).sum() > 0)
        .then(
            pl.when(rotation_cut)
            .then(pl.col("_previous_rotation_run_seconds"))
            .otherwise(0.0)
            .sum()
            / pl.when(rotation_cut).then(1).otherwise(0).sum()
        )
        .otherwise(0.0)
        .alias("avg_time_rotations_cut"),
        pl.when(pl.when(rotation_stalled).then(1).otherwise(0).sum() > 0)
        .then(
            pl.when(rotation_stalled)
            .then(pl.col("_rotation_run_elapsed_seconds"))
            .otherwise(0.0)
            .sum()
            / pl.when(rotation_stalled).then(1).otherwise(0).sum()
        )
        .otherwise(0.0)
        .alias("avg_time_rotations_stalled"),
        pl.when(pl.col("_reaction_observed"))
        .then(pl.col("_reaction_time_seconds"))
        .otherwise(None)
        .mean()
        .fill_null(0.0)
        .alias("avg_reaction_time"),
        pl.when(pl.col("_reaction_observed"))
        .then(pl.col("_reaction_time_seconds"))
        .otherwise(0.0)
        .sum()
        .alias("reaction_time_total"),
        pl.when(pl.col("_reaction_observed"))
        .then(1)
        .otherwise(0)
        .sum()
        .alias("reaction_time_count"),
        time_weighted_average(
            pl.mean_horizontal(teammate_distance_exprs),
            "avg_distance_to_teammates",
        ),
        time_weighted_average(
            teammate_value_for_rank(closest_teammate_distance, "distance_to_{slot}"),
            "avg_distance_to_teammate_closest_to_ball",
        ),
        time_weighted_average(
            teammate_value_for_rank(furthest_teammate_distance, "distance_to_{slot}"),
            "avg_distance_to_teammate_further_from_ball",
        ),
        time_weighted_average(pl.col("distance_to_ball"), "avg_distance_from_ball"),
        time_weighted_average(
            pl.col("distance_to_opp_net"),
            "avg_distance_from_offensive_net",
        ),
        time_weighted_average(
            pl.col("distance_to_own_net"),
            "avg_distance_from_defensive_net",
        ),
        time_weighted_average(
            pl.mean_horizontal(teammate_angles),
            "avg_angle_to_teammates",
        ),
        time_weighted_average(
            teammate_value_for_rank(closest_teammate_distance, "angle_to_{slot}"),
            "avg_angle_to_teammate_closest_to_ball",
        ),
        time_weighted_average(
            teammate_value_for_rank(furthest_teammate_distance, "angle_to_{slot}"),
            "avg_angle_to_teammate_further_from_ball",
        ),
        time_weighted_average(pl.col("angle_to_ball"), "avg_angle_from_ball"),
        time_weighted_average(
            pl.col("angle_to_opp_net"),
            "avg_angle_from_offensive_net",
        ),
        time_weighted_average(
            pl.col("angle_to_own_net"),
            "avg_angle_from_defensive_net",
        ),
    ]

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
                pl.col("decal_id").drop_nulls().first(),
                pl.col("wheels_id").drop_nulls().first(),
                pl.col("boost_id").drop_nulls().first(),
                pl.col("antenna_id").drop_nulls().first(),
                pl.col("topper_id").drop_nulls().first(),
                pl.col("engine_audio_id").drop_nulls().first(),
                pl.col("trail_id").drop_nulls().first(),
                pl.col("goal_explosion_id").drop_nulls().first(),
                pl.col("primary_paint_finish_id").drop_nulls().first(),
                pl.col("accent_paint_finish_id").drop_nulls().first(),
                pl.col("camera_fov").drop_nulls().first(),
                pl.col("camera_height").drop_nulls().first(),
                pl.col("camera_angle").drop_nulls().first(),
                pl.col("camera_distance").drop_nulls().first(),
                pl.col("camera_stiffness").drop_nulls().first(),
                pl.col("camera_swivel").drop_nulls().first(),
                pl.col("camera_transition").drop_nulls().first(),
                pl.col("is_bot").max(),
                pl.col("score").max(),
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
            + frame_time_stats
            + frame_value_stats
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
            pl.lit(None).alias("decal_id"),
            pl.lit(None).alias("wheels_id"),
            pl.lit(None).alias("boost_id"),
            pl.lit(None).alias("antenna_id"),
            pl.lit(None).alias("topper_id"),
            pl.lit(None).alias("engine_audio_id"),
            pl.lit(None).alias("trail_id"),
            pl.lit(None).alias("goal_explosion_id"),
            pl.lit(None).alias("primary_paint_finish_id"),
            pl.lit(None).alias("accent_paint_finish_id"),
            pl.lit(None).alias("camera_fov"),
            pl.lit(None).alias("camera_height"),
            pl.lit(None).alias("camera_angle"),
            pl.lit(None).alias("camera_distance"),
            pl.lit(None).alias("camera_stiffness"),
            pl.lit(None).alias("camera_swivel"),
            pl.lit(None).alias("camera_transition"),
            pl.lit(False).alias("is_bot"),
            pl.lit(0.0).alias("score"),
            pl.lit(False).alias("_has_position"),
            pl.lit(False).alias("_has_static_car"),
            pl.lit(True).alias("_slot_has_presence_events"),
            pl.lit(0.0).alias("time_in_game"),
            pl.col("seconds_elapsed").max().alias("_max_seconds_elapsed"),
            pl.lit(0.0).alias("_event_length_seconds"),
            pl.lit(0.0).alias("_frame_time_seconds"),
            pl.len().alias("_row_count"),
            pl.len().alias("_event_row_count"),
        ]
        + [pl.lit(0.0).alias(column) for column in FRAME_TIME_COLUMNS]
        + [pl.lit(0.0).alias(column) for column in FRAME_VALUE_COLUMNS]
        + [
            pl.when(pl.col("_presence_state") == 1)
            .then(pl.col("_presence_interval_seconds"))
            .otherwise(0.0)
            .sum()
            .alias("_presence_time_seconds"),
        ]
    )


def player_frame(
    rows,
    *,
    player_slots: Sequence[str] = PLAYER_SLOTS,
    include_event_presence: bool = True,
):
    player_frames = [player_slot_frame(rows, slot, player_slots) for slot in player_slots]
    if include_event_presence:
        player_frames.append(event_presence_player_frame(rows))

    players = pl.concat(
        player_frames,
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
            pl.col("decal_id").drop_nulls().first(),
            pl.col("wheels_id").drop_nulls().first(),
            pl.col("boost_id").drop_nulls().first(),
            pl.col("antenna_id").drop_nulls().first(),
            pl.col("topper_id").drop_nulls().first(),
            pl.col("engine_audio_id").drop_nulls().first(),
            pl.col("trail_id").drop_nulls().first(),
            pl.col("goal_explosion_id").drop_nulls().first(),
            pl.col("primary_paint_finish_id").drop_nulls().first(),
            pl.col("accent_paint_finish_id").drop_nulls().first(),
            pl.col("camera_fov").drop_nulls().first(),
            pl.col("camera_height").drop_nulls().first(),
            pl.col("camera_angle").drop_nulls().first(),
            pl.col("camera_distance").drop_nulls().first(),
            pl.col("camera_stiffness").drop_nulls().first(),
            pl.col("camera_swivel").drop_nulls().first(),
            pl.col("camera_transition").drop_nulls().first(),
            pl.col("is_bot").max(),
            pl.col("score").sum(),
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
        + [pl.col(column).sum().alias(column) for column in FRAME_TIME_COLUMNS]
        + [
            pl.col(column).sum().alias(column)
            for column in FRAME_VALUE_COLUMNS
            if column not in {"avg_speed", "max_speed"}
        ]
        + [
            pl.col("avg_speed").max().alias("avg_speed"),
            pl.col("max_speed").max().alias("max_speed"),
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


def primary_event_stats(events, has_xg, has_frame_data):
    shot = shot_event_expr()
    shot_attempt = shot_attempt_expr()
    missed_shot = string_col("event_type") == "missed-shot"
    missed_pass = string_col("event_type") == "missed-pass"
    goal = goal_event_expr()
    touch = touch_event_expr()
    boost_pad = boost_pad_expr()
    save = string_col("event_type") == "save"
    pass_event = string_col("event_type") == "pass"
    flick = string_col("event_type") == "flick"
    air_dribble = string_col("event_type").is_in(["air-dribble", "air_dribble"])
    ground_dribble = string_col("event_type").is_in(["ground-dribble", "ground_dribble"])
    double_commit = inferred_double_commit_expr()
    small_boost = boost_pad & (string_col("boost_pickup_type") == "small")
    big_boost = boost_pad & (string_col("boost_pickup_type") == "big")
    boost_amount = scaled_boost_pickup_amount_expr()
    boost_overfill_amount = (event_player_boost_expr() + boost_amount - 100.0).clip(
        lower_bound=0.0
    )
    boost_overfill = boost_pad & (boost_overfill_amount > 0.0)
    boost_stolen, boost_protected = boost_zone_exprs()
    entry = string_col("event_type") == "entry"
    exit_event = string_col("event_type") == "exit"
    controlled = flag_col("controlled")
    opponent_bump = opponent_bump_expr()
    shot_breakdowns = [
        ("off_demo", flag_col("off_demo")),
        ("off_kickoff", flag_col("off_kickoff")),
        ("off_challenge_win", flag_col("off_challenge_win")),
        ("off_bump", flag_col("off_bump")),
        ("off_air_dribble", flag_col("off_air_dribble")),
        ("off_ground_dribble", flag_col("off_ground_dribble")),
        ("off_flick", flag_col("off_flick")),
        ("off_double_tap", flag_col("off_double_tap")),
        ("off_pass", flag_col("off_pass")),
        ("off_fake", flag_col("off_fake")),
        ("off_whiff", flag_col("off_whiff")),
        ("off_rotation_cut", flag_col("off_rotation_cut")),
        ("aerial", flag_col("aerialing")),
        ("air_dribble", flag_col("air_dribble")),
        ("ground_dribble", flag_col("ground_dribble")),
        ("flick", flag_col("flick_shot")),
        ("rebound", flag_col("rebound")),
        ("off_flip_reset", flag_col("off_flip_reset")),
        ("off_wall", flag_col("off_wall")),
        ("off_ceiling", flag_col("off_ceiling")),
    ]
    metric_breakdowns = event_location_breakdowns() if has_frame_data else []
    shot_speed = number_col("ball_speed_from_last_event")
    shot_angle = number_col("ball_angle_from_last_event")
    shot_distance = number_col("distance_to_goal")
    pass_speed = number_col("ball_speed_from_last_event")
    pass_angle = number_col("ball_angle_from_last_event")
    pass_distance = number_col("ball_distance_from_last_event")
    flick_distance = number_col("ball_distance_from_last_event")
    flick_speed = number_col("ball_speed_from_last_event")
    event_frame_delta_seconds = number_col("_stats_event_frame_delta_seconds")

    expressions = [
        official_stat_count("shot", shot, "_recorded_shots"),
        official_stat_count("goal", goal, "goals"),
        official_stat_count("save", save, "saves"),
        count_if(shot_attempt, "shot_attempts"),
        count_if(pl.col("event_type") == "missed-shot", "missed_shots"),
        count_if(missed_pass, "missed_passes"),
        count_if(touch, "touches"),
        count_if(pl.col("event_type") == "pass", "passes"),
        count_if(pl.col("event_type") == "turnover", "turnovers"),
        count_if(pl.col("event_type") == "challenge", "challenge_wins"),
        count_if(pl.col("event_type") == "kickoff", "kickoff_wins"),
        count_if(pl.col("event_type") == "shadow", "shadows"),
        count_if(pl.col("event_type") == "press", "presses"),
        count_if(pl.col("event_type") == "fake", "fakes"),
        count_if(pl.col("event_type") == "whiff", "whiffs"),
        count_if(pl.col("event_type") == "demo", "demos_applied"),
        count_if(opponent_bump, "bumps"),
        count_if(entry, "entries"),
        count_if(entry & controlled, "controlled_entries"),
        count_if(entry & ~controlled, "uncontrolled_entries"),
        count_if(exit_event, "exits"),
        count_if(exit_event & controlled, "controlled_exits"),
        count_if(exit_event & ~controlled, "uncontrolled_exits"),
        count_if(pl.col("event_type") == "retrieval", "retrievals"),
        count_if(pl.col("event_type").is_in(["air-dribble", "air_dribble"]), "air_dribbles"),
        count_if(pl.col("event_type").is_in(["ground-dribble", "ground_dribble"]), "ground_dribbles"),
        count_if(pl.col("event_type") == "flick", "flicks"),
        count_if(pl.col("event_type") == "flip-reset", "flip_resets"),
        count_if(double_commit, "double_commits"),
        sum_if(air_dribble, event_frame_delta_seconds, "time_air_dribble"),
        sum_if(ground_dribble, event_frame_delta_seconds, "time_ground_dribble"),
        avg_if(flick, flick_distance, "avg_flicks_distance"),
        avg_if(flick, flick_speed, "avg_flicks_speed"),
        avg_if(shot, shot_speed, "avg_shot_speed"),
        avg_if(shot, shot_angle, "avg_shot_angle"),
        avg_if(shot, shot_distance, "avg_shot_distance"),
        avg_if(pass_event, pass_speed, "avg_pass_speed"),
        avg_if(pass_event, pass_angle, "avg_pass_angle"),
        avg_if(pass_event, pass_distance, "avg_pass_distance"),
        max_if(shot, shot_speed, "max_shot_speed"),
        max_if(shot, shot_angle, "max_shot_angle"),
        max_if(shot, shot_distance, "max_shot_distance"),
        max_if(pass_event, pass_speed, "max_pass_speed"),
        max_if(pass_event, pass_angle, "max_pass_angle"),
        max_if(pass_event, pass_distance, "max_pass_distance"),
        count_if(air_dribble & flag_col("off_wall"), "off_wall_air_dribbles"),
        count_if(air_dribble & ~flag_col("off_wall") & ~flag_col("off_ceiling"), "off_ground_air_dribbles"),
        count_if(air_dribble & flag_col("off_ceiling"), "off_ceiling_air_dribbles"),
        count_if(boost_pad, "boost_pickups"),
        count_if(small_boost, "small_boost_pickups"),
        count_if(big_boost, "big_boost_pickups"),
        count_if(boost_overfill, "count_boost_overfill"),
        sum_if(boost_overfill, boost_overfill_amount, "boost_overfill_amount"),
        avg_if(boost_overfill, boost_overfill_amount, "avg_boost_overfill_amount"),
        count_if(boost_pad, "boost_pads_collected"),
        count_if(small_boost, "small_boost_pads_collected"),
        count_if(big_boost, "big_boost_pads_collected"),
        sum_if(boost_pad, boost_amount, "boost_collected"),
        sum_if(small_boost, boost_amount, "small_boost_amount_collected"),
        sum_if(big_boost, boost_amount, "big_boost_amount_collected"),
        count_if(boost_pad & boost_stolen, "boost_pads_stolen"),
        count_if(small_boost & boost_stolen, "small_boost_pads_stolen"),
        count_if(big_boost & boost_stolen, "big_boost_pads_stolen"),
        sum_if(boost_pad & boost_stolen, boost_amount, "boost_amount_stolen"),
        sum_if(
            small_boost & boost_stolen,
            boost_amount,
            "small_boost_amount_stolen",
        ),
        sum_if(
            big_boost & boost_stolen,
            boost_amount,
            "big_boost_amount_stolen",
        ),
        count_if(boost_pad & boost_protected, "boost_pads_protected"),
        count_if(small_boost & boost_protected, "small_boost_pads_protected"),
        count_if(big_boost & boost_protected, "big_boost_pads_protected"),
        sum_if(boost_pad & boost_protected, boost_amount, "boost_amount_protected"),
        sum_if(
            small_boost & boost_protected,
            boost_amount,
            "small_boost_amount_protected",
        ),
        sum_if(
            big_boost & boost_protected,
            boost_amount,
            "big_boost_amount_protected",
        ),
    ]
    expressions.extend(
        count_if(shot & condition, f"{name}_shots")
        for name, condition in shot_breakdowns
    )
    expressions.extend(
        count_if(shot_attempt & condition, f"{name}_shot_attempts")
        for name, condition in shot_breakdowns
    )
    expressions.extend(
        count_if(missed_shot & condition, f"{name}_missed_shots")
        for name, condition in shot_breakdowns
    )
    for name, condition in metric_breakdowns:
        expressions.extend(
            [
                count_if(shot & condition, f"{name}_shots"),
                count_if(shot_attempt & condition, f"{name}_shot_attempts"),
                count_if(missed_shot & condition, f"{name}_missed_shots"),
                count_if(missed_pass & condition, f"{name}_missed_passes"),
                count_if(goal & condition, f"{name}_goals"),
                count_if(save & condition, f"{name}_saves"),
            ]
        )

    if has_xg:
        xg = number_col("xG")
        expressions.append(sum_if(shot_attempt, xg, "expected_goals"))
        expressions.extend(
            sum_if(shot_attempt & condition, xg, f"{name}_expected_goals")
            for name, condition in shot_breakdowns
        )
        for name, condition in metric_breakdowns:
            expressions.append(sum_if(shot_attempt & condition, xg, f"{name}_expected_goals"))

    return (
        events.filter(string_col("event_player_1_id") != "")
        .group_by(
            [
                "replay_id",
                string_col("event_player_1_id").alias("player_id"),
            ]
        )
        .agg(expressions)
        .with_columns(
            pl.max_horizontal("_recorded_shots", "goals").alias("shots")
        )
        .drop("_recorded_shots")
    )


def secondary_event_stats(events, has_xg):
    assist = goal_event_expr() & (
        flag_col("official_assist") | (string_col("event_player_2_id") != "")
    )
    shot = shot_event_expr()
    opponent_bump = opponent_bump_expr()

    expressions = [
        official_stat_count("assist", assist, "assists"),
        count_if(shot, "shot_assists"),
        count_if(pl.col("event_type") == "pass", "passes_received"),
        count_if(pl.col("event_type") == "demo", "demos_taken"),
        count_if(opponent_bump, "bumps_taken"),
        count_if(pl.col("event_type") == "challenge", "challenge_losses"),
        count_if(pl.col("event_type") == "kickoff", "kickoff_losses"),
        count_if(pl.col("event_type") == "shadow", "shadows_taken"),
        count_if(pl.col("event_type") == "press", "presses_taken"),
        count_if(pl.col("event_type") == "fake", "fakes_taken"),
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


def teammate_bump_stats(events):
    teammate_bump = teammate_bump_expr()
    bumpers = events.filter(
        teammate_bump & (string_col("event_player_1_id") != "")
    ).select(
        [
            pl.col("replay_id"),
            string_col("event_player_1_id").alias("player_id"),
        ]
    )
    bumped = events.filter(
        teammate_bump & (string_col("event_player_2_id") != "")
    ).select(
        [
            pl.col("replay_id"),
            string_col("event_player_2_id").alias("player_id"),
        ]
    )

    return (
        pl.concat([bumpers, bumped], how="vertical_relaxed")
        .group_by(["replay_id", "player_id"])
        .agg(pl.len().alias("teammate_bumps"))
    )


def active_player_team_event_stats(
    rows,
    has_xg,
    player_slots: Sequence[str] = PLAYER_SLOTS,
):
    active_events = pl.concat(
        [
            player_slot_rows(
                rows,
                slot,
                include_inactive=True,
                player_slots=player_slots,
            )
            for slot in player_slots
        ],
        how="vertical_relaxed",
    ).filter(
        (pl.col("player_id") != "")
        & string_col("event_team").is_in(["blue", "orange"])
    )

    shot = shot_event_expr()
    shot_attempt = shot_attempt_expr()
    missed_shot = string_col("event_type") == "missed-shot"
    missed_pass = string_col("event_type") == "missed-pass"
    goal = goal_event_expr()
    same_team = pl.col("event_team") == pl.col("team")
    opposing_team = pl.col("event_team") != pl.col("team")
    active = pl.col("_active_for_event_context")
    controlled = flag_col("controlled")
    entry = pl.col("event_type") == "entry"
    exit_event = pl.col("event_type") == "exit"

    expressions = [
        official_stat_count(
            "shot", shot, "_recorded_shots_for", active & same_team
        ),
        official_stat_count(
            "shot", shot, "_recorded_shots_against", active & opposing_team
        ),
        official_stat_count("goal", goal, "goals_for", active & same_team),
        official_stat_count("goal", goal, "goals_against", active & opposing_team),
        count_if(active & shot_attempt & same_team, "shot_attempts_for"),
        count_if(active & shot_attempt & opposing_team, "shot_attempts_against"),
        count_if(active & missed_shot & same_team, "missed_shots_for"),
        count_if(active & missed_shot & opposing_team, "missed_shots_against"),
        count_if(active & missed_pass & same_team, "missed_passes_for"),
        count_if(active & missed_pass & opposing_team, "missed_passes_against"),
        count_if(
            active & (pl.col("event_type") == "demo") & same_team,
            "demos_applied_for",
        ),
        count_if(
            active & (pl.col("event_type") == "demo") & opposing_team,
            "demos_taken_against",
        ),
        count_if(active & (pl.col("event_type") == "fake") & same_team, "fakes_for"),
        count_if(active & (pl.col("event_type") == "fake") & opposing_team, "fakes_against"),
        count_if(active & entry & same_team, "entries_for"),
        count_if(active & entry & controlled & same_team, "controlled_entries_for"),
        count_if(active & entry & ~controlled & same_team, "uncontrolled_entries_for"),
        count_if(active & entry & opposing_team, "entries_against"),
        count_if(active & entry & controlled & opposing_team, "controlled_entries_against"),
        count_if(active & entry & ~controlled & opposing_team, "uncontrolled_entries_against"),
        count_if(active & exit_event & same_team, "exits_for"),
        count_if(active & exit_event & controlled & same_team, "controlled_exits_for"),
        count_if(active & exit_event & ~controlled & same_team, "uncontrolled_exits_for"),
        count_if(active & exit_event & opposing_team, "exits_against"),
        count_if(active & exit_event & controlled & opposing_team, "controlled_exits_against"),
        count_if(active & exit_event & ~controlled & opposing_team, "uncontrolled_exits_against"),
    ]

    if has_xg:
        xg = number_col("xG")
        expressions.extend(
            [
                sum_if(active & shot_attempt & same_team, xg, "expected_goals_for"),
                sum_if(active & shot_attempt & opposing_team, xg, "expected_goals_against"),
            ]
        )

    return (
        active_events.group_by(["replay_id", "player_id"])
        .agg(expressions)
        .with_columns(
            [
                pl.max_horizontal("_recorded_shots_for", "goals_for").alias(
                    "shots_for"
                ),
                pl.max_horizontal(
                    "_recorded_shots_against", "goals_against"
                ).alias("shots_against"),
            ]
        )
        .drop(["_recorded_shots_for", "_recorded_shots_against"])
    )


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
        "shot_attempts",
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
            pl.when(pl.col("shot_attempts") > 0)
            .then(pl.col("expected_goals") / pl.col("shot_attempts"))
            .otherwise(0.0)
            .alias("expected_goals_per_shot_attempt"),
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
    average_metric_columns = {
        column for column in metric_columns if column.startswith("avg_")
    }
    max_metric_columns = {column for column in metric_columns if column.startswith("max_")}
    special_metric_columns = average_metric_columns | max_metric_columns
    aggregations = [
        pl.col(column).sum().alias(column)
        for column in metric_columns
        if column not in group_columns
        and column != "games_played"
        and column not in special_metric_columns
    ]

    if "replay_id" in group_columns:
        for column in sorted(special_metric_columns):
            if column not in group_columns:
                aggregations.append(pl.col(column).drop_nulls().first().alias(column))
    else:
        if {
            "reaction_time_total",
            "reaction_time_count",
            "avg_reaction_time",
        }.issubset(metric_columns):
            aggregations.append(
                pl.when(pl.col("reaction_time_count").sum() > 0)
                .then(
                    pl.col("reaction_time_total").sum()
                    / pl.col("reaction_time_count").sum()
                )
                .otherwise(0.0)
                .alias("avg_reaction_time")
            )

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
        aggregations.append(pl.col(column).drop_nulls().last().alias(column))

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
    float_columns = [
        column
        for column, dtype in stats.schema.items()
        if dtype in {pl.Float32, pl.Float64}
    ]
    if float_columns:
        stats = stats.with_columns(
            [pl.col(column).round(12) for column in float_columns]
        )
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
    has_frame_data_hint: bool | None = None,
    has_presence_events_hint: bool | None = None,
) -> pl.DataFrame:
    if xg_model_path is not None:
        from .xg import apply_xg_to_pbp

        rows = apply_xg_to_pbp(rows, xg_model_path).lazy()

    schema = rows.collect_schema().names()
    has_frame_data = False

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
            number_col("event_length").alias("event_length"),
            number_col("event_duration").alias("event_duration"),
        ]
    ).cache()
    has_frame_data = has_frame_data_hint
    if has_frame_data is None:
        has_frame_data = (
            rows.select((string_col("event_type") == "").any().alias("has_frame_rows"))
            .collect()
            .item()
        )
    events = rows.filter(string_col("event_type") != "")
    has_presence_events = has_presence_events_hint
    if has_presence_events is None:
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

    player_slots = PLAYER_SLOTS
    if has_frame_data:
        max_team_size = int(
            rows.select(number_col("team_size").max().alias("team_size"))
            .collect()
            .item()
            or 4
        )
        max_team_size = max(1, min(max_team_size, 4))
        player_slots = [
            slot
            for slot in PLAYER_SLOTS
            if int(slot.rsplit("_", 1)[1]) <= max_team_size
        ]

    if has_presence_events and not has_frame_data:
        rows = rows.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("_stats_frame_delta_seconds"),
                pl.lit(None, dtype=pl.Float64).alias(
                    "_stats_event_frame_delta_seconds"
                ),
            ]
        )
        events = rows.filter(string_col("event_type") != "")
        player_source = events
    else:
        rows = add_frame_delta_seconds(rows)
        events = rows.filter(string_col("event_type") != "")
        player_source = rows

    players = player_frame(
        player_source,
        player_slots=player_slots,
        include_event_presence=not has_frame_data,
    )
    primary = primary_event_stats(events, has_xg, has_frame_data)
    secondary = secondary_event_stats(events, has_xg)
    teammate_bumps = teammate_bump_stats(events)
    active_team = active_player_team_event_stats(events, has_xg, player_slots)

    stats = players.with_columns(
        [
            pl.lit(1, dtype=pl.Int64).alias("games_played"),
        ]
    ).join(primary, on=["replay_id", "player_id"], how="left")

    stats = stats.join(secondary, on=["replay_id", "player_id"], how="left")
    stats = stats.join(teammate_bumps, on=["replay_id", "player_id"], how="left")
    stats = stats.join(active_team, on=["replay_id", "player_id"], how="left")

    if not has_frame_data:
        frame_only_columns = [
            column
            for column in (
                FRAME_ONLY_TIME_COLUMNS
                + FRAME_ONLY_EVENT_TIME_COLUMNS
                + FRAME_ONLY_VALUE_COLUMNS
            )
            if column in stats.collect_schema().names()
        ]
        if frame_only_columns:
            stats = stats.drop(frame_only_columns)

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
    gpu: str | None,
) -> pl.DataFrame:
    replay_inputs, tabular_inputs = _split_path_inputs(frames)

    if replay_inputs:
        tabular_inputs.extend(
            _parse_replay_inputs(
                replay_inputs,
                workers=workers,
                parse_export=parse_export,
                force=force,
                limit=limit,
                gpu=gpu,
            )
        )
    elif limit is not None:
        tabular_inputs = tabular_inputs[:limit]

    requested_workers = max(int(workers or 1), 1)
    if len(tabular_inputs) <= STATS_SINGLE_SCAN_FILE_LIMIT:
        rows = _lazy_scan_files(tabular_inputs, extra_columns=xg_columns)
        stems = [path.stem for path in tabular_inputs]
        has_frame_data_hint = (
            True
            if stems and all(stem.endswith("_frames") for stem in stems)
            else False
            if stems and all(stem.endswith("_pbp") for stem in stems)
            else None
        )
        stats = calculate_stats_from_lazy_rows(
            rows,
            group_by=group_by,
            rates=rates,
            xg_model_path=xg_model_path,
            xg_columns=xg_columns,
            has_frame_data_hint=has_frame_data_hint,
            has_presence_events_hint=False if has_frame_data_hint is True else None,
        )
        if tabular_inputs:
            print(
                f"\r\x1b[2Kbuilt stats {len(tabular_inputs)}/{len(tabular_inputs)}",
                flush=True,
            )
        return stats

    files_per_batch = max(
        1,
        min(
            STATS_FILE_BATCH_SIZE,
            (len(tabular_inputs) + requested_workers - 1) // requested_workers,
        ),
    )
    bytes_per_batch = max(STATS_BATCH_BYTES // min(requested_workers, STATS_MAX_PARALLEL_BATCHES), 16 * 1024 * 1024)
    batches: list[list[Path]] = []
    batch: list[Path] = []
    batch_bytes = 0
    for path in tabular_inputs:
        try:
            path_bytes = path.stat().st_size
        except OSError:
            path_bytes = 0

        exceeds_limit = batch and (
            len(batch) >= files_per_batch
            or batch_bytes + path_bytes > bytes_per_batch
        )
        if exceeds_limit:
            batches.append(batch)
            batch = []
            batch_bytes = 0

        batch.append(path)
        batch_bytes += path_bytes

    if batch:
        batches.append(batch)

    def build_batch(batch_paths: list[Path]) -> pl.DataFrame:
        rows = _lazy_scan_files(batch_paths, extra_columns=xg_columns)
        return calculate_stats_from_lazy_rows(
            rows,
            group_by=DEFAULT_GROUP_BY,
            rates=False,
            xg_model_path=xg_model_path,
            xg_columns=xg_columns,
            has_frame_data_hint=(
                True
                if batch_paths
                and all(path.stem.endswith("_frames") for path in batch_paths)
                else False
                if batch_paths
                and all(path.stem.endswith("_pbp") for path in batch_paths)
                else None
            ),
            has_presence_events_hint=(
                False
                if batch_paths
                and all(path.stem.endswith("_frames") for path in batch_paths)
                else None
            ),
        )

    total = len(tabular_inputs)
    done = 0
    partials_by_index: dict[int, pl.DataFrame] = {}
    max_workers = min(requested_workers, STATS_MAX_PARALLEL_BATCHES, len(batches))

    def show_batch_progress(batch_paths: list[Path]) -> None:
        nonlocal done
        for path in batch_paths:
            done += 1
            replay_id = path.stem.removesuffix("_pbp").removesuffix("_frames")
            print(
                f"\r\x1b[2Kbuilt stats {replay_id} ({done}/{total})",
                end="",
                flush=True,
            )

    try:
        if max_workers == 1:
            for index, batch_paths in enumerate(batches):
                partials_by_index[index] = build_batch(batch_paths)
                show_batch_progress(batch_paths)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(build_batch, batch_paths): (index, batch_paths)
                    for index, batch_paths in enumerate(batches)
                }
                for future in as_completed(futures):
                    index, batch_paths = futures[future]
                    partials_by_index[index] = future.result()
                    show_batch_progress(batch_paths)
    finally:
        if done:
            print(flush=True)

    partials = [partials_by_index[index] for index in range(len(batches))]
    try:
        if len(partials) == 1:
            stats = finish_stats(
                partials[0].lazy(),
                group_by=group_by,
                rates=rates,
                has_xg="expected_goals" in partials[0].columns,
            )
        else:
            combined = pl.concat(partials, how="vertical_relaxed").lazy()
            has_xg = any("expected_goals" in partial.columns for partial in partials)
            stats = finish_stats(combined, group_by=group_by, rates=rates, has_xg=has_xg)
            del combined
    finally:
        partials_by_index.clear()
        del partials
        gc.collect()

    return stats


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
    gpu: str | None = None,
) -> pl.DataFrame:
    """Build per-player replay stats from parsed play-by-play or frame data.

    Args:
        frames: Replay, CSV, Parquet, folder, or parsed replay data in a
            supported Polars, pandas, or list-based tabular shape.
        group_by: Output grouping columns. Defaults to one row per replay and
            player.
        rates: Whether to add per-five-minute and per-game rate columns.
        workers: Number of parallel stats workers, also passed to the Rust
            parser when replay inputs need parsing.
        parse_export: Output folder for generated PBP Parquet files when
            replay inputs need parsing.
        force: Whether to overwrite existing parser exports for replay inputs.
        limit: Optional file or replay count limit.
        xg_model_path: Optional saved xG model file or folder. When provided,
            source rows are scored before stats are aggregated.
        gpu: Optional parser GPU mode when replay inputs need parsing. Python
            ``None`` is the default and uses CPU only.

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
            gpu=gpu,
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
            gpu=gpu,
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
    return_type: Literal["export", "polars", "pandas", "list"] = "polars",
    export: str | PathLike[str] | None = None,
    group_by: Sequence[str] | str | None = None,
    rates: bool = False,
    workers: int = 4,
    parse_export: str | PathLike[str] = "data/frames",
    force: bool = False,
    limit: int | None = None,
    xg_model_path: str | PathLike[str] | None = None,
    gpu: str | None = None,
):
    """Aggregate replay stats and optionally export them.

    Args:
        frames: Replay file, replay folder, one or more replay files, CSV or
            Parquet PBP/frame file, folder of CSV or Parquet PBP/frame files,
            one or more CSV or Parquet PBP/frame files, or parsed replay data
            in a supported tabular format.
        return_type: Whether to return the export path, Polars, pandas, or list output.
        export: Optional output path for CSV or Parquet export.
        group_by: Output grouping columns. Defaults to ``["replay_id",
            "player_id"]``.
        rates: Whether to add per-five-minute and per-game rate columns.
        workers: Number of parallel stats workers, also passed to the Rust
            parser when replay inputs need parsing.
        parse_export: Output folder for generated PBP Parquet files when
            replay inputs need parsing.
        force: Whether to overwrite existing parser exports for replay inputs.
        limit: Optional file or replay count limit.
        xg_model_path: Optional saved xG model file or folder. When provided,
            source rows are scored before expected-goal stats are aggregated.
        gpu: Optional parser GPU mode when replay inputs need Rust parsing.
            Python ``None`` is the default and uses CPU only.

    Returns:
        Aggregated stats in the requested return format.
    """
    return_type = str(return_type).lower()
    if export is not None and xg_model_path is None and _is_path_input(frames):
        replay_inputs, tabular_inputs = _split_path_inputs(frames)
        export_path = _resolve_input_path(export)
        if replay_inputs and not tabular_inputs and export_path.suffix.lower() == ".csv":
            _native_stats_export(
                replay_inputs,
                export_path,
                workers=workers,
                force=True,
                limit=limit,
                gpu=gpu,
            )
            stats = finish_stats(
                pl.read_csv(export_path, null_values=CSV_NULL_VALUES).lazy(),
                group_by=group_by,
                rates=rates,
                has_xg=False,
            )
            if export_path.exists():
                export_path.unlink()
            stats.write_csv(export_path)
            if return_type in {"export", "polars"}:
                del stats
                gc.collect()
                return export_path
            if return_type == "pandas":
                return stats.to_pandas()
            if return_type == "list":
                return stats.to_dicts()
            raise ValueError(
                "return_type must be one of: 'export', 'polars', 'pandas', 'list'"
            )

    stats = calculate_player_replay_stats(
        frames,
        group_by=group_by,
        rates=rates,
        workers=workers,
        parse_export=parse_export,
        force=force,
        limit=limit,
        xg_model_path=xg_model_path,
        gpu=gpu,
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
            return_type = "export"

    if return_type == "export":
        if export is None:
            raise ValueError("return_type='export' requires an export path")
        del stats
        gc.collect()
        return export

    if return_type == "polars":
        return stats

    if return_type == "pandas":
        return stats.to_pandas()

    if return_type == "list":
        return stats.to_dicts()

    raise ValueError("return_type must be one of: 'export', 'polars', 'pandas', 'list'")
