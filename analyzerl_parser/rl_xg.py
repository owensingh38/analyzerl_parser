import os
import json
import gc
import joblib
import numpy as np
import polars as pl
from pathlib import Path
from scipy import sparse
from typing import Any, Mapping
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency guard
    optuna = None

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

PBP_FOLDER = 'data/frames'
MODEL_FOLDER = 'data/models/xg'
XG_LABEL = 'standard'
SUPPORTED_INPUT_SUFFIXES = {'.csv', '.parquet'}
SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
BACK_NET_Y = 6000.0
CEILING_Z = 2044.0
GOAL_HEIGHT = 642.775
GOAL_CENTER_TO_POST = 892.755
GRAVITY = 650.0
BALL_RADIUS = 91.25
SUPERSONIC_THRESHOLD = 2200.0
GROUND_Z = 40.0
WALL_DISTANCE = 350.0
FIELD_X = SIDE_WALL_X
FIELD_Y = BACK_WALL_Y
FIELD_Z = CEILING_Z
SHOT_GOAL_PROJECTION_MAX_SECONDS = 5.0
MISSING_OPPONENT_POSITION = 1_000_000.0
CAR_INTERCEPT_MIN_SPEED = 500.0
CAR_INTERCEPT_BOOST_SPEED_BONUS = 12.0
INTERCEPT_MARGIN_SECONDS = 0.15
TRAJECTORY_SAMPLE_COUNT = 12
AERIAL_INTERCEPT_Z = 300.0
DEFAULT_HITBOX_LENGTH = 118.0074
DEFAULT_HITBOX_WIDTH = 84.2
DEFAULT_HITBOX_HEIGHT = 36.15907
DEFAULT_HITBOX_OFFSET = 13.87566
DEFAULT_HITBOX_ELEVATION = 20.755
DEFAULT_XGB_SPACE = {
    'n_estimators': ('int', 300, 2000),
    'max_depth': ('int', 2, 8),
    'learning_rate': ('float', 0.01, 0.2, True),
    'subsample': ('float', 0.6, 1.0, False),
    'colsample_bytree': ('float', 0.6, 1.0, False),
    'min_child_weight': ('float', 1.0, 8.0, True),
    'reg_lambda': ('float', 1e-3, 20.0, True),
    'reg_alpha': ('float', 1e-3, 10.0, True),
    'gamma': ('float', 1e-3, 10.0, True),
    'max_bin': ('int', 128, 512),
}


def unique_list(values):
    return list(dict.fromkeys(values))


NUMERIC_ROLE_FIELDS = [
    'pos_x', 'pos_y', 'pos_z',
    'vel_x', 'vel_y', 'vel_z',
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
    'rot_x', 'rot_y', 'rot_z',
    'boost', 'boost_active', 'boost_collect',
    'throttle', 'steer', 'handbrake', 'ball_cam',
    'dodge_active', 'jump_active', 'double_jump_active', 'supersonic',
    'jumped', 'flipped',
    'distance_to_ball', 'angle_to_ball',
    'distance_to_own_net', 'angle_to_own_net',
    'distance_to_opp_net', 'angle_to_opp_net',
    'distance_from_last_event'
]
CATEGORICAL_ROLE_FIELDS = ['id', 'car_id', 'rank', 'rank_tier', 'pro_player']

BALL_EVENT_FRAME_CONTINUOUS = [
    'field_start_x', 'field_start_y', 'field_start_z',
    'field_end_x', 'field_end_y', 'field_end_z',
    'distance_start_to_end',
    'angle_start_to_end_xy', 'angle_start_to_end_xz',
    'velocity_x', 'velocity_y', 'velocity_z',
    'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
    'arc_peak',
    'arc_peak_distance',
    'contact_x', 'contact_y', 'contact_z',
    'field_start_x_last',
    'field_start_y_last',
    'field_start_z_last',
    'distance_start_last_to_start',
    'angle_start_last_to_start_xy',
    'angle_start_last_to_start_xz',
    'speed_from_last',
]
BALL_EVENT_FRAME_BOOLEAN = ['projected_inside_net']
SHOOTER_EVENT_FRAME_CONTINUOUS = [
    'velocity_x', 'velocity_y', 'velocity_z',
    'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
    'yaw', 'pitch', 'roll',
    'boost_amount',
    'contact_x', 'contact_y', 'contact_z',
    'angle_contact_to_ball_start_xy',
    'angle_contact_to_ball_start_xz',
    'angle_contact_to_ball_end_xy',
    'angle_contact_to_ball_end_xz',
    'angle_front_car_to_ground',
]
SHOOTER_EVENT_FRAME_BOOLEAN = ['powersliding', 'boosting', 'flipping']
OPPONENT_EVENT_FRAME_CONTINUOUS = [
    'position_x', 'position_y', 'position_z',
    'velocity_x', 'velocity_y', 'velocity_z',
    'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
    'yaw', 'pitch', 'roll',
    'boost_amount',
    'proximity_to_ball_start_x', 'proximity_to_ball_start_y', 'proximity_to_ball_start_z',
    'proximity_to_ball_end_x', 'proximity_to_ball_end_y', 'proximity_to_ball_end_z',
    'proximity_to_defensive_net_location_x',
    'proximity_to_defensive_net_location_y',
    'proximity_to_defensive_net_location_z',
    'angle_proximity_to_ball_start_xy',
    'angle_proximity_to_ball_start_xz',
    'angle_proximity_to_ball_end_xy',
    'angle_proximity_to_ball_end_xz',
    'angle_proximity_to_defensive_net_to_defensive_net_xy',
    'angle_proximity_to_defensive_net_to_defensive_net_xz',
    'time_to_ball_end',
    'min_time_to_ball_trajectory',
    'intercept_time_margin',
    'intercept_x', 'intercept_y', 'intercept_z',
    'min_distance_to_trajectory',
    'boost_needed_to_intercept',
    'boost_surplus_at_intercept',
    'time_to_ball_start_estimate',
    'time_to_ball_end_estimate',
    'time_to_defensive_net_estimate',
    'time_to_shot_path_estimate',
    'shot_path_intercept_time_margin',
    'shot_path_intercept_x', 'shot_path_intercept_y', 'shot_path_intercept_z',
    'min_distance_to_shot_path',
    'angle_front_car_to_ground',
]
OPPONENT_EVENT_FRAME_BOOLEAN = [
    'on_field',
    'powersliding',
    'boosting',
    'flipping',
    'intercept_possible',
    'intercept_requires_aerial',
    'shot_path_intercept_possible',
]
CONTEXT_EVENT_FRAME_BOOLEAN = [
    'off_demo',
    'off_kickoff',
    'off_challenge_win',
    'off_bump',
    'off_controlled_entry',
    'off_controlled_exit',
    'off_retrieval',
    'off_uncontrolled_entry',
    'off_uncontrolled_exit',
    'off_air_dribble',
    'off_ground_dribble',
    'off_flick',
    'pass_in_play',
    'rebound',
    'off_flip_reset',
    'off_double_tap',
    'off_wall',
    'off_ceiling',
]
CONTEXT_EVENT_FRAME_CONTINUOUS = [
    'history_seconds_since_kickoff',
    'history_weighted_touch_count',
    'history_weighted_turnover_count',
    'history_weighted_pass_count',
    'history_weighted_shot_count',
    'history_weighted_demo_count',
    'history_weighted_bump_count',
    'history_weighted_challenge_count',
    'seconds_from_last_event',
]
CONTEXT_FEATURES = CONTEXT_EVENT_FRAME_BOOLEAN + CONTEXT_EVENT_FRAME_CONTINUOUS

MODEL_CONTINUOUS = [
    f'ball_{field}' for field in BALL_EVENT_FRAME_CONTINUOUS
] + [
    f'context_{field}' for field in CONTEXT_EVENT_FRAME_CONTINUOUS
]
MODEL_BOOLEAN = [
    f'ball_{field}' for field in BALL_EVENT_FRAME_BOOLEAN
] + [
    f'context_{field}' for field in CONTEXT_EVENT_FRAME_BOOLEAN
]
MODEL_SHOOTER = SHOOTER_EVENT_FRAME_CONTINUOUS + SHOOTER_EVENT_FRAME_BOOLEAN
MODEL_OPPONENT = OPPONENT_EVENT_FRAME_CONTINUOUS + OPPONENT_EVENT_FRAME_BOOLEAN


def previous_event_type_values():
    return [
        value.replace('previous_event_type_', '')
        for value in MODEL_CONTINUOUS
        if value.startswith('previous_event_type_')
    ]


def model_continuous_columns():
    columns = [
        value
        for value in MODEL_CONTINUOUS
        if not value.startswith('previous_event_type_')
    ]
    if previous_event_type_values():
        columns.insert(0, 'previous_event_type')
    return unique_list(columns)


def model_non_player_columns():
    return unique_list(model_continuous_columns() + MODEL_BOOLEAN)


def model_role_numeric_fields():
    model_fields = set(MODEL_SHOOTER + MODEL_OPPONENT)
    required_source_fields = {
        'pos_x', 'pos_y', 'pos_z',
        'vel_x', 'vel_y', 'vel_z',
        'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
        'rot_x', 'rot_y', 'rot_z',
        'boost', 'boost_active', 'handbrake', 'dodge_active', 'flipped',
    }
    return [field for field in NUMERIC_ROLE_FIELDS if field in model_fields or field in required_source_fields]


def model_role_source_fields():
    return unique_list(['id'] + model_role_numeric_fields())


PBP_ID_COLUMNS = [
    'game_id',
    'event_number',
    'event_type',
    'frame_number',
    'seconds_elapsed',
    'team_size',
    'event_player_1_id',
    'event_player_1_team',
]
PBP_DERIVED_SOURCE_COLUMNS = [
    'event_ball_pos_x',
    'event_ball_pos_y',
    'event_ball_pos_z',
    'ball_pos_x',
    'ball_pos_y',
    'ball_pos_z',
    'ball_vel_x',
    'ball_vel_y',
    'ball_vel_z',
    'ball_ang_vel_x',
    'ball_ang_vel_y',
    'ball_ang_vel_z',
    'ball_speed_from_last_event',
    *CONTEXT_FEATURES,
]
BASE_PBP_COLS = unique_list(
    PBP_ID_COLUMNS
    + PBP_DERIVED_SOURCE_COLUMNS
)

NULL_VALUES = ['', 'NA', 'N/A', 'NaN', 'nan', 'None', 'none', 'null', 'NULL']


def desired_pbp_columns():
    columns = list(BASE_PBP_COLS)
    for team in ['blue', 'orange']:
        for player_number in range(1, 5):
            slot = f'{team}_player_{player_number}'
            columns.extend(f'{slot}_{field}' for field in unique_list(CATEGORICAL_ROLE_FIELDS + NUMERIC_ROLE_FIELDS))
    return list(dict.fromkeys(columns))


def model_pbp_columns():
    columns = list(BASE_PBP_COLS)
    player_fields = model_role_source_fields()
    for team in ['blue', 'orange']:
        for player_number in range(1, 5):
            slot = f'{team}_player_{player_number}'
            columns.extend(f'{slot}_{field}' for field in player_fields)
    return list(dict.fromkeys(columns))


def file_columns(path):
    if path.endswith('.parquet'):
        try:
            return pl.scan_parquet(path).collect_schema().names()
        except Exception:
            return None
    with open(path, 'r', encoding='utf-8-sig') as handle:
        return handle.readline().rstrip('\n\r').split(',')


def pbp_schema_overrides(header_set, requested):
    string_columns = {
        'game_id',
        'event_type',
        'event_player_1_id',
        'event_player_1_team',
        'previous_event_type',
    }
    integer_columns = {'event_number', 'frame_number'}
    boolean_columns = set(MODEL_BOOLEAN)
    boolean_columns.update(CONTEXT_EVENT_FRAME_BOOLEAN)
    for team in ['blue', 'orange']:
        for player_number in range(1, 5):
            slot = f'{team}_player_{player_number}'
            for field in CATEGORICAL_ROLE_FIELDS:
                string_columns.add(f'{slot}_{field}')
            for field in ['boost_active', 'handbrake', 'ball_cam', 'dodge_active',
                          'jump_active', 'double_jump_active', 'supersonic', 'jumped', 'flipped']:
                boolean_columns.add(f'{slot}_{field}')
    for field in SHOOTER_EVENT_FRAME_BOOLEAN:
        boolean_columns.add(f'shooter_{field}')
    for role_number in range(1, 4):
        for field in OPPONENT_EVENT_FRAME_BOOLEAN:
            boolean_columns.add(f'opponent_{role_number}_{field}')

    overrides = {}
    for column in header_set:
        if column in string_columns:
            overrides[column] = pl.Utf8
        elif column in integer_columns:
            overrides[column] = pl.Int32
        elif column in boolean_columns:
            overrides[column] = pl.Boolean
        elif column in requested:
            overrides[column] = pl.Float32
    return overrides


def pbp_segment_expr():
    team_size = pl.col('team_size').cast(pl.Float32, strict=True)
    return (
        pl.when(team_size == 1)
        .then(pl.lit('duels'))
        .when(team_size == 2)
        .then(pl.lit('doubles'))
        .when(team_size == 3)
        .then(pl.lit('standard'))
        .when(team_size == 4)
        .then(pl.lit('quads'))
        .otherwise(pl.lit('standard'))
    )


def scan_pbp_polars(paths, requested_columns=None, event_filter=True):
    requested = list(dict.fromkeys(requested_columns or desired_pbp_columns()))
    if 'event_type' not in requested:
        requested.insert(0, 'event_type')
    schema_groups = {}
    skipped = 0
    for path in paths:
        header = file_columns(path)
        if header is None:
            skipped += 1
            continue
        if 'event_type' not in header:
            skipped += 1
            continue
        kind = 'parquet' if path.endswith('.parquet') else 'csv'
        schema_groups.setdefault((kind, tuple(header)), []).append(path)

    if not schema_groups:
        return pl.DataFrame(schema={column: pl.Null for column in requested}).lazy()

    lazy_frames = []
    for (kind, header), group_paths in schema_groups.items():
        header_set = set(header)
        include_columns = [column for column in requested if column in header_set]
        missing_columns = [column for column in requested if column not in header_set]
        select_exprs = [pl.col(column) for column in include_columns]
        select_exprs.extend(pl.lit(None).alias(column) for column in missing_columns)
        if kind == 'parquet':
            scan = pl.scan_parquet(group_paths)
        else:
            scan = pl.scan_csv(
                group_paths,
                null_values=NULL_VALUES,
                schema_overrides=pbp_schema_overrides(header_set, requested),
                infer_schema_length=10000,
                ignore_errors=False,
                cache=False,
                low_memory=True,
                glob=False,
            )
        lazy_frame = scan.select(select_exprs).select(requested)
        if event_filter:
            lazy_frame = lazy_frame.filter(pl.col('event_type').is_in(['shot', 'goal']))
        lazy_frames.append(lazy_frame)

    return pl.concat(lazy_frames, how='vertical_relaxed')


def scan_single_pbp_polars(path, requested, event_filter=True):
    header = file_columns(path)
    if header is None or 'event_type' not in header:
        return None
    header_set = set(header)
    include_columns = [column for column in requested if column in header_set]
    missing_columns = [column for column in requested if column not in header_set]
    select_exprs = [pl.col(column) for column in include_columns]
    select_exprs.extend(pl.lit(None).alias(column) for column in missing_columns)
    if path.endswith('.parquet'):
        scan = pl.scan_parquet(path)
    else:
        scan = pl.scan_csv(
            path,
            null_values=NULL_VALUES,
            schema_overrides=pbp_schema_overrides(header_set, requested),
            infer_schema_length=10000,
            ignore_errors=False,
            cache=False,
            low_memory=True,
            glob=False,
        )
    lazy_frame = scan.select(select_exprs).select(requested)
    if event_filter:
        lazy_frame = lazy_frame.filter(pl.col('event_type').is_in(['shot', 'goal']))
    return lazy_frame


def collect_pbp_polars_skip_corrupt_parquet(paths, requested_columns=None, event_filter=True):
    requested = list(dict.fromkeys(requested_columns or desired_pbp_columns()))
    if 'event_type' not in requested:
        requested.insert(0, 'event_type')
    frames = []
    for path in paths:
        try:
            lazy_frame = scan_single_pbp_polars(path, requested, event_filter=event_filter)
            if lazy_frame is None:
                continue
            frames.append(lazy_frame.collect(engine='streaming'))
        except Exception:
            if path.endswith('.parquet'):
                continue
            raise
    if not frames:
        return pl.DataFrame(schema={column: pl.Null for column in requested})
    return pl.concat(frames, how='vertical_relaxed')


def prepare_model_shots(lazy_frame):
    start_exprs = [
        pl.coalesce([
            pl.col(f'event_ball_pos_{axis}').cast(pl.Float32, strict=False),
            pl.col(f'ball_pos_{axis}').cast(pl.Float32, strict=False),
        ]).alias(f'_ball_field_start_{axis}')
        for axis in ['x', 'y', 'z']
    ]
    last_exprs = [
        pl.col(f'_ball_field_start_{axis}')
        .shift(1)
        .over('game_id')
        .alias(f'_ball_field_start_{axis}_last')
        for axis in ['x', 'y', 'z']
    ]
    return (
        lazy_frame
        .with_columns(start_exprs)
        .sort(['game_id', 'event_number'])
        .with_columns(last_exprs)
        .filter(pl.col('event_type').is_in(['shot', 'goal']))
        .with_columns([
            (pl.col('event_type') == 'goal').cast(pl.Int8).alias('is_goal'),
            pl.col('event_number').cast(pl.Int32, strict=False).alias('event_number'),
            pl.lit(XG_LABEL).alias('xG_model_segment'),
        ])
    )


def input_files(input_folder):
    input_folder = Path(input_folder).expanduser().resolve()

    if not input_folder.exists():
        raise FileNotFoundError(f'Input folder does not exist: {input_folder}')

    if not input_folder.is_dir():
        raise NotADirectoryError(f'Input path must be a folder: {input_folder}')

    paths = [
        str(path)
        for path in sorted(input_folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    ]

    if not paths:
        raise FileNotFoundError(
            f'No supported frame/PBP files found in {input_folder}. '
            f'Expected one or more of: {sorted(SUPPORTED_INPUT_SUFFIXES)}'
        )

    return paths


def load_pbp_polars(paths):
    return scan_pbp_polars(paths).collect(engine='streaming')


def array_float(df, column, default=np.nan):
    if column not in df.columns:
        return np.full(df.height, default, dtype=np.float32)
    series = df.get_column(column)
    if series.dtype == pl.Boolean:
        values = series.cast(pl.Float32, strict=False).to_numpy()
        return np.asarray(values, dtype=np.float32)
    if series.dtype in [pl.String, pl.Categorical, pl.Enum, pl.Utf8]:
        series = (
            series.cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .replace({
                'true': '1',
                'false': '0',
                'yes': '1',
                'no': '0',
            })
        )
    values = series.cast(pl.Float32, strict=False).to_numpy()
    return np.asarray(values, dtype=np.float32)


def array_string(df, column, default=''):
    if column not in df.columns:
        return np.full(df.height, default, dtype=str)
    values = df.get_column(column).cast(pl.Utf8, strict=True).fill_null(default).to_numpy()
    return np.asarray(values, dtype=str)


def array_bool(df, column):
    if column not in df.columns:
        return np.zeros(df.height, dtype=np.float64)
    series = df.get_column(column)
    if series.dtype == pl.Boolean:
        return series.cast(pl.Float64, strict=True).fill_null(0.0).to_numpy()
    lowered = series.cast(pl.Utf8, strict=False).str.to_lowercase()
    return (
        lowered
        .replace({
            'true': '1',
            '1': '1',
            'yes': '1',
            'y': '1',
            'false': '0',
            '0': '0',
            'no': '0',
            'n': '0',
        })
        .cast(pl.Float64, strict=True)
        .fill_null(0.0)
        .to_numpy()
    )


def safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=np.isfinite(denominator) & (np.abs(denominator) > 1e-9),
    )


def safe_speed(x, y, z):
    return np.sqrt(np.nan_to_num(x) ** 2 + np.nan_to_num(y) ** 2 + np.nan_to_num(z) ** 2)


def boost_units(values):
    return np.where(values > 100.0, np.rint(values * 100.0 / 255.0), values)


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def prefer_float(df, column, computed):
    computed = np.asarray(computed, dtype=np.float64)
    if column not in df.columns:
        return computed
    existing = array_float(df, column).astype(np.float64, copy=False)
    return np.where(np.isfinite(existing), existing, computed)


def preserve_missing_sentinel(values, transformed):
    values = np.asarray(values, dtype=np.float64)
    transformed = np.asarray(transformed, dtype=np.float64)
    missing = np.abs(values) >= MISSING_OPPONENT_POSITION * 0.5
    return np.where(missing, MISSING_OPPONENT_POSITION, transformed)


def orient_feature_values(column, values, direction):
    values = np.asarray(values, dtype=np.float64)
    if (
        column.endswith('_x')
        or column.endswith('_y')
        or column.endswith('_x_last')
        or column.endswith('_y_last')
    ):
        return preserve_missing_sentinel(values, values * direction)
    if column.endswith('_xy') or column.endswith('_yaw'):
        oriented = normalize_angle(values + np.where(direction < 0, np.pi, 0.0))
        return preserve_missing_sentinel(values, oriented)
    return values


def feature_float(df, column, computed, direction, prefer_existing=False):
    values = prefer_float(df, column, computed) if prefer_existing else np.asarray(computed, dtype=np.float64)
    return orient_feature_values(column, values, direction)


def angle_xy_from_points(start_x, start_y, end_x, end_y):
    return np.arctan2(end_y - start_y, end_x - start_x)


def angle_xz_from_points(start_x, start_y, start_z, end_x, end_y, end_z):
    horizontal = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    return np.arctan2(end_z - start_z, horizontal)


def project_ball_end(start_x, start_y, start_z, vel_x, vel_y, vel_z, shooter_team):
    goal_y = np.where(shooter_team == 'orange', -BACK_WALL_Y, BACK_WALL_Y)
    time_to_goal = safe_divide(goal_y - start_y, vel_y)
    valid_goal_time = (
        np.isfinite(time_to_goal)
        & (time_to_goal > 0.0)
        & (time_to_goal <= SHOT_GOAL_PROJECTION_MAX_SECONDS)
    )
    projection_time = np.where(valid_goal_time, time_to_goal, SHOT_GOAL_PROJECTION_MAX_SECONDS)
    end_x = start_x + vel_x * projection_time
    end_y = start_y + vel_y * projection_time
    end_z = np.maximum(start_z + vel_z * projection_time - 0.5 * GRAVITY * projection_time ** 2, 0.0)
    projected_inside_net = (
        valid_goal_time
        & (np.abs(end_x) <= GOAL_CENTER_TO_POST)
        & (end_z >= 0.0)
        & (end_z <= GOAL_HEIGHT)
    )
    peak_time = np.minimum(np.maximum(safe_divide(vel_z, np.full_like(vel_z, GRAVITY)), 0.0), projection_time)
    peak_x = start_x + vel_x * peak_time
    peak_y = start_y + vel_y * peak_time
    peak_z = np.maximum(start_z + vel_z * peak_time - 0.5 * GRAVITY * peak_time ** 2, 0.0)
    peak_distance = np.sqrt(
        (peak_x - start_x) ** 2
        + (peak_y - start_y) ** 2
        + (peak_z - start_z) ** 2
    )
    return end_x, end_y, end_z, projection_time, projected_inside_net, peak_z, peak_distance


def rotate_vector(values_x, values_y, values_z, vector_x, vector_y, vector_z):
    values_w = quaternion_w(values_x, values_y, values_z)
    dot_uv = values_x * vector_x + values_y * vector_y + values_z * vector_z
    dot_uu = values_x ** 2 + values_y ** 2 + values_z ** 2
    return (
        2.0 * dot_uv * values_x + (values_w ** 2 - dot_uu) * vector_x + 2.0 * values_w * (values_y * vector_z - values_z * vector_y),
        2.0 * dot_uv * values_y + (values_w ** 2 - dot_uu) * vector_y + 2.0 * values_w * (values_z * vector_x - values_x * vector_z),
        2.0 * dot_uv * values_z + (values_w ** 2 - dot_uu) * vector_z + 2.0 * values_w * (values_x * vector_y - values_y * vector_x),
    )


def inverse_rotate_vector(values_x, values_y, values_z, vector_x, vector_y, vector_z):
    return rotate_vector(-values_x, -values_y, -values_z, vector_x, vector_y, vector_z)


def closest_hitbox_local(local_x, local_y, local_z):
    return (
        np.clip(
            local_x,
            -DEFAULT_HITBOX_LENGTH / 2.0 + DEFAULT_HITBOX_OFFSET,
            DEFAULT_HITBOX_LENGTH / 2.0 + DEFAULT_HITBOX_OFFSET,
        ),
        np.clip(local_y, -DEFAULT_HITBOX_WIDTH / 2.0, DEFAULT_HITBOX_WIDTH / 2.0),
        np.clip(
            local_z,
            -DEFAULT_HITBOX_HEIGHT / 2.0 + DEFAULT_HITBOX_ELEVATION,
            DEFAULT_HITBOX_HEIGHT / 2.0 + DEFAULT_HITBOX_ELEVATION,
        ),
    )


def closest_hitbox_point(pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, target_x, target_y, target_z):
    local_x, local_y, local_z = inverse_rotate_vector(
        rot_x,
        rot_y,
        rot_z,
        target_x - pos_x,
        target_y - pos_y,
        target_z - pos_z,
    )
    contact_x, contact_y, contact_z = closest_hitbox_local(local_x, local_y, local_z)
    world_dx, world_dy, world_dz = rotate_vector(rot_x, rot_y, rot_z, contact_x, contact_y, contact_z)
    return pos_x + world_dx, pos_y + world_dy, pos_z + world_dz, contact_x, contact_y, contact_z


def estimate_car_time_to_point(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, boost, target_x, target_y, target_z):
    distance = np.sqrt((target_x - pos_x) ** 2 + (target_y - pos_y) ** 2 + (target_z - pos_z) ** 2)
    speed = np.clip(
        safe_speed(vel_x, vel_y, vel_z) + np.nan_to_num(boost, nan=0.0) * CAR_INTERCEPT_BOOST_SPEED_BONUS,
        CAR_INTERCEPT_MIN_SPEED,
        SUPERSONIC_THRESHOLD,
    )
    return safe_divide(distance, speed)


def trajectory_intercept_features(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, rot_x, rot_y, rot_z, boost,
                                  ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz, trajectory_time):
    min_time = np.full_like(ball_x, np.inf, dtype=np.float64)
    best_margin = np.full_like(ball_x, np.inf, dtype=np.float64)
    best_time = np.zeros_like(ball_x, dtype=np.float64)
    best_x = np.array(ball_x, dtype=np.float64, copy=True)
    best_y = np.array(ball_y, dtype=np.float64, copy=True)
    best_z = np.array(ball_z, dtype=np.float64, copy=True)
    best_distance = np.sqrt((ball_x - pos_x) ** 2 + (ball_y - pos_y) ** 2 + (ball_z - pos_z) ** 2)
    min_distance = np.full_like(ball_x, np.inf, dtype=np.float64)
    path_time = np.maximum(np.nan_to_num(trajectory_time, nan=0.0), 0.0)
    for sample_idx in range(TRAJECTORY_SAMPLE_COUNT + 1):
        sample_time = path_time * sample_idx / TRAJECTORY_SAMPLE_COUNT
        sample_x = ball_x + ball_vx * sample_time
        sample_y = ball_y + ball_vy * sample_time
        sample_z = np.maximum(ball_z + ball_vz * sample_time - 0.5 * GRAVITY * sample_time ** 2, 0.0)
        distance = np.sqrt((sample_x - pos_x) ** 2 + (sample_y - pos_y) ** 2 + (sample_z - pos_z) ** 2)
        hitbox_x, hitbox_y, hitbox_z, _, _, _ = closest_hitbox_point(
            pos_x,
            pos_y,
            pos_z,
            rot_x,
            rot_y,
            rot_z,
            sample_x,
            sample_y,
            sample_z,
        )
        hitbox_distance = np.sqrt((sample_x - hitbox_x) ** 2 + (sample_y - hitbox_y) ** 2 + (sample_z - hitbox_z) ** 2)
        travel_time = estimate_car_time_to_point(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, boost, sample_x, sample_y, sample_z)
        margin = travel_time - sample_time
        min_time = np.minimum(min_time, travel_time)
        min_distance = np.minimum(min_distance, hitbox_distance)
        update = margin < best_margin
        best_margin = np.where(update, margin, best_margin)
        best_time = np.where(update, sample_time, best_time)
        best_x = np.where(update, sample_x, best_x)
        best_y = np.where(update, sample_y, best_y)
        best_z = np.where(update, sample_z, best_z)
        best_distance = np.where(update, distance, best_distance)
    current_speed = np.maximum(safe_speed(vel_x, vel_y, vel_z), CAR_INTERCEPT_MIN_SPEED)
    required_speed = safe_divide(best_distance, best_time)
    boost_needed = np.clip((required_speed - current_speed) / CAR_INTERCEPT_BOOST_SPEED_BONUS, 0.0, 100.0)
    return {
        'min_time_to_ball_trajectory': min_time,
        'intercept_possible': best_margin <= INTERCEPT_MARGIN_SECONDS,
        'intercept_time_margin': best_margin,
        'intercept_x': best_x,
        'intercept_y': best_y,
        'intercept_z': best_z,
        'intercept_requires_aerial': best_z >= AERIAL_INTERCEPT_Z,
        'min_distance_to_trajectory': min_distance,
        'boost_needed_to_intercept': boost_needed,
        'boost_surplus_at_intercept': boost - boost_needed,
    }


def shot_path_intercept_features(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, rot_x, rot_y, rot_z, boost,
                                 ball_start_x, ball_start_y, ball_start_z, ball_end_x, ball_end_y, ball_end_z,
                                 trajectory_time):
    min_time = np.full_like(ball_start_x, np.inf, dtype=np.float64)
    best_margin = np.full_like(ball_start_x, np.inf, dtype=np.float64)
    best_x = np.array(ball_start_x, dtype=np.float64, copy=True)
    best_y = np.array(ball_start_y, dtype=np.float64, copy=True)
    best_z = np.array(ball_start_z, dtype=np.float64, copy=True)
    min_distance = np.full_like(ball_start_x, np.inf, dtype=np.float64)
    path_time = np.maximum(np.nan_to_num(trajectory_time, nan=0.0), 0.0)
    for sample_idx in range(TRAJECTORY_SAMPLE_COUNT + 1):
        fraction = sample_idx / TRAJECTORY_SAMPLE_COUNT
        sample_time = path_time * fraction
        sample_x = ball_start_x + (ball_end_x - ball_start_x) * fraction
        sample_y = ball_start_y + (ball_end_y - ball_start_y) * fraction
        sample_z = ball_start_z + (ball_end_z - ball_start_z) * fraction
        hitbox_x, hitbox_y, hitbox_z, _, _, _ = closest_hitbox_point(
            pos_x,
            pos_y,
            pos_z,
            rot_x,
            rot_y,
            rot_z,
            sample_x,
            sample_y,
            sample_z,
        )
        hitbox_distance = np.sqrt((sample_x - hitbox_x) ** 2 + (sample_y - hitbox_y) ** 2 + (sample_z - hitbox_z) ** 2)
        travel_time = estimate_car_time_to_point(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, boost, sample_x, sample_y, sample_z)
        margin = travel_time - sample_time
        min_time = np.minimum(min_time, travel_time)
        min_distance = np.minimum(min_distance, hitbox_distance)
        update = margin < best_margin
        best_margin = np.where(update, margin, best_margin)
        best_x = np.where(update, sample_x, best_x)
        best_y = np.where(update, sample_y, best_y)
        best_z = np.where(update, sample_z, best_z)
    return {
        'time_to_shot_path_estimate': min_time,
        'shot_path_intercept_possible': best_margin <= INTERCEPT_MARGIN_SECONDS,
        'shot_path_intercept_time_margin': best_margin,
        'shot_path_intercept_x': best_x,
        'shot_path_intercept_y': best_y,
        'shot_path_intercept_z': best_z,
        'min_distance_to_shot_path': min_distance,
    }


def norm_x(values, direction):
    return values * direction


def norm_y(values, direction):
    return values * direction


def quaternion_w(values_x, values_y, values_z):
    return np.sqrt(np.maximum(0.0, 1.0 - values_x ** 2 - values_y ** 2 - values_z ** 2))


def yaw_from_quaternion(values_x, values_y, values_z):
    values_w = quaternion_w(values_x, values_y, values_z)
    return normalize_angle(
        np.arctan2(
            2.0 * (values_w * values_z + values_x * values_y),
            1.0 - 2.0 * (values_y ** 2 + values_z ** 2),
        )
    )


def euler_from_quaternion(values_x, values_y, values_z):
    values_w = quaternion_w(values_x, values_y, values_z)
    roll = np.arctan2(
        2.0 * (values_w * values_x + values_y * values_z),
        1.0 - 2.0 * (values_x ** 2 + values_y ** 2),
    )
    sin_pitch = 2.0 * (values_w * values_y - values_z * values_x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = yaw_from_quaternion(values_x, values_y, values_z)
    return yaw, pitch, roll


def front_vector_from_quaternion(values_x, values_y, values_z):
    values_w = quaternion_w(values_x, values_y, values_z)
    return (
        1.0 - 2.0 * (values_y ** 2 + values_z ** 2),
        2.0 * (values_x * values_y + values_w * values_z),
        2.0 * (values_x * values_z - values_w * values_y),
    )


def angle_front_car_to_ground(values_x, values_y, values_z):
    front_x, front_y, front_z = front_vector_from_quaternion(values_x, values_y, values_z)
    front_horizontal = np.sqrt(front_x ** 2 + front_y ** 2)
    return np.arctan2(front_z, front_horizontal)


def player_slots(df):
    return sorted({
        column.rsplit('_', 1)[0]
        for column in df.columns
        if column.startswith(('blue_player_', 'orange_player_')) and column.endswith('_id')
    })


def role_value(df, slots, player_ids, field, categorical=False):
    if categorical:
        output = np.full(df.height, '', dtype=object)
    else:
        output = np.full(df.height, np.nan, dtype=np.float64)
    matched = np.zeros(df.height, dtype=bool)
    for slot in slots:
        id_column = f'{slot}_id'
        value_column = f'{slot}_{field}'
        if id_column not in df.columns or value_column not in df.columns:
            continue
        slot_ids = array_string(df, id_column)
        slot_match = (slot_ids == player_ids) & ~matched
        if not np.any(slot_match):
            continue
        values = array_string(df, value_column) if categorical else array_float(df, value_column)
        output[slot_match] = values[slot_match]
        matched |= slot_match
    return output


def transform_role_numeric(values, field, direction):
    values = values.astype(np.float64, copy=False)
    if field in ['pos_x', 'vel_x', 'ang_vel_x', 'pos_y', 'vel_y', 'ang_vel_y']:
        values = values * direction
    if field == 'boost':
        values = boost_units(values)
    return values


def opponent_role_value(df, role_number, field, shooter_team, direction=None):
    blue_slot = f'blue_player_{role_number}'
    orange_slot = f'orange_player_{role_number}'
    blue_col = f'{blue_slot}_{field}'
    orange_col = f'{orange_slot}_{field}'
    categorical = field in CATEGORICAL_ROLE_FIELDS
    if categorical:
        blue = array_string(df, blue_col) if blue_col in df.columns else np.full(df.height, '', dtype=object)
        orange = array_string(df, orange_col) if orange_col in df.columns else np.full(df.height, '', dtype=object)
        return np.where(shooter_team == 'blue', orange, np.where(shooter_team == 'orange', blue, ''))
    blue = array_float(df, blue_col) if blue_col in df.columns else np.full(df.height, np.nan, dtype=np.float64)
    orange = array_float(df, orange_col) if orange_col in df.columns else np.full(df.height, np.nan, dtype=np.float64)
    values = np.where(shooter_team == 'blue', orange, np.where(shooter_team == 'orange', blue, np.nan))
    return transform_role_numeric(values, field, direction) if direction is not None else values


def ordered_numeric_values(values_by_slot, order, role_number):
    if not values_by_slot:
        return np.array([], dtype=np.float64)
    values = np.column_stack(values_by_slot)
    ordered = np.take_along_axis(values, order, axis=1)
    return ordered[:, role_number - 1]


def ordered_string_values(values_by_slot, order, role_number):
    if not values_by_slot:
        return np.array([], dtype=str)
    values = np.column_stack(values_by_slot)
    ordered = np.take_along_axis(values, order, axis=1)
    return ordered[:, role_number - 1].astype(str)


def model_segment_values(df):
    team_size = array_float(df, 'team_size')
    segments = np.full(df.height, '', dtype=object)
    segments[(segments == '') & (team_size == 1)] = 'duels'
    segments[(segments == '') & (team_size == 2)] = 'doubles'
    segments[(segments == '') & (team_size == 3)] = 'standard'
    return segments


def selected_model_columns(columns):
    column_set = set(columns)
    selected = ['game_id', 'event_number', 'is_goal']
    selected.extend(column for column in model_non_player_columns() if column in column_set)
    selected.extend(f'shooter_{field}' for field in MODEL_SHOOTER if f'shooter_{field}' in column_set)
    for role_number in range(1, 4):
        selected.extend(
            f'opponent_{role_number}_{field}'
            for field in MODEL_OPPONENT
            if f'opponent_{role_number}_{field}' in column_set
        )
    return list(dict.fromkeys(selected))


def compact_model_df(model_df):
    float64_cols = [
        column
        for column, dtype in model_df.schema.items()
        if dtype == pl.Float64
    ]
    if not float64_cols:
        return model_df
    return model_df.with_columns([
        pl.col(column).cast(pl.Float32, strict=True).alias(column)
        for column in float64_cols
    ])


def build_segment_features(shots):
    slots = player_slots(shots)
    shooter_id = array_string(shots, 'event_player_1_id')
    shooter_team = np.char.lower(array_string(shots, 'event_player_1_team'))
    role_numeric_fields = model_role_numeric_fields()
    direction = np.where(shooter_team == 'orange', np.float32(-1.0), np.float32(1.0)).astype(np.float32, copy=False)
    ball_start_x = np.where(
        np.isfinite(array_float(shots, 'event_ball_pos_x')),
        array_float(shots, 'event_ball_pos_x'),
        array_float(shots, 'ball_pos_x'),
    )
    ball_start_y = np.where(
        np.isfinite(array_float(shots, 'event_ball_pos_y')),
        array_float(shots, 'event_ball_pos_y'),
        array_float(shots, 'ball_pos_y'),
    )
    ball_start_z = np.where(
        np.isfinite(array_float(shots, 'event_ball_pos_z')),
        array_float(shots, 'event_ball_pos_z'),
        array_float(shots, 'ball_pos_z'),
    )
    ball_x = ball_start_x * direction
    ball_y = ball_start_y * direction
    ball_z = ball_start_z
    ball_vx_raw = array_float(shots, 'ball_vel_x')
    ball_vy_raw = array_float(shots, 'ball_vel_y')
    ball_vz = array_float(shots, 'ball_vel_z')
    ball_vx = ball_vx_raw * direction
    ball_vy = ball_vy_raw * direction
    ball_ang_vx = array_float(shots, 'ball_ang_vel_x')
    ball_ang_vy = array_float(shots, 'ball_ang_vel_y')
    ball_ang_vz = array_float(shots, 'ball_ang_vel_z')
    (
        ball_end_x,
        ball_end_y,
        ball_end_z,
        ball_projection_time,
        ball_projected_inside_net,
        ball_arc_peak,
        ball_arc_peak_distance,
    ) = project_ball_end(ball_start_x, ball_start_y, ball_start_z, ball_vx_raw, ball_vy_raw, ball_vz, shooter_team)
    data = {
        'game_id': array_string(shots, 'game_id'),
        'event_number': array_float(shots, 'event_number').astype(np.int32),
        'is_goal': (array_string(shots, 'event_type') == 'goal').astype(np.int8),
    }

    ball_start_x_last = array_float(shots, '_ball_field_start_x_last')
    ball_start_y_last = array_float(shots, '_ball_field_start_y_last')
    ball_start_z_last = array_float(shots, '_ball_field_start_z_last')
    ball_distance_start_to_end = np.sqrt(
        (ball_end_x - ball_start_x) ** 2
        + (ball_end_y - ball_start_y) ** 2
        + (ball_end_z - ball_start_z) ** 2
    )
    ball_distance_start_last_to_start = np.sqrt(
        (ball_start_x - ball_start_x_last) ** 2
        + (ball_start_y - ball_start_y_last) ** 2
        + (ball_start_z - ball_start_z_last) ** 2
    )
    ball_speed_from_last = safe_divide(ball_distance_start_last_to_start, array_float(shots, 'seconds_from_last_event'))
    ball_speed_from_last_fallback = array_float(shots, 'ball_speed_from_last_event')
    ball_speed_from_last = np.where(np.isfinite(ball_speed_from_last), ball_speed_from_last, ball_speed_from_last_fallback)
    ball_event_computed = {
        'field_start_x': ball_start_x,
        'field_start_y': ball_start_y,
        'field_start_z': ball_start_z,
        'field_end_x': ball_end_x,
        'field_end_y': ball_end_y,
        'field_end_z': ball_end_z,
        'distance_start_to_end': ball_distance_start_to_end,
        'angle_start_to_end_xy': angle_xy_from_points(ball_start_x, ball_start_y, ball_end_x, ball_end_y),
        'angle_start_to_end_xz': angle_xz_from_points(ball_start_x, ball_start_y, ball_start_z, ball_end_x, ball_end_y, ball_end_z),
        'velocity_x': ball_vx_raw,
        'velocity_y': ball_vy_raw,
        'velocity_z': ball_vz,
        'angular_velocity_x': ball_ang_vx,
        'angular_velocity_y': ball_ang_vy,
        'angular_velocity_z': ball_ang_vz,
        'arc_peak': ball_arc_peak,
        'arc_peak_distance': ball_arc_peak_distance,
        'field_start_x_last': ball_start_x_last,
        'field_start_y_last': ball_start_y_last,
        'field_start_z_last': ball_start_z_last,
        'distance_start_last_to_start': ball_distance_start_last_to_start,
        'angle_start_last_to_start_xy': angle_xy_from_points(ball_start_x_last, ball_start_y_last, ball_start_x, ball_start_y),
        'angle_start_last_to_start_xz': angle_xz_from_points(ball_start_x_last, ball_start_y_last, ball_start_z_last, ball_start_x, ball_start_y, ball_start_z),
        'speed_from_last': ball_speed_from_last,
    }
    for field in BALL_EVENT_FRAME_CONTINUOUS:
        if field in {'contact_x', 'contact_y', 'contact_z'}:
            continue
        column = f'ball_{field}'
        data[column] = feature_float(
            shots,
            column,
            ball_event_computed[field],
            direction,
            prefer_existing=False,
        )
    data['ball_projected_inside_net'] = ball_projected_inside_net
    for field in CONTEXT_EVENT_FRAME_CONTINUOUS:
        data[f'context_{field}'] = array_float(shots, field)
    for field in CONTEXT_EVENT_FRAME_BOOLEAN:
        data[f'context_{field}'] = array_bool(shots, field)

    shooter_raw = {}
    for field in role_numeric_fields:
        raw_values = role_value(shots, slots, shooter_id, field, categorical=False)
        shooter_raw[field] = raw_values

    shooter_pos_x_raw = shooter_raw.get('pos_x', np.full(shots.height, np.nan))
    shooter_pos_y_raw = shooter_raw.get('pos_y', np.full(shots.height, np.nan))
    shooter_pos_z = shooter_raw.get('pos_z', np.full(shots.height, np.nan))
    shooter_vel_x_raw = shooter_raw.get('vel_x', np.full(shots.height, np.nan))
    shooter_vel_y_raw = shooter_raw.get('vel_y', np.full(shots.height, np.nan))
    shooter_vel_z = shooter_raw.get('vel_z', np.full(shots.height, np.nan))
    shooter_rot_x = shooter_raw.get('rot_x', np.full(shots.height, np.nan))
    shooter_rot_y = shooter_raw.get('rot_y', np.full(shots.height, np.nan))
    shooter_rot_z = shooter_raw.get('rot_z', np.full(shots.height, np.nan))
    shooter_yaw_raw, shooter_pitch, shooter_roll = euler_from_quaternion(shooter_rot_x, shooter_rot_y, shooter_rot_z)
    (
        shooter_contact_world_x,
        shooter_contact_world_y,
        shooter_contact_world_z,
        shooter_contact_x,
        shooter_contact_y,
        shooter_contact_z,
    ) = closest_hitbox_point(
        shooter_pos_x_raw,
        shooter_pos_y_raw,
        shooter_pos_z,
        shooter_rot_x,
        shooter_rot_y,
        shooter_rot_z,
        ball_start_x,
        ball_start_y,
        ball_start_z,
    )
    contact_to_ball_x = ball_start_x - shooter_contact_world_x
    contact_to_ball_y = ball_start_y - shooter_contact_world_y
    contact_to_ball_z = ball_start_z - shooter_contact_world_z
    contact_to_ball_distance = np.sqrt(contact_to_ball_x ** 2 + contact_to_ball_y ** 2 + contact_to_ball_z ** 2)
    ball_contact_x = ball_start_x - safe_divide(contact_to_ball_x, contact_to_ball_distance) * BALL_RADIUS
    ball_contact_y = ball_start_y - safe_divide(contact_to_ball_y, contact_to_ball_distance) * BALL_RADIUS
    ball_contact_z = ball_start_z - safe_divide(contact_to_ball_z, contact_to_ball_distance) * BALL_RADIUS
    data['ball_contact_x'] = feature_float(shots, 'ball_contact_x', ball_contact_x, direction, prefer_existing=False)
    data['ball_contact_y'] = feature_float(shots, 'ball_contact_y', ball_contact_y, direction, prefer_existing=False)
    data['ball_contact_z'] = feature_float(shots, 'ball_contact_z', ball_contact_z, direction, prefer_existing=False)
    shooter_event_computed = {
        'velocity_x': shooter_vel_x_raw,
        'velocity_y': shooter_vel_y_raw,
        'velocity_z': shooter_vel_z,
        'angular_velocity_x': shooter_raw.get('ang_vel_x', np.full(shots.height, np.nan)),
        'angular_velocity_y': shooter_raw.get('ang_vel_y', np.full(shots.height, np.nan)),
        'angular_velocity_z': shooter_raw.get('ang_vel_z', np.full(shots.height, np.nan)),
        'yaw': shooter_yaw_raw,
        'pitch': shooter_pitch,
        'roll': shooter_roll,
        'boost_amount': boost_units(shooter_raw.get('boost', np.full(shots.height, np.nan))),
        'contact_x': shooter_contact_world_x,
        'contact_y': shooter_contact_world_y,
        'contact_z': shooter_contact_world_z,
        'angle_contact_to_ball_start_xy': angle_xy_from_points(shooter_contact_world_x, shooter_contact_world_y, ball_start_x, ball_start_y),
        'angle_contact_to_ball_start_xz': angle_xz_from_points(shooter_contact_world_x, shooter_contact_world_y, shooter_contact_world_z, ball_start_x, ball_start_y, ball_start_z),
        'angle_contact_to_ball_end_xy': angle_xy_from_points(shooter_contact_world_x, shooter_contact_world_y, ball_end_x, ball_end_y),
        'angle_contact_to_ball_end_xz': angle_xz_from_points(shooter_contact_world_x, shooter_contact_world_y, shooter_contact_world_z, ball_end_x, ball_end_y, ball_end_z),
        'angle_front_car_to_ground': angle_front_car_to_ground(shooter_rot_x, shooter_rot_y, shooter_rot_z),
    }
    for field in SHOOTER_EVENT_FRAME_CONTINUOUS:
        column = f'shooter_{field}'
        data[column] = feature_float(
            shots,
            column,
            shooter_event_computed[field],
            direction,
            prefer_existing=False,
        )
    shooter_event_booleans = {
        'powersliding': np.nan_to_num(shooter_raw.get('handbrake', np.zeros(shots.height)), nan=0.0) != 0.0,
        'boosting': np.nan_to_num(shooter_raw.get('boost_active', np.zeros(shots.height)), nan=0.0) != 0.0,
        'flipping': (
            (np.nan_to_num(shooter_raw.get('flipped', np.zeros(shots.height)), nan=0.0) != 0.0)
            | (np.nan_to_num(shooter_raw.get('dodge_active', np.zeros(shots.height)), nan=0.0) != 0.0)
        ),
    }
    for field in SHOOTER_EVENT_FRAME_BOOLEAN:
        column = f'shooter_{field}'
        data[column] = shooter_event_booleans[field]

    opponent_raw = {field: [] for field in role_numeric_fields}
    opponent_distance_slots = []
    for slot_number in range(1, 4):
        raw_pos_x = opponent_role_value(shots, slot_number, 'pos_x', shooter_team)
        raw_pos_y = opponent_role_value(shots, slot_number, 'pos_y', shooter_team)
        raw_pos_z = opponent_role_value(shots, slot_number, 'pos_z', shooter_team)
        opponent_raw['pos_x'].append(raw_pos_x)
        opponent_raw['pos_y'].append(raw_pos_y)
        opponent_raw['pos_z'].append(raw_pos_z)
        for field in role_numeric_fields:
            if field in ['pos_x', 'pos_y', 'pos_z']:
                continue
            opponent_raw[field].append(opponent_role_value(shots, slot_number, field, shooter_team))
        pos_x = raw_pos_x * direction
        pos_y = raw_pos_y * direction
        pos_z = raw_pos_z
        opponent_distance_slots.append(np.sqrt((ball_x - pos_x) ** 2 + (ball_y - pos_y) ** 2 + (ball_z - pos_z) ** 2))

    distance_matrix = np.column_stack(opponent_distance_slots)
    sort_distances = np.where(np.isfinite(distance_matrix), distance_matrix, np.inf)
    opponent_order = np.argsort(sort_distances, axis=1)

    for role_number in range(1, 4):
        raw_pos_x = ordered_numeric_values(opponent_raw['pos_x'], opponent_order, role_number)
        raw_pos_y = ordered_numeric_values(opponent_raw['pos_y'], opponent_order, role_number)
        raw_pos_z = ordered_numeric_values(opponent_raw['pos_z'], opponent_order, role_number)
        raw_vel_x = ordered_numeric_values(opponent_raw['vel_x'], opponent_order, role_number)
        raw_vel_y = ordered_numeric_values(opponent_raw['vel_y'], opponent_order, role_number)
        raw_vel_z = ordered_numeric_values(opponent_raw['vel_z'], opponent_order, role_number)
        role_raw = {
            field: ordered_numeric_values(opponent_raw[field], opponent_order, role_number)
            for field in role_numeric_fields
        }
        raw_rot_x = role_raw['rot_x']
        raw_rot_y = role_raw['rot_y']
        raw_rot_z = role_raw['rot_z']
        raw_boost = boost_units(role_raw.get('boost', np.full(shots.height, np.nan)))
        active_opponent = np.isfinite(raw_pos_x) & np.isfinite(raw_pos_y) & np.isfinite(raw_pos_z)
        defensive_net_x = np.zeros(shots.height, dtype=np.float64)
        defensive_net_y = np.where(shooter_team == 'blue', BACK_NET_Y, -BACK_NET_Y)
        defensive_net_z = np.zeros(shots.height, dtype=np.float64)
        (
            proximity_start_x,
            proximity_start_y,
            proximity_start_z,
            _,
            _,
            _,
        ) = closest_hitbox_point(raw_pos_x, raw_pos_y, raw_pos_z, raw_rot_x, raw_rot_y, raw_rot_z, ball_start_x, ball_start_y, ball_start_z)
        (
            proximity_end_x,
            proximity_end_y,
            proximity_end_z,
            _,
            _,
            _,
        ) = closest_hitbox_point(raw_pos_x, raw_pos_y, raw_pos_z, raw_rot_x, raw_rot_y, raw_rot_z, ball_end_x, ball_end_y, ball_end_z)
        (
            proximity_net_x,
            proximity_net_y,
            proximity_net_z,
            _,
            _,
            _,
        ) = closest_hitbox_point(raw_pos_x, raw_pos_y, raw_pos_z, raw_rot_x, raw_rot_y, raw_rot_z, defensive_net_x, defensive_net_y, defensive_net_z)
        time_to_ball_start = estimate_car_time_to_point(
            raw_pos_x,
            raw_pos_y,
            raw_pos_z,
            raw_vel_x,
            raw_vel_y,
            raw_vel_z,
            raw_boost,
            ball_start_x,
            ball_start_y,
            ball_start_z,
        )
        time_to_ball_end = estimate_car_time_to_point(
            raw_pos_x,
            raw_pos_y,
            raw_pos_z,
            raw_vel_x,
            raw_vel_y,
            raw_vel_z,
            raw_boost,
            ball_end_x,
            ball_end_y,
            ball_end_z,
        )
        time_to_defensive_net = estimate_car_time_to_point(
            raw_pos_x,
            raw_pos_y,
            raw_pos_z,
            raw_vel_x,
            raw_vel_y,
            raw_vel_z,
            raw_boost,
            defensive_net_x,
            defensive_net_y,
            defensive_net_z,
        )
        intercept = trajectory_intercept_features(
            raw_pos_x,
            raw_pos_y,
            raw_pos_z,
            raw_vel_x,
            raw_vel_y,
            raw_vel_z,
            raw_rot_x,
            raw_rot_y,
            raw_rot_z,
            raw_boost,
            ball_start_x,
            ball_start_y,
            ball_start_z,
            ball_vx_raw,
            ball_vy_raw,
            ball_vz,
            ball_projection_time,
        )
        shot_path_intercept = shot_path_intercept_features(
            raw_pos_x,
            raw_pos_y,
            raw_pos_z,
            raw_vel_x,
            raw_vel_y,
            raw_vel_z,
            raw_rot_x,
            raw_rot_y,
            raw_rot_z,
            raw_boost,
            ball_start_x,
            ball_start_y,
            ball_start_z,
            ball_end_x,
            ball_end_y,
            ball_end_z,
            ball_projection_time,
        )
        opponent_yaw, opponent_pitch, opponent_roll = euler_from_quaternion(raw_rot_x, raw_rot_y, raw_rot_z)
        opponent_event_computed = {
            'position_x': np.where(active_opponent, raw_pos_x, MISSING_OPPONENT_POSITION),
            'position_y': np.where(active_opponent, raw_pos_y, MISSING_OPPONENT_POSITION),
            'position_z': np.where(active_opponent, raw_pos_z, MISSING_OPPONENT_POSITION),
            'velocity_x': np.where(active_opponent, raw_vel_x, 0.0),
            'velocity_y': np.where(active_opponent, raw_vel_y, 0.0),
            'velocity_z': np.where(active_opponent, raw_vel_z, 0.0),
            'angular_velocity_x': np.where(active_opponent, role_raw.get('ang_vel_x', np.full(shots.height, np.nan)), 0.0),
            'angular_velocity_y': np.where(active_opponent, role_raw.get('ang_vel_y', np.full(shots.height, np.nan)), 0.0),
            'angular_velocity_z': np.where(active_opponent, role_raw.get('ang_vel_z', np.full(shots.height, np.nan)), 0.0),
            'yaw': np.where(active_opponent, opponent_yaw, 0.0),
            'pitch': np.where(active_opponent, opponent_pitch, 0.0),
            'roll': np.where(active_opponent, opponent_roll, 0.0),
            'boost_amount': np.where(active_opponent, raw_boost, 0.0),
            'proximity_to_ball_start_x': np.where(active_opponent, proximity_start_x, MISSING_OPPONENT_POSITION),
            'proximity_to_ball_start_y': np.where(active_opponent, proximity_start_y, MISSING_OPPONENT_POSITION),
            'proximity_to_ball_start_z': np.where(active_opponent, proximity_start_z, MISSING_OPPONENT_POSITION),
            'proximity_to_ball_end_x': np.where(active_opponent, proximity_end_x, MISSING_OPPONENT_POSITION),
            'proximity_to_ball_end_y': np.where(active_opponent, proximity_end_y, MISSING_OPPONENT_POSITION),
            'proximity_to_ball_end_z': np.where(active_opponent, proximity_end_z, MISSING_OPPONENT_POSITION),
            'proximity_to_defensive_net_location_x': np.where(active_opponent, proximity_net_x, MISSING_OPPONENT_POSITION),
            'proximity_to_defensive_net_location_y': np.where(active_opponent, proximity_net_y, MISSING_OPPONENT_POSITION),
            'proximity_to_defensive_net_location_z': np.where(active_opponent, proximity_net_z, MISSING_OPPONENT_POSITION),
            'angle_proximity_to_ball_start_xy': np.where(active_opponent, angle_xy_from_points(proximity_start_x, proximity_start_y, ball_start_x, ball_start_y), MISSING_OPPONENT_POSITION),
            'angle_proximity_to_ball_start_xz': np.where(active_opponent, angle_xz_from_points(proximity_start_x, proximity_start_y, proximity_start_z, ball_start_x, ball_start_y, ball_start_z), MISSING_OPPONENT_POSITION),
            'angle_proximity_to_ball_end_xy': np.where(active_opponent, angle_xy_from_points(proximity_end_x, proximity_end_y, ball_end_x, ball_end_y), MISSING_OPPONENT_POSITION),
            'angle_proximity_to_ball_end_xz': np.where(active_opponent, angle_xz_from_points(proximity_end_x, proximity_end_y, proximity_end_z, ball_end_x, ball_end_y, ball_end_z), MISSING_OPPONENT_POSITION),
            'angle_proximity_to_defensive_net_to_defensive_net_xy': np.where(active_opponent, angle_xy_from_points(proximity_net_x, proximity_net_y, defensive_net_x, defensive_net_y), MISSING_OPPONENT_POSITION),
            'angle_proximity_to_defensive_net_to_defensive_net_xz': np.where(active_opponent, angle_xz_from_points(proximity_net_x, proximity_net_y, proximity_net_z, defensive_net_x, defensive_net_y, defensive_net_z), MISSING_OPPONENT_POSITION),
            'time_to_ball_end': np.where(active_opponent, time_to_ball_end, MISSING_OPPONENT_POSITION),
            'min_time_to_ball_trajectory': np.where(active_opponent, intercept['min_time_to_ball_trajectory'], MISSING_OPPONENT_POSITION),
            'intercept_time_margin': np.where(active_opponent, intercept['intercept_time_margin'], MISSING_OPPONENT_POSITION),
            'intercept_x': np.where(active_opponent, intercept['intercept_x'], MISSING_OPPONENT_POSITION),
            'intercept_y': np.where(active_opponent, intercept['intercept_y'], MISSING_OPPONENT_POSITION),
            'intercept_z': np.where(active_opponent, intercept['intercept_z'], MISSING_OPPONENT_POSITION),
            'min_distance_to_trajectory': np.where(active_opponent, intercept['min_distance_to_trajectory'], MISSING_OPPONENT_POSITION),
            'boost_needed_to_intercept': np.where(active_opponent, intercept['boost_needed_to_intercept'], MISSING_OPPONENT_POSITION),
            'boost_surplus_at_intercept': np.where(active_opponent, intercept['boost_surplus_at_intercept'], MISSING_OPPONENT_POSITION),
            'time_to_ball_start_estimate': np.where(active_opponent, time_to_ball_start, MISSING_OPPONENT_POSITION),
            'time_to_ball_end_estimate': np.where(active_opponent, time_to_ball_end, MISSING_OPPONENT_POSITION),
            'time_to_defensive_net_estimate': np.where(active_opponent, time_to_defensive_net, MISSING_OPPONENT_POSITION),
            'time_to_shot_path_estimate': np.where(active_opponent, shot_path_intercept['time_to_shot_path_estimate'], MISSING_OPPONENT_POSITION),
            'shot_path_intercept_time_margin': np.where(active_opponent, shot_path_intercept['shot_path_intercept_time_margin'], MISSING_OPPONENT_POSITION),
            'shot_path_intercept_x': np.where(active_opponent, shot_path_intercept['shot_path_intercept_x'], MISSING_OPPONENT_POSITION),
            'shot_path_intercept_y': np.where(active_opponent, shot_path_intercept['shot_path_intercept_y'], MISSING_OPPONENT_POSITION),
            'shot_path_intercept_z': np.where(active_opponent, shot_path_intercept['shot_path_intercept_z'], MISSING_OPPONENT_POSITION),
            'min_distance_to_shot_path': np.where(active_opponent, shot_path_intercept['min_distance_to_shot_path'], MISSING_OPPONENT_POSITION),
            'angle_front_car_to_ground': np.where(active_opponent, angle_front_car_to_ground(raw_rot_x, raw_rot_y, raw_rot_z), 0.0),
        }
        for field in OPPONENT_EVENT_FRAME_CONTINUOUS:
            column = f'opponent_{role_number}_{field}'
            data[column] = feature_float(
                shots,
                column,
                opponent_event_computed[field],
                direction,
                prefer_existing=False,
            )
        opponent_event_booleans = {
            'on_field': active_opponent,
            'powersliding': np.nan_to_num(role_raw.get('handbrake', np.zeros(shots.height)), nan=0.0) != 0.0,
            'boosting': np.nan_to_num(role_raw.get('boost_active', np.zeros(shots.height)), nan=0.0) != 0.0,
            'flipping': (
                (np.nan_to_num(role_raw.get('flipped', np.zeros(shots.height)), nan=0.0) != 0.0)
                | (np.nan_to_num(role_raw.get('dodge_active', np.zeros(shots.height)), nan=0.0) != 0.0)
            ),
            'intercept_possible': active_opponent & intercept['intercept_possible'],
            'intercept_requires_aerial': active_opponent & intercept['intercept_requires_aerial'],
            'shot_path_intercept_possible': active_opponent & shot_path_intercept['shot_path_intercept_possible'],
        }
        for field in OPPONENT_EVENT_FRAME_BOOLEAN:
            column = f'opponent_{role_number}_{field}'
            data[column] = opponent_event_booleans[field]

    model_df = pl.DataFrame(data)
    return compact_model_df(model_df.select(selected_model_columns(model_df.columns)))


def to_csr(matrix):
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(matrix)


def make_sparse_matrix(model_df, numeric_cols, categorical_cols, preprocessor=None):
    parts = []
    if numeric_cols:
        numeric = model_df.select([
            pl.col(column).cast(pl.Float32, strict=True).alias(column)
            for column in numeric_cols
        ]).to_numpy()
        numeric[~np.isfinite(numeric)] = np.nan
        if preprocessor is None:
            numeric_imputer = SimpleImputer(strategy='median')
            numeric = numeric_imputer.fit_transform(numeric)
        else:
            numeric_imputer = preprocessor['numeric_imputer']
            numeric = numeric_imputer.transform(numeric)
        numeric = np.asarray(numeric, dtype=np.float32)
        parts.append(sparse.csr_matrix(numeric, dtype=np.float32))
    else:
        numeric_imputer = None if preprocessor is None else preprocessor['numeric_imputer']

    if categorical_cols:
        categorical = model_df.select([
            pl.col(column).cast(pl.Utf8, strict=True).fill_null('missing').alias(column)
            for column in categorical_cols
        ]).to_numpy()
        if preprocessor is None:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            categorical_sparse = encoder.fit_transform(categorical)
        else:
            encoder = preprocessor['encoder']
            categorical_sparse = encoder.transform(categorical)
        parts.append(to_csr(categorical_sparse))
    else:
        encoder = None if preprocessor is None else preprocessor['encoder']

    matrix = sparse.hstack(parts, format='csr', dtype=np.float32) if parts else sparse.csr_matrix((model_df.height, 0), dtype=np.float32)
    if preprocessor is None:
        return matrix, {'numeric_imputer': numeric_imputer, 'encoder': encoder}
    return matrix



def fit_isotonic_calibrator(model, train_matrix, y_train, train_groups=None, cv_splits=3):
    """Fit isotonic calibration from replay-grouped OOF predictions when possible.

    Falls back to in-sample training predictions only when grouped OOF calibration is not
    feasible. The returned metadata makes that fallback explicit in metrics/artifacts.
    """
    y_train = np.asarray(y_train, dtype=np.int8)
    if len(np.unique(y_train)) < 2:
        return None, {'calibration_status': 'skipped_single_class'}

    train_pred = None
    method = 'in_sample'
    if train_groups is not None:
        train_groups = np.asarray(train_groups)
        n_group_splits = min(int(cv_splits), len(np.unique(train_groups)))
        if n_group_splits >= 2:
            oof_pred = np.full(len(y_train), np.nan, dtype=np.float64)
            fold_splits = group_cv_folds(y_train, train_groups, n_group_splits)
            usable = True
            for fold_train_idx, fold_valid_idx in fold_splits:
                if len(np.unique(y_train[fold_train_idx])) < 2:
                    usable = False
                    break
                fold_model = train_booster(
                    model['params'],
                    train_matrix[fold_train_idx],
                    y_train[fold_train_idx],
                    num_boost_round=model['num_boost_round'],
                )
                oof_pred[fold_valid_idx] = predict_scores(
                    fold_model,
                    train_matrix[fold_valid_idx],
                )
            if usable and np.isfinite(oof_pred).all():
                train_pred = oof_pred
                method = f'group_oof_{n_group_splits}_fold'

    if train_pred is None:
        train_pred = predict_scores(model['booster'], train_matrix)

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    calibrator.fit(train_pred, y_train)
    return calibrator, {
        'calibration_status': 'fit',
        'calibration_method': method,
        'calibration_n': int(len(y_train)),
        'calibration_pred_min': float(np.min(train_pred)),
        'calibration_pred_max': float(np.max(train_pred)),
    }


def resolve_xgb_space(space):
    if not space:
        return DEFAULT_XGB_SPACE
    return dict(space)


def resolve_tree_device(gpu):
    return 'cuda' if gpu else 'cpu'


def split_boost_round(params):
    model_params = dict(params)
    num_boost_round = int(model_params.pop('num_boost_round', model_params.pop('n_estimators', 300)))
    return model_params, num_boost_round


def make_dmatrix(matrix, labels=None):
    return xgb.DMatrix(matrix, label=labels, nthread=1)


def predict_scores(booster, matrix):
    return booster.predict(make_dmatrix(matrix))


def suggest_xgb_params(trial, space):
    params = {}
    for name, spec in resolve_xgb_space(space).items():
        if isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
            continue
        if not isinstance(spec, tuple) or len(spec) < 3:
            raise ValueError(f'Unsupported Optuna search spec for {name}: {spec}')
        kind = spec[0]
        low = spec[1]
        high = spec[2]
        if kind == 'int':
            params[name] = trial.suggest_int(name, low, high)
        elif kind == 'float':
            log = bool(spec[3]) if len(spec) > 3 else False
            params[name] = trial.suggest_float(name, low, high, log=log)
        elif kind == 'categorical':
            params[name] = trial.suggest_categorical(name, list(spec[1]))
        else:
            raise ValueError(f'Unsupported Optuna search kind for {name}: {kind}')
    return params


def train_booster(params, train_matrix, y_train, num_boost_round, valid_matrix=None, y_valid=None, stop=None):
    dtrain = make_dmatrix(train_matrix, y_train)
    evals = []
    early_stopping_rounds = None
    if valid_matrix is not None and y_valid is not None:
        dvalid = make_dmatrix(valid_matrix, y_valid)
        evals.append((dvalid, 'valid'))
        if stop is not None:
            early_stopping_rounds = int(stop)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster


def group_cv_folds(y, groups, folds):
    splitter = GroupKFold(n_splits=folds)
    return list(splitter.split(np.arange(len(y)), y, groups))


def fit_optuna_xgb(base_params, search_space, train_matrix, y_train, train_groups, folds, iters, seed, stop, jobs):
    if optuna is None:
        raise ImportError('optuna is required when iters > 0')

    fold_splits = group_cv_folds(y_train, train_groups, folds)
    startup_trials = min(max(32, folds * 8), max(int(iters) // 5, 32))
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=startup_trials,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(50, startup_trials),
        n_warmup_steps=1,
        interval_steps=1,
    )
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )
    trial_numbers = []
    trial_values = []
    trial_states = []
    trial_completed_folds = []
    trial_mean_rounds = []
    study_jobs = max(1, int(jobs or 1))

    def objective(trial):
        trial_params = suggest_xgb_params(trial, search_space)
        fold_params, num_boost_round = split_boost_round({**base_params, **trial_params})
        fold_losses = []
        fold_best_rounds = []
        for fold_idx, (fold_train_idx, fold_valid_idx) in enumerate(fold_splits, start=1):
            booster = train_booster(
                fold_params,
                train_matrix[fold_train_idx],
                y_train[fold_train_idx],
                num_boost_round=num_boost_round,
                valid_matrix=train_matrix[fold_valid_idx],
                y_valid=y_train[fold_valid_idx],
                stop=stop,
            )
            fold_pred = predict_scores(booster, train_matrix[fold_valid_idx])
            fold_loss = float(log_loss(y_train[fold_valid_idx], fold_pred, labels=[0, 1]))
            best_iteration = getattr(booster, 'best_iteration', None)
            if best_iteration is None:
                best_iteration = num_boost_round - 1
            fold_losses.append(fold_loss)
            fold_best_rounds.append(int(best_iteration) + 1 if best_iteration is not None else None)
            trial.report(fold_loss, step=fold_idx)
            if trial.should_prune():
                mean_loss = float(np.mean(fold_losses))
                trial_numbers.append(trial.number)
                trial_values.append(mean_loss)
                trial_states.append('PRUNED')
                trial_completed_folds.append(fold_idx)
                trial_mean_rounds.append(None)
                raise optuna.TrialPruned()
        mean_loss = float(np.mean(fold_losses))
        mean_rounds = int(np.round(np.mean([rounds for rounds in fold_best_rounds if rounds is not None])))
        trial.set_user_attr('mean_rounds', mean_rounds)
        trial_numbers.append(trial.number)
        trial_values.append(mean_loss)
        trial_states.append('COMPLETE')
        trial_completed_folds.append(len(fold_splits))
        trial_mean_rounds.append(mean_rounds)
        return mean_loss

    study.optimize(
        objective,
        n_trials=int(iters),
        show_progress_bar=False,
        gc_after_trial=False,
        n_jobs=study_jobs,
    )
    best_params = dict(study.best_trial.params)
    if 'num_boost_round' in best_params or 'n_estimators' in best_params:
        best_params['num_boost_round'] = int(
            study.best_trial.user_attrs.get(
                'mean_rounds',
                best_params.get('num_boost_round', best_params.get('n_estimators', 300)),
            )
        )
        best_params.pop('n_estimators', None)
    cv_results = (
        pl.DataFrame({
            'number': trial_numbers,
            'value': trial_values,
            'state': trial_states,
            'completed_folds': trial_completed_folds,
            'mean_rounds': trial_mean_rounds,
        })
        if trial_numbers
        else pl.DataFrame([{'status': 'no_completed_trials'}])
    )
    return best_params, cv_results, study.best_value

def print_sparse_summary(segment, label, matrix):
    _ = segment, label, matrix


def feature_columns(model_df):
    skip_cols = {'game_id', 'event_number', 'is_goal'}
    explicit_categorical = {
        'previous_event_type',
        'shooter_car_id',
        'shooter_rank',
        'shooter_pro_player',
    }
    categorical_cols = []
    numeric_cols = []
    for column in model_df.columns:
        if column in skip_cols:
            continue
        dtype = model_df.schema[column]
        if (column.endswith('_id') and column != 'shooter_car_id') or column.endswith('_rank_tier'):
            continue
        if column in explicit_categorical or column.endswith('_rank') or column.endswith('_pro_player'):
            categorical_cols.append(column)
        elif dtype in [pl.String, pl.Categorical, pl.Enum, pl.Utf8]:
            categorical_cols.append(column)
        else:
            numeric_cols.append(column)
    return numeric_cols, categorical_cols


def observed_numeric_columns(model_df, numeric_cols):
    if not numeric_cols:
        return []
    observed = model_df.select([
        pl.col(column)
        .cast(pl.Float64, strict=True)
        .is_finite()
        .fill_null(False)
        .any()
        .alias(column)
        for column in numeric_cols
    ]).row(0, named=True)
    return [column for column in numeric_cols if observed.get(column, False)]


def feature_names(numeric_cols, categorical_cols, preprocessor):
    names = list(numeric_cols)
    encoder = preprocessor.get('encoder')
    if encoder is not None and categorical_cols:
        names.extend(encoder.get_feature_names_out(categorical_cols).tolist())
    return names


def shap_summary_data(booster, matrix, names, max_rows=5000, seed=42):
    if booster is None or matrix is None or not names:
        return pl.DataFrame({'feature': [], 'mean_abs_shap': []}), None, None, []
    row_count = matrix.shape[0]
    if row_count <= 0:
        return pl.DataFrame({'feature': [], 'mean_abs_shap': []}), None, None, []
    shap_matrix = matrix
    if row_count > max_rows:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(row_count, size=int(max_rows), replace=False))
        shap_matrix = matrix[sample_idx]
    contributions = booster.predict(make_dmatrix(shap_matrix), pred_contribs=True)
    if contributions.ndim != 2 or contributions.shape[1] < 2:
        return pl.DataFrame({'feature': [], 'mean_abs_shap': []}), None, None, []
    shap_values = contributions[:, :-1]
    values = np.mean(np.abs(shap_values), axis=0)
    feature_count = min(len(names), values.shape[0])
    if feature_count == 0:
        return pl.DataFrame({'feature': [], 'mean_abs_shap': []}), None, None, []
    shap_values = shap_values[:, :feature_count]
    feature_values = shap_matrix[:, :feature_count]
    if sparse.issparse(feature_values):
        feature_values = feature_values.toarray()
    feature_values = np.asarray(feature_values, dtype=np.float32)
    feature_names = names[:feature_count]
    importance = pl.DataFrame({
        'feature': names[:feature_count],
        'mean_abs_shap': values[:feature_count],
    }).sort('mean_abs_shap', descending=True)
    return importance, shap_values, feature_values, feature_names


def shap_feature_importance(booster, matrix, names, max_rows=5000, seed=42):
    importance, _, _, _ = shap_summary_data(booster, matrix, names, max_rows=max_rows, seed=seed)
    return importance


def normalized_feature_colors(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, 0.5, dtype=np.float64)
    low, high = np.nanpercentile(values[finite], [5, 95])
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) <= 1e-12:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def plotting_modules():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    shap_cmap = LinearSegmentedColormap.from_list(
        'analyzerl_shap',
        ['#1592e6', '#7b3fc6', '#ff0051'],
    )
    return plt, shap_cmap


def plot_feature_importance(model, matrix, names, path, csv_path=None, seed=42):
    if isinstance(model, dict):
        model = model.get('booster')
    if model is None:
        return
    importance, shap_values, feature_values, feature_names = shap_summary_data(model, matrix, names, seed=seed)
    if csv_path is not None:
        csv_safe_frame(importance).write_csv(csv_path)
    if importance.is_empty() or shap_values is None or feature_values is None:
        return
    top_features = importance.head(20).get_column('feature').to_list()
    feature_index = {feature: idx for idx, feature in enumerate(feature_names)}
    top_indices = [feature_index[feature] for feature in top_features if feature in feature_index]
    if not top_indices:
        return

    plt, shap_cmap = plotting_modules()
    rng = np.random.default_rng(seed)
    labels = [feature_names[idx] for idx in top_indices][::-1]
    plt.figure(figsize=(11, max(7, 0.34 * len(labels) + 2.0)))
    ax = plt.gca()
    max_abs = 0.0
    for y_pos, idx in enumerate(top_indices[::-1]):
        shap_column = np.asarray(shap_values[:, idx], dtype=np.float64)
        value_column = np.asarray(feature_values[:, idx], dtype=np.float64)
        finite = np.isfinite(shap_column)
        if not np.any(finite):
            continue
        shap_column = shap_column[finite]
        value_column = value_column[finite]
        max_abs = max(max_abs, float(np.nanmax(np.abs(shap_column))))
        order = np.argsort(np.abs(shap_column))
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.linspace(-1.0, 1.0, len(order)) if len(order) > 1 else 0.0
        jitter_width = 0.34 * (1.0 - np.minimum(np.abs(ranks), 1.0) ** 0.7)
        jitter = rng.uniform(-jitter_width, jitter_width)
        colors = normalized_feature_colors(value_column)
        ax.scatter(
            shap_column,
            np.full(shap_column.shape, y_pos, dtype=np.float64) + jitter,
            c=colors,
            cmap=shap_cmap,
            vmin=0.0,
            vmax=1.0,
            s=10,
            alpha=0.88,
            linewidths=0,
            rasterized=True,
        )
    ax.axvline(0.0, color='#888888', linewidth=1.0)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    if max_abs > 0:
        ax.set_xlim(-max_abs * 1.08, max_abs * 1.08)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.35)
    ax.set_xlabel('SHAP value (impact on model output)')
    ax.set_title('AnalyzeRL xG SHAP Feature Importance')
    sm = plt.cm.ScalarMappable(cmap=shap_cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label('Feature value')
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(['Low', 'High'])
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()


def plot_roc_auc(y_true, y_pred, segment, path):
    plt, _ = plotting_modules()
    auc_value = float(roc_auc_score(y_true, y_pred))
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    plt.plot(false_positive_rate, true_positive_rate, color='#2f6f73', label=f'{segment} (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='#777777')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AnalyzeRL xG ROC-AUC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()


def plot_calibration(y_true, y_pred, segment, path):
    plt, _ = plotting_modules()
    clipped = np.clip(y_pred, 0, 1)
    bins = np.linspace(0, 1, 21)
    bin_ids = np.digitize(clipped, bins, right=True)
    predicted = []
    observed = []
    for bin_id in range(1, len(bins) + 1):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        predicted.append(float(np.mean(clipped[mask])))
        observed.append(float(np.mean(y_true[mask])))
    plt.figure(figsize=(8, 7))
    plt.plot(predicted, observed, 's-', color='#2f6f73', label=segment)
    plt.plot([0, 1], [0, 1], linestyle='--', color='#777777', label='Perfect calibration')
    plt.xlabel('Predicted Probability (mean)')
    plt.ylabel('Fraction of positives')
    plt.title('AnalyzeRL xG Calibration')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()


def csv_safe_frame(frame):
    expressions = []
    for column, dtype in frame.schema.items():
        if dtype in [pl.List, pl.Struct, pl.Object] or str(dtype).startswith(('List', 'Struct', 'Object')):
            expressions.append(
                pl.col(column)
                .map_elements(lambda value: json.dumps(value) if value is not None else None, return_dtype=pl.Utf8)
                .alias(column)
            )
        else:
            expressions.append(pl.col(column))
    return frame.select(expressions)


def train_segment(
    segment: str,
    segment_shots: pl.DataFrame,
    model_folder: str | Path,
    return_scored: bool = False,
    nested: bool = True,
    folds: int = 3,
    iters: int = 0,
    jobs: int | None = None,
    params: Mapping[str, Any] | None = None,
    space: Mapping[str, Any] | None = None,
    seed: int | None = 38,
    stop: int = 50,
    gpu: bool = False,
):
    segment_folder = os.path.join(model_folder, segment) if nested else model_folder
    os.makedirs(segment_folder, exist_ok=True)
    model_df = build_segment_features(segment_shots)
    del segment_shots
    y = model_df.get_column('is_goal').cast(pl.Int8, strict=True).to_numpy()
    ids = model_df.select(['game_id', 'event_number']) if return_scored else None
    metrics = {
        'segment': segment,
        'n_events': int(model_df.height),
        'n_goals': int(y.sum()),
    }
    numeric_cols, categorical_cols = feature_columns(model_df)
    numeric_cols = observed_numeric_columns(model_df, numeric_cols)
    if len(np.unique(y)) < 2 or model_df.height < 20:
        metrics.update({'status': 'skipped_insufficient_outcomes'})
        csv_safe_frame(pl.DataFrame([metrics])).write_csv(os.path.join(segment_folder, f'cv_results_{segment}.csv'))
        return metrics, None

    model_params = dict(params or {})
    if 'tree_method' not in model_params:
        model_params['tree_method'] = 'hist'
    if 'objective' not in model_params:
        model_params['objective'] = 'binary:logistic'
    if 'eval_metric' not in model_params:
        model_params['eval_metric'] = 'logloss'
    if 'device' not in model_params:
        model_params['device'] = resolve_tree_device(gpu)
    if seed is not None and 'random_state' not in model_params:
        model_params['random_state'] = seed
    if jobs is not None and 'n_jobs' not in model_params:
        model_params['n_jobs'] = jobs
    if jobs is not None and 'nthread' not in model_params:
        model_params['nthread'] = jobs
    row_numbers = np.arange(model_df.height)
    groups = array_string(model_df, 'game_id')
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        metrics.update({'status': 'skipped_insufficient_replay_groups'})
        csv_safe_frame(pl.DataFrame([metrics])).write_csv(os.path.join(segment_folder, f'cv_results_{segment}.csv'))
        return metrics, None
    train_idx = None
    test_idx = None
    fallback_split = None
    splitter = GroupShuffleSplit(n_splits=25, test_size=0.2, random_state=42)
    for candidate_train_idx, candidate_test_idx in splitter.split(row_numbers, y, groups):
        if fallback_split is None and len(np.unique(y[candidate_train_idx])) >= 2:
            fallback_split = (candidate_train_idx, candidate_test_idx)
        if len(np.unique(y[candidate_train_idx])) >= 2 and len(np.unique(y[candidate_test_idx])) >= 2:
            train_idx, test_idx = candidate_train_idx, candidate_test_idx
            break
    if train_idx is None and fallback_split is not None:
        train_idx, test_idx = fallback_split
    if train_idx is None:
        metrics.update({'status': 'skipped_insufficient_train_outcomes'})
        csv_safe_frame(pl.DataFrame([metrics])).write_csv(os.path.join(segment_folder, f'cv_results_{segment}.csv'))
        return metrics, None
    x_train_df = model_df[train_idx]
    x_test_df = model_df[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    train_groups = groups[train_idx]
    train_matrix, preprocessor = make_sparse_matrix(x_train_df, numeric_cols, categorical_cols)
    test_matrix = make_sparse_matrix(x_test_df, numeric_cols, categorical_cols, preprocessor=preprocessor)
    del x_train_df, x_test_df
    if not return_scored:
        del model_df
    gc.collect()
    print_sparse_summary(segment, 'train', train_matrix)
    print_sparse_summary(segment, 'holdout', test_matrix)
    cv_splits = min(int(folds), len(np.unique(train_groups)))
    can_group_cv = cv_splits >= 2 and len(np.unique(y_train)) >= 2
    if can_group_cv:
        cv_probe = GroupKFold(n_splits=cv_splits)
        for cv_train_idx, cv_test_idx in cv_probe.split(np.arange(len(y_train)), y_train, train_groups):
            if len(np.unique(y_train[cv_train_idx])) < 2 or len(np.unique(y_train[cv_test_idx])) < 2:
                can_group_cv = False
                break
    if not can_group_cv:
        fit_params, num_boost_round = split_boost_round(model_params)
        booster = train_booster(fit_params, train_matrix, y_train, num_boost_round=num_boost_round)
        model = {'booster': booster, 'params': fit_params, 'num_boost_round': num_boost_round}
        cv_results = pl.DataFrame([{
            'status': 'fit_without_grouped_cv',
            'reason': 'insufficient outcome-balanced replay groups',
        }])
        metrics['status'] = 'fit_without_grouped_cv'
    elif int(iters) > 0:
        search_space = resolve_xgb_space(space)
        metrics['search_iters'] = int(iters)
        metrics['search_space'] = sorted(search_space.keys())
        best_params, cv_results, best_value = fit_optuna_xgb(
            model_params,
            search_space,
            train_matrix,
            y_train,
            train_groups,
            cv_splits,
            int(iters),
            seed,
            int(stop),
            jobs,
        )
        metrics['best_params'] = best_params
        metrics['best_cv_log_loss'] = float(best_value)
        final_fit_idx = None
        final_valid_idx = None
        final_splitter = GroupShuffleSplit(n_splits=25, test_size=0.15, random_state=seed)
        for candidate_fit_idx, candidate_valid_idx in final_splitter.split(
            np.arange(len(y_train)),
            y_train,
            train_groups,
        ):
            if (
                len(np.unique(y_train[candidate_fit_idx])) >= 2
                and len(np.unique(y_train[candidate_valid_idx])) >= 2
            ):
                final_fit_idx, final_valid_idx = candidate_fit_idx, candidate_valid_idx
                break
        final_params, num_boost_round = split_boost_round({**model_params, **best_params})
        if final_fit_idx is None or final_valid_idx is None:
            booster = train_booster(final_params, train_matrix, y_train, num_boost_round=num_boost_round)
        else:
            booster = train_booster(
                final_params,
                train_matrix[final_fit_idx],
                y_train[final_fit_idx],
                num_boost_round=num_boost_round,
                valid_matrix=train_matrix[final_valid_idx],
                y_valid=y_train[final_valid_idx],
                stop=int(stop),
            )
            best_iteration = getattr(booster, 'best_iteration', None)
            if best_iteration is not None:
                metrics['best_iteration'] = int(best_iteration) + 1
        model = {'booster': booster, 'params': final_params, 'num_boost_round': num_boost_round}
    else:
        fit_params, num_boost_round = split_boost_round(model_params)
        booster = train_booster(fit_params, train_matrix, y_train, num_boost_round=num_boost_round)
        model = {'booster': booster, 'params': fit_params, 'num_boost_round': num_boost_round}
        cv_results = pl.DataFrame([{
            'status': 'fit_without_search',
            'reason': 'no search space or iterations provided',
        }])
        metrics['status'] = 'fit_without_search'
    holdout_pred_raw = predict_scores(model['booster'], test_matrix)
    calibrator, calibration_metrics = fit_isotonic_calibrator(
        model,
        train_matrix,
        y_train,
        train_groups=train_groups if can_group_cv else None,
        cv_splits=cv_splits,
    )
    metrics.update(calibration_metrics)
    holdout_pred = calibrator.transform(holdout_pred_raw) if calibrator is not None else holdout_pred_raw
    metrics.update({
        'n_test_events': int(len(y_test)),
        'n_test_goals': int(y_test.sum()),
        'n_train_replays': int(len(np.unique(train_groups))),
        'n_test_replays': int(len(np.unique(groups[test_idx]))),
        'test_log_loss_raw': float(log_loss(y_test, holdout_pred_raw, labels=[0, 1])),
        'test_brier_raw': float(brier_score_loss(y_test, holdout_pred_raw)),
        'test_log_loss': float(log_loss(y_test, holdout_pred, labels=[0, 1])),
        'test_brier': float(brier_score_loss(y_test, holdout_pred)),
        'test_roc_auc': float(roc_auc_score(y_test, holdout_pred)) if len(np.unique(y_test)) > 1 else np.nan,
        'test_average_precision': float(average_precision_score(y_test, holdout_pred)) if len(np.unique(y_test)) > 1 else np.nan,
    })

    csv_safe_frame(cv_results).write_csv(os.path.join(segment_folder, f'cv_results_{segment}.csv'))
    names = feature_names(numeric_cols, categorical_cols, preprocessor)
    plot_feature_importance(
        model,
        test_matrix,
        names,
        os.path.join(segment_folder, f'feature_importance_{segment}.png'),
        csv_path=os.path.join(segment_folder, f'feature_importance_{segment}.csv'),
        seed=seed or 42,
    )
    if len(np.unique(y_test)) > 1:
        plot_roc_auc(y_test, holdout_pred, segment, os.path.join(segment_folder, f'roc_auc_{segment}.png'))
        plot_calibration(y_test, holdout_pred, segment, os.path.join(segment_folder, f'calibration_{segment}.png'))

    predictions = None
    if return_scored:
        full_matrix = make_sparse_matrix(model_df, numeric_cols, categorical_cols, preprocessor=preprocessor)
        full_pred_raw = predict_scores(model['booster'], full_matrix)
        full_pred = calibrator.transform(full_pred_raw) if calibrator is not None else full_pred_raw
        predictions = ids.with_columns([
            pl.Series('xG', full_pred),
            pl.Series('xG_raw', full_pred_raw),
            pl.lit(segment).alias('xG_model_segment'),
        ])
        del full_matrix, ids, model_df
    with open(os.path.join(segment_folder, f'metrics_{segment}.json'), 'w') as handle:
        json.dump(metrics, handle, indent=2)
    joblib.dump(
        {
            'preprocessor': preprocessor,
            'model': model['booster'],
            'model_params': model['params'],
            'num_boost_round': model['num_boost_round'],
            'calibrator': calibrator,
            'categorical_cols': categorical_cols,
            'numeric_cols': numeric_cols,
            'feature_names': names,
        },
        os.path.join(segment_folder, f'xg_model_{segment}.joblib'),
    )
    del train_matrix, test_matrix
    gc.collect()
    return metrics, predictions


def analyzerl_xg(
    pbp_folder: str | Path = PBP_FOLDER,
    model_folder: str | Path = MODEL_FOLDER,
    n_rand: int | None = None,
    random_state: int = 42,
    return_scored: bool = False,
    folds: int = 3,
    iters: int = 0,
    jobs: int | None = None,
    params: Mapping[str, Any] | None = None,
    space: Mapping[str, Any] | None = None,
    stop: int = 50,
    gpu: bool = False,
) -> pl.DataFrame:
    """Build the standard xG model from frame or PBP exports.

    Args:
        pbp_folder: Folder containing frame or PBP `.csv` or `.parquet` files.
        model_folder: Output folder for saved model artifacts.
        n_rand: Optional random subset size for input files.
        random_state: Seed for file sampling and randomized search.
        return_scored: Whether to return the scored training rows.
        folds: Grouped cross-validation fold count used for tuning and calibration.
        iters: Randomized search iteration count. Use `0` to fit without search.
        jobs: Optional XGBoost/Optuna parallelism override.
        params: Optional direct `XGBClassifier` keyword arguments.
        space: Optional Optuna search-space specification. Uses built-in defaults when omitted.
        stop: Early-stopping rounds used during Optuna tuning and final tuned fit.
        gpu: Whether to request CUDA execution when available.

    Returns:
        A Polars DataFrame of scored rows when `return_scored=True`, otherwise an empty frame.
    """
    os.makedirs(model_folder, exist_ok=True)
    print('Starting AnalyzeRL xG model build', flush=True)
    pbp_files = input_files(pbp_folder)
    total_pbp_files = len(pbp_files)
    if n_rand is not None:
        n_rand = int(n_rand)
        if n_rand <= 0:
            raise ValueError('n_rand must be a positive integer when provided')
        if n_rand < total_pbp_files:
            rng = np.random.default_rng(random_state)
            selected_indices = np.sort(rng.choice(total_pbp_files, size=n_rand, replace=False))
            pbp_files = [pbp_files[idx] for idx in selected_indices]
    team_size_filter = pl.col('team_size').cast(pl.Float32, strict=False) == 3.0
    requested_columns = model_pbp_columns()
    try:
        threes_shots = (
            prepare_model_shots(scan_pbp_polars(pbp_files, requested_columns=requested_columns, event_filter=False))
            .filter(team_size_filter)
            .collect(engine='streaming')
        )
    except Exception:
        if not any(path.endswith('.parquet') for path in pbp_files):
            raise
        threes_shots = (
            prepare_model_shots(
                collect_pbp_polars_skip_corrupt_parquet(
                    pbp_files,
                    requested_columns=requested_columns,
                    event_filter=False,
                )
                .lazy()
            )
            .filter(team_size_filter)
            .collect(engine='streaming')
        )
    total_shots = int(threes_shots.height)
    if total_shots == 0:
        raise ValueError(f'No shot or goal events found in {XG_LABEL} frame/PBP data')
    print(f'{XG_LABEL}: collected {threes_shots.height} shot rows for fitting', flush=True)

    metrics, predictions = train_segment(
        XG_LABEL,
        threes_shots,
        model_folder,
        return_scored=return_scored,
        nested=False,
        folds=folds,
        iters=iters,
        jobs=jobs,
        params=params,
        space=space,
        seed=random_state,
        stop=stop,
        gpu=gpu,
    )

    if return_scored and predictions is not None:
        scored_pbp = threes_shots.join(predictions, on=['game_id', 'event_number'], how='left')
        if 'xG_model_segment_right' in scored_pbp.columns:
            scored_pbp = scored_pbp.drop('xG_model_segment_right')
    elif return_scored:
        scored_pbp = threes_shots.with_columns([
            pl.lit(None, dtype=pl.Float64).alias('xG'),
            pl.lit(None, dtype=pl.Float64).alias('xG_raw'),
            pl.lit(XG_LABEL).alias('xG_model_segment'),
        ])
    else:
        del threes_shots
        scored_pbp = pl.DataFrame()

    return scored_pbp

if __name__ == '__main__':
    analyzerl_xg()
 
