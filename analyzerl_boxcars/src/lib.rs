use anyhow::{anyhow, Context, Result};
use arrow_array::{ArrayRef, BooleanArray, Float32Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use boxcars::{Attribute, CamSettings, HeaderProp, RemoteId, Replay, UniqueId, UpdatedAttribute};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

mod engineering;
mod parsing;

use engineering::{boost_units, raw_boost_units, EventModel};
use parsing::build_pbp_rows;

const CAR_CONTACT_DISTANCE: f32 = 225.0;
const CAR_CONTACT_COOLDOWN_FRAMES: i32 = 15;
const CROSSBAR_HEIGHT: f32 = 642.775;
const CHALLENGE_TOUCH_BALL_DISTANCE: f32 = 425.0;
const CHALLENGE_TOUCH_PLAYER_DISTANCE: f32 = 750.0;
const CHALLENGE_EVENT_COOLDOWN_FRAMES: i32 = 30;
const PRESS_CARRIER_DISTANCE: f32 = 900.0;
const PRESS_BALL_DISTANCE: f32 = 900.0;
const PRESS_EVENT_COOLDOWN_FRAMES: i32 = 60;
const SHADOW_MIN_CARRIER_DISTANCE: f32 = 500.0;
const SHADOW_MAX_CARRIER_DISTANCE: f32 = 1800.0;
const SHADOW_LATERAL_DISTANCE: f32 = 1400.0;
const SHADOW_MIN_CARRIER_SPEED_TOWARD_NET: f32 = 250.0;
const SHADOW_EVENT_COOLDOWN_FRAMES: i32 = 90;
const DEMO_RESPAWN_FRAMES: i32 = 90;
const OFF_DEMO_SECONDS: f32 = 2.0;
const OFF_CHALLENGE_SECONDS: f32 = 5.0;
const OFF_FLIP_RESET_SECONDS: f32 = 2.0;
const OFF_KICKOFF_SECONDS: f32 = 5.0;
const OFF_ZONE_EVENT_SECONDS: f32 = 2.0;
const DEMO_EVENT_COOLDOWN_FRAMES: i32 = 150;
const POST_GOAL_KICKOFF_WINDOW_FRAMES: i32 = 600;
const REBOUND_SECONDS: f32 = 3.0;
const DOUBLE_TAP_SECONDS: f32 = 5.0;
const DOUBLE_TAP_BACK_WALL_PROJECTION_SECONDS: f32 = 3.5;
const DOUBLE_TAP_BACK_WALL_DISTANCE: f32 = 900.0;
const DOUBLE_TAP_CAR_BACK_WALL_DISTANCE: f32 = 700.0;
const DRIBBLE_WINDOW_SECONDS: f32 = 3.0;
const FLICK_WINDOW_SECONDS: f32 = 1.0;
const HOOD_DRIBBLE_HORIZONTAL_DISTANCE: f32 = 180.0;
const HOOD_DRIBBLE_MIN_VERTICAL_SEPARATION: f32 = 70.0;
const HOOD_DRIBBLE_MAX_VERTICAL_SEPARATION: f32 = 260.0;
const BALL_RADIUS: f32 = 91.25;
const SIDE_WALL_X: f32 = 4096.0;
const BACK_WALL_Y: f32 = 5120.0;
const BACK_NET_Y: f32 = 6000.0;
const CEILING_Z: f32 = 2044.0;
const GOAL_HEIGHT: f32 = 642.775;
const GOAL_CENTER_TO_POST: f32 = 892.755;
const GRAVITY: f32 = 650.0;
const MISSED_SHOT_MAX_LATERAL_MISS: f32 = 2500.0;
const MISSED_SHOT_MAX_HEIGHT: f32 = 2044.0;
const MISSED_PASS_PROJECTION_SECONDS: f32 = 2.5;
const MISSED_PASS_TARGET_RADIUS: f32 = 450.0;
const MISSED_PASS_MAX_TARGET_MISS: f32 = 1800.0;
const MISSED_PASS_MIN_SPEED: f32 = 900.0;
const MISSED_PASS_MIN_FORWARD_DOT: f32 = 0.35;
const SUPERSONIC_THRESHOLD: f32 = 2200.0;
const WALL_SHOT_DISTANCE: f32 = 350.0;
const CEILING_SHOT_DISTANCE: f32 = 350.0;
const HISTORY_HALF_LIFE_SECONDS: f32 = 8.0;
const POSSESSION_DISTANCE: f32 = 300.0;
const FLIP_RESET_CONTACT_DISTANCE: f32 = 230.0;
const FLIP_RESET_FRAME_WINDOW: i32 = 30;
const FLIP_RESET_MIN_CAR_Z: f32 = 120.0;
const FLIP_RESET_UNDERSIDE_Z: f32 = -25.0;
const DOUBLE_COMMIT_BALL_DISTANCE: f32 = 1100.0;
const DOUBLE_COMMIT_TEAMMATE_DISTANCE: f32 = 1300.0;
const DOUBLE_COMMIT_COOLDOWN_FRAMES: i32 = 45;
const WHIFF_BALL_DISTANCE: f32 = 285.0;
const WHIFF_PREVIOUS_BALL_DISTANCE: f32 = 520.0;
const WHIFF_CROSS_BALL_DISTANCE: f32 = 145.0;
const WHIFF_MIN_SPEED_TOWARD_BALL: f32 = 800.0;
const WHIFF_COMMITTED_SPEED_TOWARD_BALL: f32 = 1150.0;
const WHIFF_TOUCH_EXCLUSION_FRAMES: i32 = 10;
const WHIFF_ANY_TOUCH_EXCLUSION_FRAMES: i32 = 1;
const WHIFF_COOLDOWN_FRAMES: i32 = 120;
const WHIFF_DIRECT_TOUCH_WINDOW_FRAMES: i32 = 90;
const FAKE_POSSESSION_DISTANCE: f32 = 560.0;
const FRAME_PARQUET_ROW_GROUP_SIZE: usize = 2048;
const TIME_ON_FIELD_WORKGROUP_SIZE: u32 = 256;
const TIME_ON_FIELD_SHADER: &str = r#"
struct Params {
    interval_count: u32,
    player_count: u32,
}

@group(0) @binding(0) var<storage, read> deltas: array<f32>;
@group(0) @binding(1) var<storage, read> active: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = params.interval_count * params.player_count;
    if (idx >= total) {
        return;
    }
    let interval_idx = idx / params.player_count;
    output[idx] = deltas[interval_idx] * active[idx];
}
"#;

#[derive(Clone, Copy, Debug)]
enum ColumnKind {
    Utf8,
    Int32,
    Float32,
    Boolean,
}

#[derive(Clone, Debug)]
enum CellValue {
    Utf8(String),
    Int32(i32),
    Float32(f32),
    Boolean(bool),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExportFormat {
    Csv,
    Parquet,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuMode {
    None,
    Auto,
    Rocm,
    Cuda,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContextMode {
    Full,
    FramesOnly,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TimeOnFieldParams {
    interval_count: u32,
    player_count: u32,
}

#[derive(Clone, Debug)]
pub struct ParseArgs {
    replays: Vec<PathBuf>,
    out_json: Option<PathBuf>,
    out_pbp: PathBuf,
    out_analysis: Option<PathBuf>,
    out_players: Option<PathBuf>,
    out_frames: Option<PathBuf>,
    out_actors: Option<PathBuf>,
    out_attributes: Option<PathBuf>,
    workers: Option<usize>,
    limit: Option<usize>,
    export_meta: bool,
    export_network: bool,
    force: bool,
    pbp_format: ExportFormat,
    gpu: GpuMode,
    rotation_events: bool,
}

#[derive(Clone, Debug)]
pub struct FramesArgs {
    replays: Vec<PathBuf>,
    out_frames: PathBuf,
    workers: Option<usize>,
    limit: Option<usize>,
    parse_only: bool,
    frames_only: bool,
    no_write: bool,
    force: bool,
    frames_format: ExportFormat,
    gpu: GpuMode,
    rotation_events: bool,
}

#[derive(Clone, Debug)]
pub struct StatsArgs {
    replays: Vec<PathBuf>,
    out_stats: PathBuf,
    workers: Option<usize>,
    limit: Option<usize>,
    force: bool,
    stats_format: ExportFormat,
    gpu: GpuMode,
}

#[derive(Clone, Debug)]
pub struct IndexArgs {
    replays: PathBuf,
    out_analysis: PathBuf,
    workers: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct MatchGuidArgs {
    replays: PathBuf,
    match_guid: String,
    out_json: Option<PathBuf>,
    out_pbp: PathBuf,
    out_analysis: Option<PathBuf>,
    out_players: Option<PathBuf>,
    out_frames: Option<PathBuf>,
    out_actors: Option<PathBuf>,
    out_attributes: Option<PathBuf>,
    export_meta: bool,
    export_network: bool,
    pbp_format: ExportFormat,
}

#[derive(Clone, Debug)]
pub struct MatchGuidsArgs {
    replays: PathBuf,
    match_guids: PathBuf,
    out_json: Option<PathBuf>,
    out_pbp: PathBuf,
    out_analysis: Option<PathBuf>,
    out_players: Option<PathBuf>,
    out_frames: Option<PathBuf>,
    out_actors: Option<PathBuf>,
    out_attributes: Option<PathBuf>,
    export_meta: bool,
    export_network: bool,
    pbp_format: ExportFormat,
}

#[derive(Serialize)]
struct AnalysisRow {
    game_id: String,
    parse_status: String,
    frame_count: usize,
    keyframe_count: usize,
    tick_mark_count: usize,
    object_count: usize,
    name_count: usize,
    class_index_count: usize,
    net_cache_count: usize,
    major_version: i32,
    minor_version: i32,
    net_version: Option<i32>,
    game_type: String,
    id: String,
    replay_name: String,
    map_name: String,
    date: String,
    match_type: String,
    team_size: Option<i32>,
    playlist: String,
}

#[derive(Serialize)]
struct FrameRow {
    game_id: String,
    frame_number: usize,
    time: f32,
    delta: f32,
    new_actor_count: usize,
    deleted_actor_count: usize,
    updated_actor_count: usize,
}

#[derive(Serialize)]
struct ActorRow {
    game_id: String,
    frame_number: usize,
    time: f32,
    actor_id: i32,
    name_id: Option<i32>,
    actor_name: String,
    object_id: i32,
    object_name: String,
    spawn_location_x: Option<i32>,
    spawn_location_y: Option<i32>,
    spawn_location_z: Option<i32>,
    spawn_rotation_yaw: Option<i8>,
    spawn_rotation_pitch: Option<i8>,
    spawn_rotation_roll: Option<i8>,
}

#[derive(Serialize)]
struct DeletedActorRow {
    game_id: String,
    frame_number: usize,
    time: f32,
    actor_id: i32,
}

#[derive(Serialize)]
struct AttributeRow {
    game_id: String,
    frame_number: usize,
    time: f32,
    actor_id: i32,
    stream_id: i32,
    object_id: i32,
    object_name: String,
    attribute_type: String,
    bool_value: Option<bool>,
    int_value: Option<i32>,
    int64_value: Option<String>,
    float_value: Option<f32>,
    string_value: String,
    byte_value: Option<u8>,
    active_actor_active: Option<bool>,
    active_actor_id: Option<i32>,
    location_x: Option<f32>,
    location_y: Option<f32>,
    location_z: Option<f32>,
    rotation_x: Option<f32>,
    rotation_y: Option<f32>,
    rotation_z: Option<f32>,
    rotation_w: Option<f32>,
    linear_velocity_x: Option<f32>,
    linear_velocity_y: Option<f32>,
    linear_velocity_z: Option<f32>,
    angular_velocity_x: Option<f32>,
    angular_velocity_y: Option<f32>,
    angular_velocity_z: Option<f32>,
    boost_amount: Option<u8>,
    boost_grant_count: Option<u8>,
    demolish_attacker_id: Option<i32>,
    demolish_victim_id: Option<i32>,
    stat_event_object_id: Option<i32>,
}

#[derive(Serialize)]
struct PlayerRow {
    game_id: String,
    player_name: String,
    player_team: Option<i32>,
    online_id: String,
    platform: String,
    score: Option<i32>,
    goals: Option<i32>,
    assists: Option<i32>,
    saves: Option<i32>,
    shots: Option<i32>,
    b_bot: Option<bool>,
}

struct PbpEventRecord {
    frame_number: Option<i32>,
    event_type: String,
    values: RowValues,
}

#[derive(Clone, Copy, Debug)]
struct PbpBuildOptions {
    rotation_events: bool,
}

impl Default for PbpBuildOptions {
    fn default() -> Self {
        Self {
            rotation_events: true,
        }
    }
}

#[derive(Clone, Debug)]
struct RowValues {
    cells: Vec<Option<CellValue>>,
}

impl RowValues {
    fn new() -> Self {
        Self {
            cells: vec![None; pbp_columns_cached().len()],
        }
    }

    fn get(&self, key: &str) -> Option<&CellValue> {
        let idx = *pbp_column_index_cached().get(key)?;
        self.cells.get(idx).and_then(Option::as_ref)
    }

    fn insert(&mut self, key: String, value: String) {
        let Some(idx) = pbp_column_index_cached().get(&key).copied() else {
            return;
        };
        self.cells[idx] = parse_cell_value(pbp_column_kinds_cached()[idx], &value);
    }

    fn set_cell(&mut self, key: &str, value: CellValue) {
        let Some(idx) = pbp_column_index_cached().get(key).copied() else {
            return;
        };
        self.cells[idx] = Some(value);
    }

    fn insert_utf8(&mut self, key: &str, value: String) {
        self.set_cell(key, CellValue::Utf8(value));
    }

    fn insert_i32(&mut self, key: &str, value: i32) {
        self.set_cell(key, CellValue::Int32(value));
    }

    fn insert_f32(&mut self, key: &str, value: f32) {
        if value.is_finite() {
            self.set_cell(key, CellValue::Float32(value));
        }
    }

    fn insert_bool(&mut self, key: &str, value: bool) {
        self.set_cell(key, CellValue::Boolean(value));
    }

    fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    fn extend<I>(&mut self, values: I)
    where
        I: IntoIterator<Item = (String, String)>,
    {
        for (key, value) in values {
            self.insert(key, value);
        }
    }

    fn iter(&self) -> impl Iterator<Item = (&str, &CellValue)> {
        self.cells.iter().enumerate().filter_map(|(idx, value)| {
            value
                .as_ref()
                .map(|cell| (pbp_columns_cached()[idx].as_str(), cell))
        })
    }

    fn as_slice(&self) -> &[Option<CellValue>] {
        &self.cells
    }

    fn into_cells(self) -> Vec<Option<CellValue>> {
        self.cells
    }
}

#[derive(Clone, Debug)]
struct OfficialStatEvent {
    pri_actor_id: Option<i32>,
    frame_number: i32,
    player_name: String,
    stat_type: &'static str,
    stat_number: i32,
}

struct PendingOfficialStatEvent {
    pri_actor_id: i32,
    frame_number: i32,
    stat_type: &'static str,
    stat_number: i32,
}

#[derive(Clone, Debug)]
struct PlayerInfo {
    id: String,
    actor_id: String,
    network_id: String,
    name: String,
    team: i32,
    slot: String,
    platform: String,
    is_bot: String,
    score: String,
    title_id: String,
    first_frame_in_game: String,
    time_in_game: String,
    car_id: String,
    car_name: String,
    decal_id: String,
    wheels_id: String,
    boost_id: String,
    antenna_id: String,
    topper_id: String,
    engine_audio_id: String,
    trail_id: String,
    goal_explosion_id: String,
    primary_paint_finish_id: String,
    accent_paint_finish_id: String,
    camera_settings: Option<PlayerCameraSettings>,
}

#[derive(Clone, Copy, Debug)]
struct PlayerCameraSettings {
    fov: f32,
    height: f32,
    angle: f32,
    distance: f32,
    stiffness: f32,
    swivel: f32,
    transition: Option<f32>,
}

impl From<CamSettings> for PlayerCameraSettings {
    fn from(value: CamSettings) -> Self {
        Self {
            fov: value.fov,
            height: value.height,
            angle: value.angle,
            distance: value.distance,
            stiffness: value.stiffness,
            swivel: value.swivel,
            transition: value.transition,
        }
    }
}

#[derive(Default)]
struct PbpContext {
    server_name: String,
    game_server_id: String,
    playlist: String,
    blue_team_name: String,
    orange_team_name: String,
    players: Vec<PlayerInfo>,
    official_stats: Vec<OfficialStatEvent>,
    game_presence_events: Vec<GamePresenceEvent>,
    demo_events: Vec<CarContactEvent>,
    ball_events: Vec<BallEvent>,
    frame_states: Vec<FrameSnapshot>,
}

#[derive(Clone, Debug)]
struct GamePresenceEvent {
    frame_number: i32,
    player_name: String,
    event_type: &'static str,
}

#[derive(Clone, Debug)]
struct BoostPickupEvent {
    frame_number: i32,
    player_name: String,
    amount: i32,
    pickup_type: &'static str,
}

#[derive(Clone, Debug)]
struct FlipResetEvent {
    frame_number: i32,
    player_name: String,
    reset_origin: &'static str,
}

#[derive(Clone, Debug)]
struct CarContactEvent {
    frame_number: i32,
    event_type: String,
    player_1_name: String,
    player_2_name: String,
    car_contact_distance: f32,
    relative_speed: f32,
    event_player_1_speed: f32,
    event_player_2_speed: f32,
    event_player_1_demolished: bool,
    event_player_2_demolished: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Quat {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct EntityState {
    pos: Vec3,
    vel: Vec3,
    ang_vel: Vec3,
    rot: Quat,
    has_pos: bool,
}

#[derive(Clone, Debug, Default)]
struct PlayerFrameState {
    entity: EntityState,
    boost: Option<u8>,
    boost_updated_frame: Option<i32>,
    boost_active: bool,
    boost_collect: Option<u8>,
    throttle: Option<i32>,
    steer: Option<i32>,
    handbrake: bool,
    ball_cam: bool,
    dodge_active: bool,
    jump_active: bool,
    double_jump_active: bool,
    jumped: bool,
    flipped: bool,
    jump_air_activate_count: Option<i32>,
    double_jump_air_activate_count: Option<i32>,
    dodge_air_activate_count: Option<i32>,
    dodges_refreshed_counter: Option<i32>,
    supersonic: bool,
    flip_available: bool,
}

#[derive(Clone, Debug, Default)]
struct FrameSnapshot {
    frame_number: i32,
    seconds_remaining: Option<i32>,
    seconds_elapsed: Option<f32>,
    ball: Option<EntityState>,
    players: Vec<Option<PlayerFrameState>>,
}

#[derive(Clone, Debug)]
struct HitCandidate {
    frame_number: i32,
    player_name: String,
    collision_distance: f32,
    ball_state: EntityState,
    player_positions: Vec<Option<Vec3>>,
    goal_number: i32,
}

#[derive(Clone, Debug)]
struct BallEvent {
    frame_number: i32,
    event_type: String,
    player_name: String,
    player_2_name: String,
    player_3_name: String,
    collision_distance: f32,
    distance: f32,
    distance_to_goal: f32,
    previous_hit_frame_number: Option<i32>,
    next_hit_frame_number: Option<i32>,
    goal_number: i32,
    ball_state: EntityState,
    player_positions: Vec<Option<Vec3>>,
    goal: bool,
    shot: bool,
    missed_shot: bool,
    missed_pass: bool,
    pass_: bool,
    clear: bool,
    save: bool,
    assist: bool,
}

fn to_py_err(err: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{err:#}"))
}

#[pyfunction(signature = (replay_path, workers = None, rotation_events = true))]
fn parse_frames(
    py: Python<'_>,
    replay_path: String,
    workers: Option<usize>,
    rotation_events: bool,
) -> PyResult<Py<PyAny>> {
    if let Some(workers) = workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .ok();
    }

    let paths = replay_paths(Path::new(&replay_path)).map_err(to_py_err)?;

    let results: Vec<_> = paths
        .par_iter()
        .map(|path| {
            let game_id = path
                .file_stem()
                .and_then(|value| value.to_str())
                .ok_or_else(|| anyhow!("bad replay filename: {}", path.display()))?
                .to_string();

            let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;

            let replay = boxcars::ParserBuilder::new(&bytes)
                .must_parse_network_data()
                .parse()
                .with_context(|| format!("boxcars parsing {}", path.display()))?;

            let (_, rows) = build_pbp_rows(&game_id, &replay, PbpBuildOptions { rotation_events })?;

            Ok::<_, anyhow::Error>((game_id, rows))
        })
        .collect();

    let output = pyo3::types::PyList::empty(py);

    for result in results {
        let (game_id, rows) = result.map_err(to_py_err)?;

        let replay_dict = PyDict::new(py);
        replay_dict.set_item("game_id", game_id)?;

        let row_list = pyo3::types::PyList::empty(py);

        for row in rows {
            let row_dict = PyDict::new(py);

            for (key, value) in row.values.iter() {
                row_dict.set_item(key, cell_to_string(value))?;
            }

            row_list.append(row_dict)?;
        }

        replay_dict.set_item("rows", row_list)?;
        output.append(replay_dict)?;
    }

    Ok(output.into())
}

#[pymodule]
fn _boxcars(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_frames, m)?)?;
    Ok(())
}

pub fn frames_args(args: Vec<String>) -> Result<FramesArgs> {
    let mut replays = Vec::new();
    let mut out_frames = None;
    let mut workers = None;
    let mut limit = None;
    let mut parse_only = false;
    let mut frames_only = false;
    let mut no_write = false;
    let mut force = false;
    let mut frames_format = ExportFormat::Parquet;
    let mut gpu = GpuMode::None;
    let mut rotation_events = true;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--replays" => replays.push(next_path(&args, &mut idx)?),
            "--out-frames" => out_frames = Some(next_path(&args, &mut idx)?),
            "--workers" => workers = Some(next_value(&args, &mut idx)?.parse()?),
            "--limit" => limit = Some(next_value(&args, &mut idx)?.parse()?),
            "--format" | "--frames-format" => {
                frames_format = parse_export_format(&next_value(&args, &mut idx)?)?
            }
            "--gpu" => gpu = parse_gpu_mode(&next_value(&args, &mut idx)?)?,
            "--rotation-events" => rotation_events = parse_bool_flag(&args, &mut idx)?,
            "--no-rotation-events" => rotation_events = false,
            "--parse-only" => parse_only = true,
            "--frames-only" => frames_only = true,
            "--no-write" => no_write = true,
            "--force" => force = true,
            flag => return Err(anyhow!("unknown frames flag: {flag}")),
        }
        idx += 1;
    }
    if replays.is_empty() {
        return Err(anyhow!("missing --replays"));
    }
    Ok(FramesArgs {
        replays,
        out_frames: out_frames.ok_or_else(|| anyhow!("missing --out-frames"))?,
        workers,
        limit,
        parse_only,
        frames_only,
        no_write,
        force,
        frames_format,
        gpu,
        rotation_events,
    })
}

pub fn parse_args(args: Vec<String>) -> Result<ParseArgs> {
    let mut replays = Vec::new();
    let mut out_json = None;
    let mut out_analysis = None;
    let mut out_players = None;
    let mut out_pbp = None;
    let mut out_frames = None;
    let mut out_actors = None;
    let mut out_attributes = None;
    let mut workers = None;
    let mut limit = None;
    let mut export_meta = false;
    let mut export_network = false;
    let mut force = false;
    let mut pbp_format = ExportFormat::Csv;
    let mut gpu = GpuMode::None;
    let mut rotation_events = true;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--replays" => replays.push(next_path(&args, &mut idx)?),
            "--out-json" => out_json = Some(next_path(&args, &mut idx)?),
            "--out-analysis" => out_analysis = Some(next_path(&args, &mut idx)?),
            "--out-players" => out_players = Some(next_path(&args, &mut idx)?),
            "--out-pbp" => out_pbp = Some(next_path(&args, &mut idx)?),
            "--out-frames" => out_frames = Some(next_path(&args, &mut idx)?),
            "--out-actors" => out_actors = Some(next_path(&args, &mut idx)?),
            "--out-attributes" => out_attributes = Some(next_path(&args, &mut idx)?),
            "--workers" => workers = Some(next_value(&args, &mut idx)?.parse()?),
            "--limit" => limit = Some(next_value(&args, &mut idx)?.parse()?),
            "--format" | "--pbp-format" => {
                pbp_format = parse_export_format(&next_value(&args, &mut idx)?)?
            }
            "--gpu" => gpu = parse_gpu_mode(&next_value(&args, &mut idx)?)?,
            "--rotation-events" => rotation_events = parse_bool_flag(&args, &mut idx)?,
            "--no-rotation-events" => rotation_events = false,
            "--export-meta" => export_meta = true,
            "--export-network" => export_network = true,
            "--force" => force = true,
            flag => return Err(anyhow!("unknown parse flag: {flag}")),
        }
        idx += 1;
    }
    if replays.is_empty() {
        return Err(anyhow!("missing --replays"));
    }
    Ok(ParseArgs {
        replays,
        out_json,
        out_pbp: out_pbp.ok_or_else(|| anyhow!("missing --out-pbp"))?,
        out_analysis,
        out_players,
        out_frames,
        out_actors,
        out_attributes,
        workers,
        limit,
        export_meta,
        export_network,
        force,
        pbp_format,
        gpu,
        rotation_events,
    })
}

pub fn stats_args(args: Vec<String>) -> Result<StatsArgs> {
    let mut replays = Vec::new();
    let mut out_stats = None;
    let mut workers = None;
    let mut limit = None;
    let mut force = false;
    let mut stats_format = ExportFormat::Csv;
    let mut gpu = GpuMode::None;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--replays" => replays.push(next_path(&args, &mut idx)?),
            "--out-stats" => out_stats = Some(next_path(&args, &mut idx)?),
            "--workers" => workers = Some(next_value(&args, &mut idx)?.parse()?),
            "--limit" => limit = Some(next_value(&args, &mut idx)?.parse()?),
            "--format" | "--stats-format" => {
                stats_format = parse_export_format(&next_value(&args, &mut idx)?)?
            }
            "--gpu" => gpu = parse_gpu_mode(&next_value(&args, &mut idx)?)?,
            "--force" => force = true,
            flag => return Err(anyhow!("unknown stats flag: {flag}")),
        }
        idx += 1;
    }
    if replays.is_empty() {
        return Err(anyhow!("missing --replays"));
    }
    Ok(StatsArgs {
        replays,
        out_stats: out_stats.ok_or_else(|| anyhow!("missing --out-stats"))?,
        workers,
        limit,
        force,
        stats_format,
        gpu,
    })
}

pub fn index_args(args: Vec<String>) -> Result<IndexArgs> {
    let mut replays = None;
    let mut out_analysis = None;
    let mut workers = None;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--replays" => replays = Some(next_path(&args, &mut idx)?),
            "--out-analysis" => out_analysis = Some(next_path(&args, &mut idx)?),
            "--workers" => workers = Some(next_value(&args, &mut idx)?.parse()?),
            flag => return Err(anyhow!("unknown index flag: {flag}")),
        }
        idx += 1;
    }
    Ok(IndexArgs {
        replays: replays.ok_or_else(|| anyhow!("missing --replays"))?,
        out_analysis: out_analysis.ok_or_else(|| anyhow!("missing --out-analysis"))?,
        workers,
    })
}

pub fn match_guid_args(args: Vec<String>) -> Result<MatchGuidArgs> {
    let mut replays = None;
    let mut match_guid = None;
    let mut out_json = None;
    let mut out_analysis = None;
    let mut out_players = None;
    let mut out_pbp = None;
    let mut out_frames = None;
    let mut out_actors = None;
    let mut out_attributes = None;
    let mut export_meta = false;
    let mut export_network = false;
    let mut pbp_format = ExportFormat::Csv;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--replays" => replays = Some(next_path(&args, &mut idx)?),
            "--match-guid" => match_guid = Some(next_value(&args, &mut idx)?),
            "--out-json" => out_json = Some(next_path(&args, &mut idx)?),
            "--out-analysis" => out_analysis = Some(next_path(&args, &mut idx)?),
            "--out-players" => out_players = Some(next_path(&args, &mut idx)?),
            "--out-pbp" => out_pbp = Some(next_path(&args, &mut idx)?),
            "--out-frames" => out_frames = Some(next_path(&args, &mut idx)?),
            "--out-actors" => out_actors = Some(next_path(&args, &mut idx)?),
            "--out-attributes" => out_attributes = Some(next_path(&args, &mut idx)?),
            "--format" | "--pbp-format" => {
                pbp_format = parse_export_format(&next_value(&args, &mut idx)?)?
            }
            "--export-meta" => export_meta = true,
            "--export-network" => export_network = true,
            flag => return Err(anyhow!("unknown match-guid flag: {flag}")),
        }
        idx += 1;
    }
    Ok(MatchGuidArgs {
        replays: replays.ok_or_else(|| anyhow!("missing --replays"))?,
        match_guid: match_guid.ok_or_else(|| anyhow!("missing --match-guid"))?,
        out_json,
        out_pbp: out_pbp.ok_or_else(|| anyhow!("missing --out-pbp"))?,
        out_analysis,
        out_players,
        out_frames,
        out_actors,
        out_attributes,
        export_meta,
        export_network,
        pbp_format,
    })
}

pub fn match_guids_args(args: Vec<String>) -> Result<MatchGuidsArgs> {
    let mut replays = None;
    let mut match_guids = None;
    let mut out_json = None;
    let mut out_analysis = None;
    let mut out_players = None;
    let mut out_pbp = None;
    let mut out_frames = None;
    let mut out_actors = None;
    let mut out_attributes = None;
    let mut export_meta = false;
    let mut export_network = false;
    let mut pbp_format = ExportFormat::Csv;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--replays" => replays = Some(next_path(&args, &mut idx)?),
            "--match-guids" => match_guids = Some(next_path(&args, &mut idx)?),
            "--out-json" => out_json = Some(next_path(&args, &mut idx)?),
            "--out-analysis" => out_analysis = Some(next_path(&args, &mut idx)?),
            "--out-players" => out_players = Some(next_path(&args, &mut idx)?),
            "--out-pbp" => out_pbp = Some(next_path(&args, &mut idx)?),
            "--out-frames" => out_frames = Some(next_path(&args, &mut idx)?),
            "--out-actors" => out_actors = Some(next_path(&args, &mut idx)?),
            "--out-attributes" => out_attributes = Some(next_path(&args, &mut idx)?),
            "--format" | "--pbp-format" => {
                pbp_format = parse_export_format(&next_value(&args, &mut idx)?)?
            }
            "--export-meta" => export_meta = true,
            "--export-network" => export_network = true,
            flag => return Err(anyhow!("unknown match-guids flag: {flag}")),
        }
        idx += 1;
    }
    Ok(MatchGuidsArgs {
        replays: replays.ok_or_else(|| anyhow!("missing --replays"))?,
        match_guids: match_guids.ok_or_else(|| anyhow!("missing --match-guids"))?,
        out_json,
        out_pbp: out_pbp.ok_or_else(|| anyhow!("missing --out-pbp"))?,
        out_analysis,
        out_players,
        out_frames,
        out_actors,
        out_attributes,
        export_meta,
        export_network,
        pbp_format,
    })
}

fn next_path(args: &[String], idx: &mut usize) -> Result<PathBuf> {
    Ok(PathBuf::from(next_value(args, idx)?))
}

fn next_value(args: &[String], idx: &mut usize) -> Result<String> {
    *idx += 1;
    args.get(*idx)
        .cloned()
        .ok_or_else(|| anyhow!("missing value after flag"))
}

fn parse_export_format(value: &str) -> Result<ExportFormat> {
    match value.to_ascii_lowercase().as_str() {
        "csv" => Ok(ExportFormat::Csv),
        "parquet" => Ok(ExportFormat::Parquet),
        _ => Err(anyhow!("export format must be one of: csv, parquet")),
    }
}

fn parse_bool_flag(args: &[String], idx: &mut usize) -> Result<bool> {
    let value = next_value(args, idx)?;
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" | "on" => Ok(true),
        "false" | "0" | "no" | "n" | "off" => Ok(false),
        _ => Err(anyhow!("boolean flag value must be true or false")),
    }
}

fn parse_gpu_mode(value: &str) -> Result<GpuMode> {
    match value.to_ascii_lowercase().as_str() {
        "cpu" => Ok(GpuMode::None),
        "auto" => Ok(GpuMode::Auto),
        "rocm" | "amd" | "radeon" => Ok(GpuMode::Rocm),
        "cuda" | "nvidia" => Ok(GpuMode::Cuda),
        _ => Err(anyhow!("gpu must be one of: auto, cuda, rocm, cpu")),
    }
}

fn configure_gpu_mode(gpu: GpuMode) {
    match gpu {
        GpuMode::None => std::env::remove_var("ANALYZERL_GPU"),
        GpuMode::Auto => std::env::set_var("ANALYZERL_GPU", "auto"),
        GpuMode::Rocm => std::env::set_var("ANALYZERL_GPU", "rocm"),
        GpuMode::Cuda => std::env::set_var("ANALYZERL_GPU", "cuda"),
    }
}

fn export_extension(format: ExportFormat) -> &'static str {
    match format {
        ExportFormat::Csv => "csv",
        ExportFormat::Parquet => "parquet",
    }
}

fn check_export_args(
    export_meta: bool,
    export_network: bool,
    out_analysis: Option<&PathBuf>,
    out_players: Option<&PathBuf>,
    out_frames: Option<&PathBuf>,
    out_actors: Option<&PathBuf>,
    out_attributes: Option<&PathBuf>,
) -> Result<()> {
    if export_meta && (out_analysis.is_none() || out_players.is_none()) {
        return Err(anyhow!(
            "--export-meta requires --out-analysis and --out-players"
        ));
    }
    if export_network && (out_frames.is_none() || out_actors.is_none() || out_attributes.is_none())
    {
        return Err(anyhow!(
            "--export-network requires --out-frames, --out-actors, and --out-attributes"
        ));
    }
    Ok(())
}

pub fn parse_command(args: ParseArgs) -> Result<()> {
    configure_gpu_mode(args.gpu);
    //Prepare only the output folders requested by the selected export mode.
    check_export_args(
        args.export_meta,
        args.export_network,
        args.out_analysis.as_ref(),
        args.out_players.as_ref(),
        args.out_frames.as_ref(),
        args.out_actors.as_ref(),
        args.out_attributes.as_ref(),
    )?;
    if let Some(out_json) = &args.out_json {
        fs::create_dir_all(out_json)?;
    }
    fs::create_dir_all(&args.out_pbp)?;
    if args.export_meta {
        fs::create_dir_all(args.out_analysis.as_ref().unwrap())?;
        fs::create_dir_all(args.out_players.as_ref().unwrap())?;
    }
    if args.export_network {
        fs::create_dir_all(args.out_frames.as_ref().unwrap())?;
        fs::create_dir_all(args.out_actors.as_ref().unwrap())?;
        fs::create_dir_all(args.out_attributes.as_ref().unwrap())?;
    }
    if let Some(workers) = args.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .ok();
    }

    //Parse replays in parallel while keeping progress to one terminal line.
    let mut replay_paths = replay_paths_many(&args.replays)?;
    if let Some(limit) = args.limit {
        replay_paths.truncate(limit);
    }
    let total = replay_paths.len();
    let completed = Arc::new(AtomicUsize::new(0));
    let progress_lock = Arc::new(Mutex::new(()));
    replay_paths.par_iter().for_each(|path| {
        let game_id = replay_game_id(path);
        if let Err(err) = parse_one(path, &args) {
            eprintln!("failed {}: {err:#}", path.display());
        }
        let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
        print_progress_line("parsed", &game_id, done, total, &progress_lock);
    });
    finish_progress_line(total, &progress_lock);
    Ok(())
}

pub fn frames_command(args: FramesArgs) -> Result<()> {
    configure_gpu_mode(args.gpu);
    if !args.no_write {
        fs::create_dir_all(&args.out_frames)?;
    }
    if let Some(workers) = args.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .ok();
    }

    let mut replay_paths = replay_paths_many(&args.replays)?;
    if let Some(limit) = args.limit {
        replay_paths.truncate(limit);
    }
    let total = replay_paths.len();
    let completed = Arc::new(AtomicUsize::new(0));
    let progress_lock = Arc::new(Mutex::new(()));
    replay_paths.par_iter().for_each(|path| {
        let game_id = replay_game_id(path);
        if let Err(err) = frames_one(path, &args) {
            eprintln!("failed {}: {err:#}", path.display());
        }
        let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
        print_progress_line("parsed", &game_id, done, total, &progress_lock);
    });
    finish_progress_line(total, &progress_lock);
    Ok(())
}

#[derive(Default, Serialize)]
struct NativePlayerStatsRow {
    replay_id: String,
    player_id: String,
    player_name: String,
    team: String,
    team_name: String,
    platform: String,
    score: f32,
    games_played: i32,
    time_in_game: f32,
    time_on_field: f32,
    shots: i32,
    goals: i32,
    saves: i32,
    assists: i32,
    touches: i32,
    passes: i32,
    turnovers: i32,
    challenges: i32,
    kickoffs: i32,
    whiffs: i32,
    fakes: i32,
    demos_applied: i32,
    demos_taken: i32,
    bumps: i32,
    bumps_taken: i32,
    teammate_bumps: i32,
    entries: i32,
    exits: i32,
    retrievals: i32,
    missed_shots: i32,
    missed_passes: i32,
    shot_attempts: i32,
    car_id: String,
    car_name: String,
    decal_id: String,
    wheels_id: String,
    boost_id: String,
    antenna_id: String,
    topper_id: String,
    engine_audio_id: String,
    trail_id: String,
    goal_explosion_id: String,
    primary_paint_finish_id: String,
    accent_paint_finish_id: String,
    camera_fov: String,
    camera_height: String,
    camera_angle: String,
    camera_distance: String,
    camera_stiffness: String,
    camera_swivel: String,
    camera_transition: String,
}

pub fn stats_command(args: StatsArgs) -> Result<()> {
    configure_gpu_mode(args.gpu);
    if args.stats_format != ExportFormat::Csv {
        return Err(anyhow!("native stats currently exports csv"));
    }
    if !args.force && args.out_stats.exists() && args.out_stats.metadata()?.len() > 0 {
        return Ok(());
    }
    if let Some(parent) = args.out_stats.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(workers) = args.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .ok();
    }

    let mut replay_paths = replay_paths_many(&args.replays)?;
    if let Some(limit) = args.limit {
        replay_paths.truncate(limit);
    }
    let total = replay_paths.len();
    let completed = Arc::new(AtomicUsize::new(0));
    let progress_lock = Arc::new(Mutex::new(()));
    let rows = replay_paths
        .par_iter()
        .map(|path| {
            let game_id = replay_game_id(path);
            let output = native_stats_one(path, args.gpu);
            let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
            print_progress_line("built stats", &game_id, done, total, &progress_lock);
            output
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    finish_progress_line(total, &progress_lock);

    let mut writer = csv::Writer::from_path(&args.out_stats)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn native_stats_one(path: &Path, gpu: GpuMode) -> Result<Vec<NativePlayerStatsRow>> {
    let game_id = replay_game_id(path);
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let replay = boxcars::ParserBuilder::new(&bytes)
        .must_parse_network_data()
        .parse()
        .with_context(|| format!("boxcars parsing {}", path.display()))?;
    let (context, rows) = build_pbp_rows(&game_id, &replay, PbpBuildOptions::default())?;
    Ok(native_stats_rows(&game_id, &context, &rows, gpu))
}

fn native_stats_rows(
    game_id: &str,
    context: &PbpContext,
    rows: &[PbpEventRecord],
    gpu: GpuMode,
) -> Vec<NativePlayerStatsRow> {
    let team_names = team_names_from_context(context, rows);
    let time_on_field_minutes = native_time_on_field_minutes(context, gpu);
    let mut output = context
        .players
        .iter()
        .enumerate()
        .map(|(player_idx, player)| {
            let camera = player.camera_settings;
            let time_on_field = time_on_field_minutes
                .get(player_idx)
                .copied()
                .unwrap_or_default();
            (
                player.name.clone(),
                NativePlayerStatsRow {
                    replay_id: game_id.to_string(),
                    player_id: player.id.clone(),
                    player_name: player.name.clone(),
                    team: team_name(player.team).to_string(),
                    team_name: team_names.get(&player.team).cloned().unwrap_or_default(),
                    platform: player.platform.clone(),
                    score: player.score.parse::<f32>().unwrap_or(0.0),
                    games_played: 1,
                    time_in_game: time_on_field,
                    time_on_field,
                    car_id: player.car_id.clone(),
                    car_name: player.car_name.clone(),
                    decal_id: player.decal_id.clone(),
                    wheels_id: player.wheels_id.clone(),
                    boost_id: player.boost_id.clone(),
                    antenna_id: player.antenna_id.clone(),
                    topper_id: player.topper_id.clone(),
                    engine_audio_id: player.engine_audio_id.clone(),
                    trail_id: player.trail_id.clone(),
                    goal_explosion_id: player.goal_explosion_id.clone(),
                    primary_paint_finish_id: player.primary_paint_finish_id.clone(),
                    accent_paint_finish_id: player.accent_paint_finish_id.clone(),
                    camera_fov: camera
                        .map(|value| value.fov.to_string())
                        .unwrap_or_default(),
                    camera_height: camera
                        .map(|value| value.height.to_string())
                        .unwrap_or_default(),
                    camera_angle: camera
                        .map(|value| value.angle.to_string())
                        .unwrap_or_default(),
                    camera_distance: camera
                        .map(|value| value.distance.to_string())
                        .unwrap_or_default(),
                    camera_stiffness: camera
                        .map(|value| value.stiffness.to_string())
                        .unwrap_or_default(),
                    camera_swivel: camera
                        .map(|value| value.swivel.to_string())
                        .unwrap_or_default(),
                    camera_transition: camera
                        .and_then(|value| value.transition)
                        .map(|value| value.to_string())
                        .unwrap_or_default(),
                    ..NativePlayerStatsRow::default()
                },
            )
        })
        .collect::<HashMap<_, _>>();

    for row in rows {
        let event_type = row.event_type.as_str();
        let player_1 = row_string(&row.values, "event_player_1_name");
        let player_2 = row_string(&row.values, "event_player_2_name");
        if let Some(stats) = output.get_mut(&player_1) {
            match event_type {
                "shot" => stats.shots += 1,
                "goal" => {
                    stats.goals += 1;
                    stats.shots += 1;
                }
                "save" => stats.saves += 1,
                "touch" => stats.touches += 1,
                "pass" => {
                    stats.passes += 1;
                    stats.touches += 1;
                }
                "turnover" => {
                    stats.turnovers += 1;
                    stats.touches += 1;
                }
                "challenge" => stats.challenges += 1,
                "kickoff" => stats.kickoffs += 1,
                "whiff" => stats.whiffs += 1,
                "fake" => stats.fakes += 1,
                "demo" => stats.demos_applied += 1,
                "bump" => stats.bumps += 1,
                "entry" => stats.entries += 1,
                "exit" => stats.exits += 1,
                "retrieval" => stats.retrievals += 1,
                "missed-shot" => stats.missed_shots += 1,
                "missed-pass" => stats.missed_passes += 1,
                _ => {}
            }
            if matches!(event_type, "shot" | "goal" | "missed-shot") {
                stats.shot_attempts += 1;
            }
        }
        if let Some(stats) = output.get_mut(&player_2) {
            match event_type {
                "goal" => stats.assists += 1,
                "demo" => stats.demos_taken += 1,
                "bump" => stats.bumps_taken += 1,
                _ => {}
            }
        }
    }

    let mut rows = output.into_values().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.team
            .cmp(&right.team)
            .then_with(|| left.player_name.cmp(&right.player_name))
    });
    rows
}

fn team_names_from_context(context: &PbpContext, rows: &[PbpEventRecord]) -> HashMap<i32, String> {
    let mut names = HashMap::new();
    for row in rows {
        let blue = row_string(&row.values, "blue_team_name");
        if !blue.is_empty() {
            names.insert(0, blue);
        }
        let orange = row_string(&row.values, "orange_team_name");
        if !orange.is_empty() {
            names.insert(1, orange);
        }
    }
    for player in &context.players {
        names.entry(player.team).or_default();
    }
    names
}

fn native_time_on_field_minutes(context: &PbpContext, gpu: GpuMode) -> Vec<f32> {
    let player_count = context.players.len();
    if player_count == 0 {
        return Vec::new();
    }

    let (deltas, active) = native_time_on_field_inputs(context);
    if deltas.is_empty() {
        return vec![0.0; player_count];
    }

    let seconds = match gpu {
        GpuMode::None => native_time_on_field_cpu(&deltas, &active, player_count),
        GpuMode::Auto | GpuMode::Rocm | GpuMode::Cuda => {
            native_time_on_field_gpu(&deltas, &active, player_count)
                .unwrap_or_else(|_| native_time_on_field_cpu(&deltas, &active, player_count))
        }
    };

    seconds.into_iter().map(|value| value / 60.0).collect()
}

fn native_time_on_field_inputs(context: &PbpContext) -> (Vec<f32>, Vec<f32>) {
    let player_count = context.players.len();
    let mut deltas = Vec::with_capacity(context.frame_states.len().saturating_sub(1));
    let mut active = Vec::with_capacity(deltas.capacity() * player_count);

    for pair in context.frame_states.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        let delta = match (previous.seconds_elapsed, current.seconds_elapsed) {
            (Some(left), Some(right)) => (right - left).clamp(0.0, 1.0),
            _ => 1.0 / 30.0,
        };
        deltas.push(delta);
        for player_idx in 0..player_count {
            let player_active = previous
                .players
                .get(player_idx)
                .and_then(Option::as_ref)
                .map(|state| state.entity.has_pos)
                .unwrap_or(false);
            active.push(if player_active { 1.0 } else { 0.0 });
        }
    }

    (deltas, active)
}

fn native_time_on_field_cpu(deltas: &[f32], active: &[f32], player_count: usize) -> Vec<f32> {
    let mut totals = vec![0.0; player_count];
    for (interval_idx, delta) in deltas.iter().copied().enumerate() {
        let offset = interval_idx * player_count;
        for player_idx in 0..player_count {
            totals[player_idx] += delta * active[offset + player_idx];
        }
    }
    totals
}

fn native_time_on_field_gpu(
    deltas: &[f32],
    active: &[f32],
    player_count: usize,
) -> Result<Vec<f32>> {
    if deltas.is_empty() || player_count == 0 {
        return Ok(vec![0.0; player_count]);
    }
    if active.len() != deltas.len() * player_count {
        return Err(anyhow!("invalid time-on-field GPU input shape"));
    }

    let contributions = pollster::block_on(native_time_on_field_gpu_contributions(
        deltas,
        active,
        player_count,
    ))?;
    Ok(native_time_on_field_cpu_reduce(
        &contributions,
        deltas.len(),
        player_count,
    ))
}

async fn native_time_on_field_gpu_contributions(
    deltas: &[f32],
    active: &[f32],
    player_count: usize,
) -> Result<Vec<f32>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: wgpu::InstanceFlags::empty(),
        dx12_shader_compiler: Default::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow!("no GPU adapter available"))?;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("analyzerl native stats gpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await?;

    let output_len = active.len();
    let output_size = (output_len * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let delta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("analyzerl time-on-field deltas"),
        contents: bytemuck::cast_slice(deltas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let active_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("analyzerl time-on-field active flags"),
        contents: bytemuck::cast_slice(active),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("analyzerl time-on-field gpu output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("analyzerl time-on-field gpu staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = TimeOnFieldParams {
        interval_count: deltas.len() as u32,
        player_count: player_count as u32,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("analyzerl time-on-field params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("analyzerl time-on-field shader"),
        source: wgpu::ShaderSource::Wgsl(TIME_ON_FIELD_SHADER.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("analyzerl time-on-field bind group layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("analyzerl time-on-field bind group"),
        layout: &bind_group_layout,
        entries: &[
            buffer_entry(0, &delta_buffer),
            buffer_entry(1, &active_buffer),
            buffer_entry(2, &output_buffer),
            buffer_entry(3, &params_buffer),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("analyzerl time-on-field pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("analyzerl time-on-field pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("analyzerl time-on-field encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("analyzerl time-on-field pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups =
            ((output_len as u32) + TIME_ON_FIELD_WORKGROUP_SIZE - 1) / TIME_ON_FIELD_WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .context("waiting for GPU time-on-field map")??;
    let mapped = slice.get_mapped_range();
    let output = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging_buffer.unmap();
    Ok(output)
}

fn native_time_on_field_cpu_reduce(
    contributions: &[f32],
    interval_count: usize,
    player_count: usize,
) -> Vec<f32> {
    let mut totals = vec![0.0; player_count];
    for interval_idx in 0..interval_count {
        let offset = interval_idx * player_count;
        for player_idx in 0..player_count {
            totals[player_idx] += contributions[offset + player_idx];
        }
    }
    totals
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn frames_one(path: &Path, args: &FramesArgs) -> Result<()> {
    let game_id = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow!("bad replay filename: {}", path.display()))?
        .to_string();
    let frames_path = args.out_frames.join(format!(
        "{game_id}_frames.{}",
        export_extension(args.frames_format)
    ));
    if !args.force && frames_path.exists() && frames_path.metadata()?.len() > 0 {
        return Ok(());
    }

    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let replay = boxcars::ParserBuilder::new(&bytes)
        .must_parse_network_data()
        .parse()
        .with_context(|| format!("boxcars parsing {}", path.display()))?;
    if args.parse_only {
        //Benchmarking switch: confirms boxcars parse cost without AnalyzeRL feature generation.
    } else if args.no_write {
        let _ = materialize_frame_rows_for_benchmark(
            &game_id,
            &replay,
            !args.frames_only,
            PbpBuildOptions {
                rotation_events: args.rotation_events,
            },
        )?;
    } else {
        match args.frames_format {
            ExportFormat::Csv => write_frames_csv(
                &frames_path,
                &game_id,
                &replay,
                !args.frames_only,
                PbpBuildOptions {
                    rotation_events: args.rotation_events,
                },
            )?,
            ExportFormat::Parquet => write_frames_parquet(
                &frames_path,
                &game_id,
                &replay,
                !args.frames_only,
                PbpBuildOptions {
                    rotation_events: args.rotation_events,
                },
            )?,
        }
    }
    Ok(())
}

pub fn index_command(args: IndexArgs) -> Result<()> {
    fs::create_dir_all(&args.out_analysis)?;
    if let Some(workers) = args.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .ok();
    }
    let replay_paths = replay_paths(&args.replays)?;
    replay_paths.par_iter().for_each(|path| {
        if let Err(err) = index_one(path, &args.out_analysis) {
            eprintln!("failed header index {}: {err:#}", path.display());
        }
    });
    Ok(())
}

fn index_one(path: &Path, out_analysis: &Path) -> Result<()> {
    let game_id = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow!("bad replay filename: {}", path.display()))?
        .to_string();
    let analysis_path = out_analysis.join(format!("{game_id}_analysis.csv"));
    if analysis_path.exists() {
        return Ok(());
    }
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let replay = boxcars::ParserBuilder::new(&bytes)
        .never_parse_network_data()
        .parse()
        .with_context(|| format!("boxcars header parsing {}", path.display()))?;
    write_analysis_csv(&analysis_path, &game_id, &replay)?;
    Ok(())
}

pub fn match_guid_command(args: MatchGuidArgs) -> Result<()> {
    check_export_args(
        args.export_meta,
        args.export_network,
        args.out_analysis.as_ref(),
        args.out_players.as_ref(),
        args.out_frames.as_ref(),
        args.out_actors.as_ref(),
        args.out_attributes.as_ref(),
    )?;
    if let Some(out_json) = &args.out_json {
        fs::create_dir_all(out_json)?;
    }
    fs::create_dir_all(&args.out_pbp)?;
    if args.export_meta {
        fs::create_dir_all(args.out_analysis.as_ref().unwrap())?;
        fs::create_dir_all(args.out_players.as_ref().unwrap())?;
    }
    if args.export_network {
        fs::create_dir_all(args.out_frames.as_ref().unwrap())?;
        fs::create_dir_all(args.out_actors.as_ref().unwrap())?;
        fs::create_dir_all(args.out_attributes.as_ref().unwrap())?;
    }

    for path in replay_paths(&args.replays)? {
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let replay = boxcars::ParserBuilder::new(&bytes)
            .must_parse_network_data()
            .parse()
            .with_context(|| format!("boxcars parsing {}", path.display()))?;
        let replay_guid = header_string(&replay, "Id").unwrap_or_default();
        if replay_guid.eq_ignore_ascii_case(&args.match_guid) {
            let game_id = path
                .file_stem()
                .and_then(|value| value.to_str())
                .ok_or_else(|| anyhow!("bad replay filename: {}", path.display()))?
                .to_string();
            write_matched_outputs(&game_id, &replay, &args)?;
            println!("{game_id}");
            return Ok(());
        }
    }
    Err(anyhow!("no replay matched guid {}", args.match_guid))
}

fn write_matched_outputs(game_id: &str, replay: &Replay, args: &MatchGuidArgs) -> Result<()> {
    if let Some(out_json) = &args.out_json {
        write_json_atomic(&out_json.join(format!("{game_id}.json")), replay)?;
    }
    if args.export_meta {
        write_analysis_csv(
            &args
                .out_analysis
                .as_ref()
                .unwrap()
                .join(format!("{game_id}_analysis.csv")),
            game_id,
            replay,
        )?;
        write_players_csv(
            &args
                .out_players
                .as_ref()
                .unwrap()
                .join(format!("{game_id}_players.csv")),
            game_id,
            replay,
        )?;
    }
    if args.export_network {
        write_network_csvs(
            &args
                .out_frames
                .as_ref()
                .unwrap()
                .join(format!("{game_id}_frames.csv")),
            &args
                .out_actors
                .as_ref()
                .unwrap()
                .join(format!("{game_id}_actors.csv")),
            &args
                .out_actors
                .as_ref()
                .unwrap()
                .join(format!("{game_id}_deleted_actors.csv")),
            &args
                .out_attributes
                .as_ref()
                .unwrap()
                .join(format!("{game_id}_attributes.csv")),
            game_id,
            replay,
        )?;
    }
    let pbp_path = args.out_pbp.join(format!(
        "{game_id}_pbp.{}",
        export_extension(args.pbp_format)
    ));
    match args.pbp_format {
        ExportFormat::Csv => write_pbp_csv(&pbp_path, game_id, replay, PbpBuildOptions::default())?,
        ExportFormat::Parquet => {
            write_pbp_parquet(&pbp_path, game_id, replay, PbpBuildOptions::default())?
        }
    }
    Ok(())
}

pub fn match_guids_command(args: MatchGuidsArgs) -> Result<()> {
    check_export_args(
        args.export_meta,
        args.export_network,
        args.out_analysis.as_ref(),
        args.out_players.as_ref(),
        args.out_frames.as_ref(),
        args.out_actors.as_ref(),
        args.out_attributes.as_ref(),
    )?;
    if let Some(out_json) = &args.out_json {
        fs::create_dir_all(out_json)?;
    }
    fs::create_dir_all(&args.out_pbp)?;
    if args.export_meta {
        fs::create_dir_all(args.out_analysis.as_ref().unwrap())?;
        fs::create_dir_all(args.out_players.as_ref().unwrap())?;
    }
    if args.export_network {
        fs::create_dir_all(args.out_frames.as_ref().unwrap())?;
        fs::create_dir_all(args.out_actors.as_ref().unwrap())?;
        fs::create_dir_all(args.out_attributes.as_ref().unwrap())?;
    }
    let guid_text = fs::read_to_string(&args.match_guids)?;
    let guids = guid_text
        .lines()
        .map(|line| line.trim().to_uppercase())
        .filter(|line| !line.is_empty())
        .collect::<std::collections::HashSet<_>>();

    for path in replay_paths(&args.replays)? {
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let replay = boxcars::ParserBuilder::new(&bytes)
            .must_parse_network_data()
            .parse()
            .with_context(|| format!("boxcars parsing {}", path.display()))?;
        let replay_guid = header_string(&replay, "Id")
            .unwrap_or_default()
            .to_uppercase();
        if guids.contains(&replay_guid) {
            let game_id = path
                .file_stem()
                .and_then(|value| value.to_str())
                .ok_or_else(|| anyhow!("bad replay filename: {}", path.display()))?
                .to_string();
            if let Some(out_json) = &args.out_json {
                write_json_atomic(&out_json.join(format!("{game_id}.json")), &replay)?;
            }
            if args.export_meta {
                write_analysis_csv(
                    &args
                        .out_analysis
                        .as_ref()
                        .unwrap()
                        .join(format!("{game_id}_analysis.csv")),
                    &game_id,
                    &replay,
                )?;
                write_players_csv(
                    &args
                        .out_players
                        .as_ref()
                        .unwrap()
                        .join(format!("{game_id}_players.csv")),
                    &game_id,
                    &replay,
                )?;
            }
            if args.export_network {
                write_network_csvs(
                    &args
                        .out_frames
                        .as_ref()
                        .unwrap()
                        .join(format!("{game_id}_frames.csv")),
                    &args
                        .out_actors
                        .as_ref()
                        .unwrap()
                        .join(format!("{game_id}_actors.csv")),
                    &args
                        .out_actors
                        .as_ref()
                        .unwrap()
                        .join(format!("{game_id}_deleted_actors.csv")),
                    &args
                        .out_attributes
                        .as_ref()
                        .unwrap()
                        .join(format!("{game_id}_attributes.csv")),
                    &game_id,
                    &replay,
                )?;
            }
            let pbp_path = args.out_pbp.join(format!(
                "{game_id}_pbp.{}",
                export_extension(args.pbp_format)
            ));
            match args.pbp_format {
                ExportFormat::Csv => {
                    write_pbp_csv(&pbp_path, &game_id, &replay, PbpBuildOptions::default())?
                }
                ExportFormat::Parquet => {
                    write_pbp_parquet(&pbp_path, &game_id, &replay, PbpBuildOptions::default())?
                }
            }
            println!("{game_id},{replay_guid}");
            return Ok(());
        }
    }
    Err(anyhow!(
        "no replay matched any guid in {}",
        args.match_guids.display()
    ))
}

fn replay_paths(folder: &Path) -> Result<Vec<PathBuf>> {
    if folder.is_file() {
        if folder.extension().and_then(|value| value.to_str()) == Some("replay") {
            return Ok(vec![folder.to_path_buf()]);
        }
        return Err(anyhow!("not a replay file: {}", folder.display()));
    }
    let mut paths = Vec::new();
    for entry in fs::read_dir(folder).with_context(|| format!("reading {}", folder.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) == Some("replay") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn replay_paths_many(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for input in inputs {
        paths.extend(replay_paths(input)?);
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn parse_one(path: &Path, args: &ParseArgs) -> Result<()> {
    let game_id = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow!("bad replay filename: {}", path.display()))?
        .to_string();
    let json_path = args
        .out_json
        .as_ref()
        .map(|folder| folder.join(format!("{game_id}.json")));
    let pbp_path = args.out_pbp.join(format!(
        "{game_id}_pbp.{}",
        export_extension(args.pbp_format)
    ));
    let analysis_path = args
        .out_analysis
        .as_ref()
        .map(|folder| folder.join(format!("{game_id}_analysis.csv")));
    let players_path = args
        .out_players
        .as_ref()
        .map(|folder| folder.join(format!("{game_id}_players.csv")));
    let frames_path = args
        .out_frames
        .as_ref()
        .map(|folder| folder.join(format!("{game_id}_frames.csv")));
    let actors_path = args
        .out_actors
        .as_ref()
        .map(|folder| folder.join(format!("{game_id}_actors.csv")));
    let deleted_actors_path = args
        .out_actors
        .as_ref()
        .map(|folder| folder.join(format!("{game_id}_deleted_actors.csv")));
    let attributes_path = args
        .out_attributes
        .as_ref()
        .map(|folder| folder.join(format!("{game_id}_attributes.csv")));

    //Skip the expensive network parse if every requested export already exists.
    let json_missing = json_path
        .as_ref()
        .map(|path| args.force || !path.exists())
        .unwrap_or(false);
    let pbp_missing = args.force || !pbp_path.exists();
    let meta_missing = args.export_meta
        && (args.force
            || analysis_path
                .as_ref()
                .map(|path| !path.exists())
                .unwrap_or(true)
            || players_path
                .as_ref()
                .map(|path| !path.exists())
                .unwrap_or(true));
    let network_missing = args.export_network
        && (args.force
            || frames_path
                .as_ref()
                .map(|path| !path.exists())
                .unwrap_or(true)
            || actors_path
                .as_ref()
                .map(|path| !path.exists())
                .unwrap_or(true)
            || deleted_actors_path
                .as_ref()
                .map(|path| !path.exists())
                .unwrap_or(true)
            || attributes_path
                .as_ref()
                .map(|path| !path.exists())
                .unwrap_or(true));

    if !json_missing && !network_missing && !meta_missing && !pbp_missing {
        return Ok(());
    }

    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let replay = boxcars::ParserBuilder::new(&bytes)
        .must_parse_network_data()
        .parse()
        .with_context(|| format!("boxcars parsing {}", path.display()))?;
    if json_missing {
        if let Some(json_path) = &json_path {
            //Full replay JSON is opt-in; default exports avoid oversized frame payloads.
            write_json_atomic(json_path, &replay)?;
        }
    }
    if args.export_meta && meta_missing {
        write_analysis_csv(analysis_path.as_ref().unwrap(), &game_id, &replay)?;
        write_players_csv(players_path.as_ref().unwrap(), &game_id, &replay)?;
    }
    if args.export_network && network_missing {
        write_network_csvs(
            frames_path.as_ref().unwrap(),
            actors_path.as_ref().unwrap(),
            deleted_actors_path.as_ref().unwrap(),
            attributes_path.as_ref().unwrap(),
            &game_id,
            &replay,
        )?;
    }
    if pbp_missing {
        match args.pbp_format {
            ExportFormat::Csv => write_pbp_csv(
                &pbp_path,
                &game_id,
                &replay,
                PbpBuildOptions {
                    rotation_events: args.rotation_events,
                },
            )?,
            ExportFormat::Parquet => write_pbp_parquet(
                &pbp_path,
                &game_id,
                &replay,
                PbpBuildOptions {
                    rotation_events: args.rotation_events,
                },
            )?,
        }
    }
    Ok(())
}

fn replay_game_id(path: &Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn print_progress_line(
    label: &str,
    game_id: &str,
    done: usize,
    total: usize,
    progress_lock: &Arc<Mutex<()>>,
) {
    let _guard = progress_lock.lock().unwrap();
    let mut stdout = std::io::stdout();
    let _ = write!(stdout, "\r\x1b[2K{label} {game_id} ({done}/{total})");
    let _ = stdout.flush();
}

fn finish_progress_line(total: usize, progress_lock: &Arc<Mutex<()>>) {
    if total == 0 {
        return;
    }
    let _guard = progress_lock.lock().unwrap();
    let mut stdout = std::io::stdout();
    let _ = writeln!(stdout);
    let _ = stdout.flush();
}

pub fn animate_json_command(args: Vec<String>) -> Result<()> {
    let mut replay_path = None;
    let mut frame_step = 2usize;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--replay" => replay_path = Some(next_path(&args, &mut idx)?),
            "--frame-step" => frame_step = next_value(&args, &mut idx)?.parse::<usize>()?.max(1),
            flag => return Err(anyhow!("unknown animate-json flag: {flag}")),
        }
        idx += 1;
    }
    let replay_path = replay_path.ok_or_else(|| anyhow!("missing --replay"))?;
    let game_id = replay_path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow!("bad replay filename: {}", replay_path.display()))?
        .to_string();
    let bytes =
        fs::read(&replay_path).with_context(|| format!("reading {}", replay_path.display()))?;
    let replay = boxcars::ParserBuilder::new(&bytes)
        .must_parse_network_data()
        .parse()
        .with_context(|| format!("boxcars parsing {}", replay_path.display()))?;
    let context = pbp_context(&replay);
    let replay_id = header_string(&replay, "Id").unwrap_or_else(|| game_id.clone());
    let replay_name = header_string(&replay, "ReplayName").unwrap_or_default();
    let replay_date = header_string(&replay, "Date").unwrap_or_default();
    let (final_blue_score, final_orange_score) = header_final_score(&replay);
    let pbp_bytes = write_pbp_to_writer(
        csv::Writer::from_writer(Vec::new()),
        &game_id,
        &replay,
        PbpBuildOptions::default(),
    )?;
    let pbp_csv = String::from_utf8(pbp_bytes)?;
    let frames = context
        .frame_states
        .iter()
        .step_by(frame_step)
        .map(|snapshot| {
            let ball = snapshot.ball.map(|state| {
                serde_json::json!({
                    "pos": [state.pos.x, state.pos.y, state.pos.z],
                    "vel": [state.vel.x, state.vel.y, state.vel.z],
                    "ang_vel": [state.ang_vel.x, state.ang_vel.y, state.ang_vel.z],
                    "rot": [state.rot.x, state.rot.y, state.rot.z, state.rot.w],
                })
            });
            let players = context
                .players
                .iter()
                .enumerate()
                .filter_map(|(idx, player)| {
                    let state = snapshot.players.get(idx).and_then(|value| value.as_ref())?;
                    Some(serde_json::json!({
                        "id": player.id.clone(),
                        "name": player.name.clone(),
                        "slot": player.slot.clone(),
                        "team": team_name(player.team),
                        "pos": [state.entity.pos.x, state.entity.pos.y, state.entity.pos.z],
                        "vel": [state.entity.vel.x, state.entity.vel.y, state.entity.vel.z],
                        "ang_vel": [state.entity.ang_vel.x, state.entity.ang_vel.y, state.entity.ang_vel.z],
                        "rot": [state.entity.rot.x, state.entity.rot.y, state.entity.rot.z, state.entity.rot.w],
                        "boost_raw": state.boost,
                        "boost": state.boost.map(|value| boost_units(i32::from(value))),
                        "boost_active": state.boost_active,
                        "boost_collect": state.boost_collect,
                        "throttle": state.throttle,
                        "steer": state.steer,
                        "handbrake": state.handbrake,
                        "ball_cam": state.ball_cam,
                        "dodge_active": state.dodge_active,
                        "jump_active": state.jump_active,
                        "double_jump_active": state.double_jump_active,
                        "jumped": state.jumped,
                        "flipped": state.flipped,
                        "jump_air_activate_count": state.jump_air_activate_count,
                        "double_jump_air_activate_count": state.double_jump_air_activate_count,
                        "dodge_air_activate_count": state.dodge_air_activate_count,
                        "dodges_refreshed_counter": state.dodges_refreshed_counter,
                        "flip_available": state.flip_available,
                        "supersonic": state.supersonic,
                    }))
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "frame_number": snapshot.frame_number,
                "replay_seconds": snapshot.frame_number as f32 / 30.0,
                "seconds_elapsed": snapshot.seconds_elapsed,
                "seconds_remaining": snapshot.seconds_remaining,
                "ball": ball,
                "players": players,
            })
        })
        .collect::<Vec<_>>();
    let payload = serde_json::json!({
        "game_id": game_id,
        "replay_id": replay_id,
        "replay_name": replay_name,
        "replay_date": replay_date,
        "blue_team_name": context.blue_team_name.clone(),
        "orange_team_name": context.orange_team_name.clone(),
        "final_blue_score": final_blue_score,
        "final_orange_score": final_orange_score,
        "frame_step": frame_step,
        "players": context.players.iter().map(|player| {
            serde_json::json!({
                "id": player.id.clone(),
                "name": player.name.clone(),
                "slot": player.slot.clone(),
                "team": team_name(player.team),
                "platform": player.platform.clone(),
                "car_id": player.car_id.clone(),
                "car_name": player.car_name.clone(),
                "camera_settings": player.camera_settings.map(|camera| serde_json::json!({
                    "fov": camera.fov,
                    "height": camera.height,
                    "angle": camera.angle,
                    "distance": camera.distance,
                    "stiffness": camera.stiffness,
                    "swivel": camera.swivel,
                    "transition": camera.transition,
                })),
            })
        }).collect::<Vec<_>>(),
        "pbp_csv": pbp_csv,
        "frames": frames,
    });
    serde_json::to_writer(std::io::stdout(), &payload)?;
    Ok(())
}

fn header_final_score(replay: &Replay) -> (i32, i32) {
    let mut blue_score = 0;
    let mut orange_score = 0;

    if let Some(players) = header_array(replay, "PlayerStats") {
        for player in players {
            let goals = prop_i32(player, "Goals").unwrap_or(0);
            match prop_i32(player, "Team").or_else(|| prop_i32(player, "PlayerTeam")) {
                Some(1) => orange_score += goals,
                Some(0) => blue_score += goals,
                _ => {}
            }
        }
    }

    (blue_score, orange_score)
}

pub fn inspect_flip_command(args: Vec<String>) -> Result<()> {
    let mut replay_path = None;
    let mut start_frame = 0usize;
    let mut end_frame = usize::MAX;
    let mut player_filter = None::<String>;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--replay" => replay_path = Some(next_path(&args, &mut idx)?),
            "--start" => start_frame = next_value(&args, &mut idx)?.parse::<usize>()?,
            "--end" => end_frame = next_value(&args, &mut idx)?.parse::<usize>()?,
            "--player" => player_filter = Some(next_value(&args, &mut idx)?),
            flag => return Err(anyhow!("unknown inspect-flip flag: {flag}")),
        }
        idx += 1;
    }
    let replay_path = replay_path.ok_or_else(|| anyhow!("missing --replay"))?;
    let bytes =
        fs::read(&replay_path).with_context(|| format!("reading {}", replay_path.display()))?;
    let replay = boxcars::ParserBuilder::new(&bytes)
        .must_parse_network_data()
        .parse()
        .with_context(|| format!("boxcars parsing {}", replay_path.display()))?;
    let mut pri_name: HashMap<i32, String> = HashMap::new();
    let mut car_pri: HashMap<i32, i32> = HashMap::new();
    let mut component_vehicle: HashMap<i32, i32> = HashMap::new();
    let mut actor_object_name: HashMap<i32, String> = HashMap::new();
    let mut writer = csv::Writer::from_writer(std::io::stdout());
    writer.write_record([
        "frame_number",
        "actor_id",
        "player_name",
        "component_name",
        "object_name",
        "attribute_type",
        "value",
    ])?;
    if let Some(network_frames) = &replay.network_frames {
        for (frame_number, frame) in network_frames.frames.iter().enumerate() {
            for new_actor in &frame.new_actors {
                actor_object_name.insert(
                    new_actor.actor_id.0,
                    object_name(&replay, new_actor.object_id.0),
                );
            }
            for updated_actor in &frame.updated_actors {
                let name = object_name(&replay, updated_actor.object_id.0);
                match (&name[..], &updated_actor.attribute) {
                    ("Engine.PlayerReplicationInfo:PlayerName", Attribute::String(value)) => {
                        pri_name.insert(updated_actor.actor_id.0, value.clone());
                    }
                    ("Engine.Pawn:PlayerReplicationInfo", Attribute::ActiveActor(value)) => {
                        car_pri.insert(updated_actor.actor_id.0, value.actor.0);
                    }
                    ("TAGame.CarComponent_TA:Vehicle", Attribute::ActiveActor(value)) => {
                        if value.active {
                            component_vehicle.insert(updated_actor.actor_id.0, value.actor.0);
                        }
                    }
                    _ => {}
                }

                if frame_number < start_frame || frame_number > end_frame {
                    continue;
                }
                let component_name = actor_object_name
                    .get(&updated_actor.actor_id.0)
                    .cloned()
                    .unwrap_or_default();
                let interesting = flip_inspect_name(&name)
                    || flip_inspect_name(&component_name)
                    || matches!(
                        attribute_type(&updated_actor.attribute),
                        "int" | "float" | "byte" | "boolean" | "location"
                    ) && (name.contains("Dodge") || name.contains("Jump"));
                if !interesting {
                    continue;
                }
                let player_name = component_vehicle
                    .get(&updated_actor.actor_id.0)
                    .and_then(|car_actor| player_name_for_car(*car_actor, &car_pri, &pri_name))
                    .or_else(|| player_name_for_car(updated_actor.actor_id.0, &car_pri, &pri_name))
                    .or_else(|| pri_name.get(&updated_actor.actor_id.0).cloned())
                    .unwrap_or_default();
                if let Some(filter) = &player_filter {
                    if !player_name.eq_ignore_ascii_case(filter) {
                        continue;
                    }
                }
                writer.write_record([
                    frame_number.to_string(),
                    updated_actor.actor_id.0.to_string(),
                    player_name,
                    component_name,
                    name,
                    attribute_type(&updated_actor.attribute).to_string(),
                    inspect_attribute_value(&updated_actor.attribute),
                ])?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

fn flip_inspect_name(name: &str) -> bool {
    name.contains("Dodge")
        || name.contains("Jump")
        || name.contains("Flip")
        || name.contains("AirActivate")
        || name.contains("DodgesRefreshedCounter")
        || name.contains("ReplicatedActive")
        || name.contains("ReplicatedActivityTime")
}

fn inspect_attribute_value(attribute: &Attribute) -> String {
    match attribute {
        Attribute::Boolean(value) => value.to_string(),
        Attribute::Byte(value) => value.to_string(),
        Attribute::Float(value) => value.to_string(),
        Attribute::Int(value) => value.to_string(),
        Attribute::Int64(value) => value.to_string(),
        Attribute::String(value) => value.clone(),
        Attribute::Location(value) => format!("{},{},{}", value.x, value.y, value.z),
        Attribute::ActiveActor(value) => format!("active={},actor={}", value.active, value.actor.0),
        _ => attribute_type(attribute).to_string(),
    }
}

fn write_json_atomic(path: &Path, replay: &Replay) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer(BufWriter::new(file), replay)?;
    Ok(())
}

fn write_analysis_csv(path: &Path, game_id: &str, replay: &Replay) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    let frame_count = replay
        .network_frames
        .as_ref()
        .map(|frames| frames.frames.len())
        .unwrap_or(0);
    writer.serialize(AnalysisRow {
        game_id: game_id.to_string(),
        parse_status: "parsed".to_string(),
        frame_count,
        keyframe_count: replay.keyframes.len(),
        tick_mark_count: replay.tick_marks.len(),
        object_count: replay.objects.len(),
        name_count: replay.names.len(),
        class_index_count: replay.class_indices.len(),
        net_cache_count: replay.net_cache.len(),
        major_version: replay.major_version,
        minor_version: replay.minor_version,
        net_version: replay.net_version,
        game_type: replay.game_type.clone(),
        id: header_string(replay, "Id").unwrap_or_default(),
        replay_name: header_string(replay, "ReplayName").unwrap_or_default(),
        map_name: header_string(replay, "MapName").unwrap_or_default(),
        date: header_string(replay, "Date").unwrap_or_default(),
        match_type: header_string(replay, "MatchType").unwrap_or_default(),
        team_size: header_actual_team_size(replay).or_else(|| header_i32(replay, "TeamSize")),
        playlist: header_string(replay, "Playlist").unwrap_or_default(),
    })?;
    writer.flush()?;
    Ok(())
}

fn write_network_csvs(
    frames_path: &Path,
    actors_path: &Path,
    deleted_actors_path: &Path,
    attributes_path: &Path,
    game_id: &str,
    replay: &Replay,
) -> Result<()> {
    let mut frame_writer = csv::Writer::from_path(frames_path)?;
    let mut actor_writer = csv::Writer::from_path(actors_path)?;
    let mut deleted_actor_writer = csv::Writer::from_path(deleted_actors_path)?;
    let mut attribute_writer = csv::Writer::from_path(attributes_path)?;
    if let Some(network_frames) = &replay.network_frames {
        for (frame_number, frame) in network_frames.frames.iter().enumerate() {
            frame_writer.serialize(FrameRow {
                game_id: game_id.to_string(),
                frame_number,
                time: frame.time,
                delta: frame.delta,
                new_actor_count: frame.new_actors.len(),
                deleted_actor_count: frame.deleted_actors.len(),
                updated_actor_count: frame.updated_actors.len(),
            })?;
            for new_actor in &frame.new_actors {
                let trajectory = new_actor.initial_trajectory;
                let location = trajectory.location;
                let rotation = trajectory.rotation;
                actor_writer.serialize(ActorRow {
                    game_id: game_id.to_string(),
                    frame_number,
                    time: frame.time,
                    actor_id: new_actor.actor_id.0,
                    name_id: new_actor.name_id,
                    actor_name: new_actor
                        .name_id
                        .and_then(|idx| replay.names.get(idx as usize))
                        .cloned()
                        .unwrap_or_default(),
                    object_id: new_actor.object_id.0,
                    object_name: object_name(replay, new_actor.object_id.0),
                    spawn_location_x: location.map(|value| value.x),
                    spawn_location_y: location.map(|value| value.y),
                    spawn_location_z: location.map(|value| value.z),
                    spawn_rotation_yaw: rotation.and_then(|value| value.yaw),
                    spawn_rotation_pitch: rotation.and_then(|value| value.pitch),
                    spawn_rotation_roll: rotation.and_then(|value| value.roll),
                })?;
            }
            for actor_id in &frame.deleted_actors {
                deleted_actor_writer.serialize(DeletedActorRow {
                    game_id: game_id.to_string(),
                    frame_number,
                    time: frame.time,
                    actor_id: actor_id.0,
                })?;
            }
            for updated_actor in &frame.updated_actors {
                attribute_writer.serialize(attribute_row(
                    game_id,
                    frame_number,
                    frame.time,
                    replay,
                    updated_actor,
                )?)?;
            }
        }
    }
    frame_writer.flush()?;
    actor_writer.flush()?;
    deleted_actor_writer.flush()?;
    attribute_writer.flush()?;
    Ok(())
}

fn object_name(replay: &Replay, object_id: i32) -> String {
    replay
        .objects
        .get(object_id as usize)
        .cloned()
        .unwrap_or_default()
}

fn attribute_row(
    game_id: &str,
    frame_number: usize,
    time: f32,
    replay: &Replay,
    updated_actor: &UpdatedAttribute,
) -> Result<AttributeRow> {
    let mut row = AttributeRow {
        game_id: game_id.to_string(),
        frame_number,
        time,
        actor_id: updated_actor.actor_id.0,
        stream_id: updated_actor.stream_id.0,
        object_id: updated_actor.object_id.0,
        object_name: object_name(replay, updated_actor.object_id.0),
        attribute_type: attribute_type(&updated_actor.attribute).to_string(),
        bool_value: None,
        int_value: None,
        int64_value: None,
        float_value: None,
        string_value: String::new(),
        byte_value: None,
        active_actor_active: None,
        active_actor_id: None,
        location_x: None,
        location_y: None,
        location_z: None,
        rotation_x: None,
        rotation_y: None,
        rotation_z: None,
        rotation_w: None,
        linear_velocity_x: None,
        linear_velocity_y: None,
        linear_velocity_z: None,
        angular_velocity_x: None,
        angular_velocity_y: None,
        angular_velocity_z: None,
        boost_amount: None,
        boost_grant_count: None,
        demolish_attacker_id: None,
        demolish_victim_id: None,
        stat_event_object_id: None,
    };

    match &updated_actor.attribute {
        Attribute::Boolean(value) => row.bool_value = Some(*value),
        Attribute::Byte(value) => row.byte_value = Some(*value),
        Attribute::Float(value) => row.float_value = Some(*value),
        Attribute::Int(value) => row.int_value = Some(*value),
        Attribute::Int64(value) => row.int64_value = Some(value.to_string()),
        Attribute::String(value) => row.string_value = value.clone(),
        Attribute::ActiveActor(value) => {
            row.active_actor_active = Some(value.active);
            row.active_actor_id = Some(value.actor.0);
        }
        Attribute::Location(value) => {
            row.location_x = Some(value.x);
            row.location_y = Some(value.y);
            row.location_z = Some(value.z);
        }
        Attribute::RigidBody(value) => {
            row.location_x = Some(value.location.x);
            row.location_y = Some(value.location.y);
            row.location_z = Some(value.location.z);
            row.rotation_x = Some(value.rotation.x);
            row.rotation_y = Some(value.rotation.y);
            row.rotation_z = Some(value.rotation.z);
            row.rotation_w = Some(value.rotation.w);
            if let Some(linear_velocity) = value.linear_velocity {
                row.linear_velocity_x = Some(linear_velocity.x);
                row.linear_velocity_y = Some(linear_velocity.y);
                row.linear_velocity_z = Some(linear_velocity.z);
            }
            if let Some(angular_velocity) = value.angular_velocity {
                row.angular_velocity_x = Some(angular_velocity.x);
                row.angular_velocity_y = Some(angular_velocity.y);
                row.angular_velocity_z = Some(angular_velocity.z);
            }
        }
        Attribute::ReplicatedBoost(value) => {
            row.boost_grant_count = Some(value.grant_count);
            row.boost_amount = Some(value.boost_amount);
        }
        Attribute::Demolish(value) => {
            row.demolish_attacker_id = Some(value.attacker.0);
            row.demolish_victim_id = Some(value.victim.0);
        }
        Attribute::DemolishFx(value) => {
            row.demolish_attacker_id = Some(value.attacker.0);
            row.demolish_victim_id = Some(value.victim.0);
        }
        Attribute::DemolishExtended(value) => {
            row.demolish_attacker_id = Some(value.attacker.actor.0);
            row.demolish_victim_id = Some(value.victim.actor.0);
        }
        Attribute::StatEvent(value) => row.stat_event_object_id = Some(value.object_id),
        _ => {}
    }
    Ok(row)
}

fn attribute_type(attribute: &Attribute) -> &'static str {
    match attribute {
        Attribute::Boolean(_) => "boolean",
        Attribute::Byte(_) => "byte",
        Attribute::AppliedDamage(_) => "applied_damage",
        Attribute::DamageState(_) => "damage_state",
        Attribute::CamSettings(_) => "cam_settings",
        Attribute::ClubColors(_) => "club_colors",
        Attribute::Demolish(_) => "demolish",
        Attribute::DemolishExtended(_) => "demolish_extended",
        Attribute::DemolishFx(_) => "demolish_fx",
        Attribute::Enum(_) => "enum",
        Attribute::Explosion(_) => "explosion",
        Attribute::ExtendedExplosion(_) => "extended_explosion",
        Attribute::FlaggedByte(_, _) => "flagged_byte",
        Attribute::ActiveActor(_) => "active_actor",
        Attribute::Float(_) => "float",
        Attribute::GameMode(_, _) => "game_mode",
        Attribute::Int(_) => "int",
        Attribute::Int64(_) => "int64",
        Attribute::Loadout(_) => "loadout",
        Attribute::TeamLoadout(_) => "team_loadout",
        Attribute::Location(_) => "location",
        Attribute::MusicStinger(_) => "music_stinger",
        Attribute::PlayerHistoryKey(_) => "player_history_key",
        Attribute::Pickup(_) => "pickup",
        Attribute::PickupNew(_) => "pickup_new",
        Attribute::QWord(_) => "qword",
        Attribute::Welded(_) => "welded",
        Attribute::Title(_, _, _, _, _, _, _, _) => "title",
        Attribute::TeamPaint(_) => "team_paint",
        Attribute::RigidBody(_) => "rigid_body",
        Attribute::String(_) => "string",
        Attribute::UniqueId(_) => "unique_id",
        Attribute::Reservation(_) => "reservation",
        Attribute::PartyLeader(_) => "party_leader",
        Attribute::PrivateMatch(_) => "private_match",
        Attribute::LoadoutOnline(_) => "loadout_online",
        Attribute::LoadoutsOnline(_) => "loadouts_online",
        Attribute::StatEvent(_) => "stat_event",
        Attribute::Rotation(_) => "rotation",
        Attribute::RepStatTitle(_) => "rep_stat_title",
        Attribute::PickupInfo(_) => "pickup_info",
        Attribute::Impulse(_) => "impulse",
        Attribute::ReplicatedBoost(_) => "replicated_boost",
        Attribute::LogoData(_) => "logo_data",
    }
}

fn write_players_csv(path: &Path, game_id: &str, replay: &Replay) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    if let Some(players) = header_array(replay, "PlayerStats") {
        for player in players {
            writer.serialize(PlayerRow {
                game_id: game_id.to_string(),
                player_name: prop_string(player, "Name")
                    .or_else(|| prop_string(player, "PlayerName"))
                    .unwrap_or_default(),
                player_team: prop_i32(player, "Team").or_else(|| prop_i32(player, "PlayerTeam")),
                online_id: prop_string(player, "OnlineID").unwrap_or_default(),
                platform: prop_string(player, "Platform").unwrap_or_default(),
                score: prop_i32(player, "Score"),
                goals: prop_i32(player, "Goals"),
                assists: prop_i32(player, "Assists"),
                saves: prop_i32(player, "Saves"),
                shots: prop_i32(player, "Shots"),
                b_bot: prop_bool(player, "bBot"),
            })?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_pbp_csv(
    path: &Path,
    game_id: &str,
    replay: &Replay,
    options: PbpBuildOptions,
) -> Result<()> {
    write_pbp_to_writer(csv::Writer::from_path(path)?, game_id, replay, options).map(|_| ())
}

fn write_pbp_to_writer<W: Write + Send + Sync + 'static>(
    mut writer: csv::Writer<W>,
    game_id: &str,
    replay: &Replay,
    options: PbpBuildOptions,
) -> Result<W> {
    let columns = pbp_columns_cached();
    let (_, rows) = build_pbp_rows(game_id, replay, options)?;
    let export_indexes = pbp_export_column_indexes_from_records(&rows);
    let export_columns = export_indexes
        .iter()
        .map(|idx| columns[*idx].as_str())
        .collect::<Vec<_>>();
    writer.write_record(export_columns)?;
    let static_defaults = vec![None; export_indexes.len()];
    let mut record = csv::StringRecord::new();
    for row in &rows {
        let values = project_cells(row.values.as_slice(), &export_indexes);
        write_csv_row(&mut writer, &values, &static_defaults, &mut record)?;
    }

    writer.flush()?;
    Ok(writer.into_inner()?)
}

fn write_pbp_parquet(
    path: &Path,
    game_id: &str,
    replay: &Replay,
    options: PbpBuildOptions,
) -> Result<()> {
    let (_, pbp_rows) = build_pbp_rows(game_id, replay, options)?;
    let mut rows = Vec::with_capacity(pbp_rows.len());

    for pbp_row in pbp_rows {
        rows.push(pbp_row.values.into_cells());
    }

    let export_indexes = pbp_export_column_indexes_from_cells(&rows, pbp_columns_cached().len());
    let columns = export_indexes
        .iter()
        .map(|idx| pbp_columns_cached()[*idx].clone())
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|column| Field::new(column, arrow_data_type(column_kind(column)), true))
            .collect::<Vec<_>>(),
    ));
    let column_kinds = columns
        .iter()
        .map(|column| column_kind(column))
        .collect::<Vec<_>>();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    if !rows.is_empty() {
        let static_defaults = vec![None; columns.len()];
        let rows = rows
            .iter()
            .map(|row| project_cells(row, &export_indexes))
            .collect::<Vec<_>>();
        write_arrow_batch(&mut writer, schema, &column_kinds, &rows, &static_defaults)?;
    }
    writer.close()?;
    Ok(())
}

fn write_frames_csv(
    path: &Path,
    game_id: &str,
    replay: &Replay,
    include_events: bool,
    options: PbpBuildOptions,
) -> Result<()> {
    let columns = frame_columns_cached();
    let column_kinds = frame_column_kinds_cached();
    let column_index = frame_column_index_cached();

    let (context, pbp_rows) = if include_events {
        build_pbp_rows(game_id, replay, options)?
    } else {
        (frame_context(replay), Vec::new())
    };
    let mut events_by_frame: HashMap<i32, Vec<RowValues>> = HashMap::new();
    for row in pbp_rows {
        let frame = row_i32(&row.values, "observed_frame_number")
            .or_else(|| row_i32(&row.values, "frame_number"));
        if let Some(frame) = frame {
            events_by_frame.entry(frame).or_default().push(row.values);
        }
    }

    let players = context.players.clone();
    let event_model = EventModel::from_frames(&context);
    let team_size = actual_team_size(&players).or_else(|| header_i32(replay, "TeamSize"));
    let player_static_values = pbp_player_static_values(&players);
    let frame_indexes = frame_row_indexes(column_index, &players);
    let mut static_row = vec![None; columns.len()];
    set_row_value(
        &mut static_row,
        &column_index,
        "game_id",
        game_id.to_string(),
    );
    set_row_value(
        &mut static_row,
        &column_index,
        "blue_team_name",
        context.blue_team_name.clone(),
    );
    set_row_value(
        &mut static_row,
        &column_index,
        "orange_team_name",
        context.orange_team_name.clone(),
    );
    if let Some(size) = team_size {
        set_row_i32(&mut static_row, &column_index, "team_size", size);
    }
    for (key, value) in &player_static_values {
        if let Some(idx) = column_index.get(key) {
            static_row[*idx] = parse_cell_value(column_kinds[*idx], value);
        }
    }

    let mut rows = Vec::new();
    for snapshot in &context.frame_states {
        let mut base = vec![None; columns.len()];
        set_idx_i32(&mut base, frame_indexes.frame_number, snapshot.frame_number);
        set_idx_i32(
            &mut base,
            frame_indexes.observed_frame_number,
            snapshot.frame_number,
        );
        if let Some(seconds) = snapshot.seconds_elapsed {
            set_idx_f32(&mut base, frame_indexes.seconds_elapsed, seconds);
        }
        if let Some(stint_number) = event_model.stint_number_for_frame(snapshot.frame_number) {
            set_row_i32(&mut base, column_index, "stint_number", stint_number);
        }
        add_frame_state_values_row_indexed(&mut base, &frame_indexes, snapshot);
        add_spatial_features_row_indexed(&mut base, &frame_indexes, snapshot, &players);

        if let Some(events) = events_by_frame.get(&snapshot.frame_number) {
            for event in events {
                let mut values = base.clone();
                overlay_event_values(&mut values, event);
                set_idx_bool(&mut values, frame_indexes.frame_has_event, true);
                set_idx_i32(
                    &mut values,
                    frame_indexes.frame_event_count,
                    events.len() as i32,
                );
                rows.push(values);
            }
        } else {
            set_idx_bool(&mut base, frame_indexes.frame_has_event, false);
            set_idx_i32(&mut base, frame_indexes.frame_event_count, 0);
            rows.push(base);
        }
    }

    let export_indexes =
        export_column_indexes_from_cells(&rows, &static_row, frame_columns_cached().len());
    let export_columns = export_indexes
        .iter()
        .map(|idx| columns[*idx].as_str())
        .collect::<Vec<_>>();
    let static_row = project_cells(&static_row, &export_indexes);
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = csv::Writer::from_writer(file);
    writer.write_record(export_columns)?;
    let mut record = csv::StringRecord::new();
    for row in rows {
        let row = project_cells(&row, &export_indexes);
        write_csv_row(&mut writer, &row, &static_row, &mut record)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_frames_parquet(
    path: &Path,
    game_id: &str,
    replay: &Replay,
    include_events: bool,
    options: PbpBuildOptions,
) -> Result<()> {
    let columns = frame_columns_cached();
    let column_kinds = frame_column_kinds_cached();

    let (context, pbp_rows) = if include_events {
        build_pbp_rows(game_id, replay, options)?
    } else {
        (frame_context(replay), Vec::new())
    };
    let mut events_by_frame: HashMap<i32, Vec<RowValues>> = HashMap::new();
    for row in pbp_rows {
        let frame = row_i32(&row.values, "observed_frame_number")
            .or_else(|| row_i32(&row.values, "frame_number"));
        if let Some(frame) = frame {
            events_by_frame.entry(frame).or_default().push(row.values);
        }
    }

    let players = context.players.clone();
    let event_model = EventModel::from_frames(&context);
    let team_size = actual_team_size(&players).or_else(|| header_i32(replay, "TeamSize"));
    let player_static_values = pbp_player_static_values(&players);
    let column_index = frame_column_index_cached();
    let frame_indexes = frame_row_indexes(column_index, &players);
    let mut static_row = vec![None; columns.len()];
    set_row_value(
        &mut static_row,
        &column_index,
        "game_id",
        game_id.to_string(),
    );
    set_row_value(
        &mut static_row,
        &column_index,
        "blue_team_name",
        context.blue_team_name.clone(),
    );
    set_row_value(
        &mut static_row,
        &column_index,
        "orange_team_name",
        context.orange_team_name.clone(),
    );
    if let Some(size) = team_size {
        set_row_i32(&mut static_row, &column_index, "team_size", size);
    }
    for (key, value) in &player_static_values {
        if let Some(idx) = column_index.get(key) {
            static_row[*idx] = parse_cell_value(column_kinds[*idx], value);
        }
    }
    let mut rows = Vec::new();

    for snapshot in &context.frame_states {
        let mut base = vec![None; columns.len()];
        set_idx_i32(&mut base, frame_indexes.frame_number, snapshot.frame_number);
        set_idx_i32(
            &mut base,
            frame_indexes.observed_frame_number,
            snapshot.frame_number,
        );
        if let Some(seconds) = snapshot.seconds_elapsed {
            set_idx_f32(&mut base, frame_indexes.seconds_elapsed, seconds);
        }
        if let Some(stint_number) = event_model.stint_number_for_frame(snapshot.frame_number) {
            set_row_i32(&mut base, column_index, "stint_number", stint_number);
        }
        add_frame_state_values_row_indexed(&mut base, &frame_indexes, snapshot);
        add_spatial_features_row_indexed(&mut base, &frame_indexes, snapshot, &players);

        if let Some(events) = events_by_frame.get(&snapshot.frame_number) {
            for event in events {
                let mut values = base.clone();
                overlay_event_values(&mut values, event);
                set_idx_bool(&mut values, frame_indexes.frame_has_event, true);
                set_idx_i32(
                    &mut values,
                    frame_indexes.frame_event_count,
                    events.len() as i32,
                );
                rows.push(values);
            }
        } else {
            set_idx_bool(&mut base, frame_indexes.frame_has_event, false);
            set_idx_i32(&mut base, frame_indexes.frame_event_count, 0);
            rows.push(base);
        }
    }

    let export_indexes =
        export_column_indexes_from_cells(&rows, &static_row, frame_columns_cached().len());
    let columns = export_indexes
        .iter()
        .map(|idx| columns[*idx].clone())
        .collect::<Vec<_>>();
    let column_kinds = export_indexes
        .iter()
        .map(|idx| column_kinds[*idx])
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|column| Field::new(column, arrow_data_type(column_kind(column)), true))
            .collect::<Vec<_>>(),
    ));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
    let static_row = project_cells(&static_row, &export_indexes);
    if !rows.is_empty() {
        let rows = rows
            .iter()
            .map(|row| project_cells(row, &export_indexes))
            .collect::<Vec<_>>();
        for chunk in rows.chunks(FRAME_PARQUET_ROW_GROUP_SIZE) {
            write_arrow_batch(
                &mut writer,
                schema.clone(),
                &column_kinds,
                chunk,
                &static_row,
            )?;
        }
    }
    writer.close()?;
    Ok(())
}

fn materialize_frame_rows_for_benchmark(
    game_id: &str,
    replay: &Replay,
    include_events: bool,
    options: PbpBuildOptions,
) -> Result<(usize, usize)> {
    let columns = frame_columns_cached();
    let column_index = frame_column_index_cached();
    let column_kinds = frame_column_kinds_cached();
    let (context, pbp_rows) = if include_events {
        build_pbp_rows(game_id, replay, options)?
    } else {
        (frame_context(replay), Vec::new())
    };
    let mut events_by_frame: HashMap<i32, Vec<RowValues>> = HashMap::new();
    for row in pbp_rows {
        let frame = row_i32(&row.values, "observed_frame_number")
            .or_else(|| row_i32(&row.values, "frame_number"));
        if let Some(frame) = frame {
            events_by_frame.entry(frame).or_default().push(row.values);
        }
    }

    let players = context.players.clone();
    let team_size = actual_team_size(&players).or_else(|| header_i32(replay, "TeamSize"));
    let player_static_values = pbp_player_static_values(&players);
    let frame_indexes = frame_row_indexes(column_index, &players);
    let mut static_row = vec![None; columns.len()];
    set_row_value(
        &mut static_row,
        column_index,
        "game_id",
        game_id.to_string(),
    );
    set_row_value(
        &mut static_row,
        column_index,
        "blue_team_name",
        context.blue_team_name.clone(),
    );
    set_row_value(
        &mut static_row,
        column_index,
        "orange_team_name",
        context.orange_team_name.clone(),
    );
    if let Some(size) = team_size {
        set_row_i32(&mut static_row, column_index, "team_size", size);
    }
    for (key, value) in &player_static_values {
        if let Some(idx) = column_index.get(key) {
            static_row[*idx] = parse_cell_value(column_kinds[*idx], value);
        }
    }

    let mut row_count = 0usize;
    let mut filled_count = 0usize;
    for snapshot in &context.frame_states {
        let mut base = vec![None; columns.len()];
        set_idx_i32(&mut base, frame_indexes.frame_number, snapshot.frame_number);
        set_idx_i32(
            &mut base,
            frame_indexes.observed_frame_number,
            snapshot.frame_number,
        );
        if let Some(seconds) = snapshot.seconds_elapsed {
            set_idx_f32(&mut base, frame_indexes.seconds_elapsed, seconds);
        }
        add_frame_state_values_row_indexed(&mut base, &frame_indexes, snapshot);
        add_spatial_features_row_indexed(&mut base, &frame_indexes, snapshot, &players);

        if let Some(events) = events_by_frame.get(&snapshot.frame_number) {
            for event in events {
                let mut values = base.clone();
                overlay_event_values(&mut values, event);
                set_idx_bool(&mut values, frame_indexes.frame_has_event, true);
                set_idx_i32(
                    &mut values,
                    frame_indexes.frame_event_count,
                    events.len() as i32,
                );
                row_count += 1;
                filled_count += filled_cell_count(&values, &static_row);
            }
        } else {
            set_idx_bool(&mut base, frame_indexes.frame_has_event, false);
            set_idx_i32(&mut base, frame_indexes.frame_event_count, 0);
            row_count += 1;
            filled_count += filled_cell_count(&base, &static_row);
        }
    }

    Ok((row_count, filled_count))
}

fn write_csv_row<W: Write>(
    writer: &mut csv::Writer<W>,
    row: &[Option<CellValue>],
    static_defaults: &[Option<CellValue>],
    record: &mut csv::StringRecord,
) -> Result<()> {
    record.clear();
    for idx in 0..row.len() {
        match cell_at(row, static_defaults, idx) {
            Some(CellValue::Utf8(value)) => record.push_field(value),
            Some(CellValue::Int32(value)) => record.push_field(&value.to_string()),
            Some(CellValue::Float32(value)) => record.push_field(&value.to_string()),
            Some(CellValue::Boolean(value)) => record.push_field(&value.to_string()),
            None => record.push_field(""),
        }
    }
    writer.write_record(record.iter())?;
    Ok(())
}

fn filled_cell_count(row: &[Option<CellValue>], static_defaults: &[Option<CellValue>]) -> usize {
    (0..row.len())
        .filter(|idx| cell_at(row, static_defaults, *idx).is_some())
        .count()
}

fn pbp_export_column_indexes_from_records(rows: &[PbpEventRecord]) -> Vec<usize> {
    let column_count = pbp_columns_cached().len();
    if rows.is_empty() {
        return (0..column_count).collect();
    }

    let mut has_value = vec![false; column_count];
    for row in rows {
        mark_export_column_values(row.values.as_slice(), &mut has_value);
    }
    non_empty_export_indexes(has_value)
}

fn pbp_export_column_indexes_from_cells(
    rows: &[Vec<Option<CellValue>>],
    column_count: usize,
) -> Vec<usize> {
    if rows.is_empty() {
        return (0..column_count).collect();
    }

    let mut has_value = vec![false; column_count];
    for row in rows {
        mark_export_column_values(row, &mut has_value);
    }
    non_empty_export_indexes(has_value)
}

fn export_column_indexes_from_cells(
    rows: &[Vec<Option<CellValue>>],
    static_defaults: &[Option<CellValue>],
    column_count: usize,
) -> Vec<usize> {
    if rows.is_empty() {
        return (0..column_count).collect();
    }

    let mut has_value = vec![false; column_count];
    for row in rows {
        for idx in 0..column_count {
            if export_cell_has_value(cell_at(row, static_defaults, idx)) {
                has_value[idx] = true;
            }
        }
    }
    non_empty_export_indexes(has_value)
}

fn mark_export_column_values(row: &[Option<CellValue>], has_value: &mut [bool]) {
    for (idx, cell) in row.iter().enumerate() {
        if idx < has_value.len() && export_cell_has_value(cell.as_ref()) {
            has_value[idx] = true;
        }
    }
}

fn non_empty_export_indexes(has_value: Vec<bool>) -> Vec<usize> {
    let indexes = has_value
        .into_iter()
        .enumerate()
        .filter_map(|(idx, has_value)| has_value.then_some(idx))
        .collect::<Vec<_>>();
    if indexes.is_empty() {
        (0..pbp_columns_cached().len()).collect()
    } else {
        indexes
    }
}

fn export_cell_has_value(cell: Option<&CellValue>) -> bool {
    match cell {
        Some(CellValue::Utf8(value)) => {
            let value = value.trim();
            !value.is_empty() && !value.eq_ignore_ascii_case("nan")
        }
        Some(CellValue::Int32(_)) | Some(CellValue::Boolean(_)) => true,
        Some(CellValue::Float32(value)) => value.is_finite(),
        None => false,
    }
}

fn project_cells(row: &[Option<CellValue>], indexes: &[usize]) -> Vec<Option<CellValue>> {
    indexes
        .iter()
        .map(|idx| row.get(*idx).cloned().unwrap_or(None))
        .collect()
}

fn pbp_columns_cached() -> &'static Vec<String> {
    static PBP_COLUMNS: OnceLock<Vec<String>> = OnceLock::new();
    PBP_COLUMNS.get_or_init(pbp_columns)
}

fn pbp_column_index_cached() -> &'static HashMap<String, usize> {
    static PBP_COLUMN_INDEX: OnceLock<HashMap<String, usize>> = OnceLock::new();
    PBP_COLUMN_INDEX.get_or_init(|| column_index(pbp_columns_cached()))
}

fn pbp_column_kinds_cached() -> &'static Vec<ColumnKind> {
    static PBP_COLUMN_KINDS: OnceLock<Vec<ColumnKind>> = OnceLock::new();
    PBP_COLUMN_KINDS.get_or_init(|| {
        pbp_columns_cached()
            .iter()
            .map(|column| column_kind(column))
            .collect()
    })
}

fn frame_columns_cached() -> &'static Vec<String> {
    static FRAME_COLUMNS: OnceLock<Vec<String>> = OnceLock::new();
    FRAME_COLUMNS.get_or_init(|| {
        let mut columns = pbp_columns();
        for column in ["frame_has_event", "frame_event_count"] {
            if !columns.iter().any(|existing| existing == column) {
                columns.push(column.to_string());
            }
        }
        columns
    })
}

fn frame_column_index_cached() -> &'static HashMap<String, usize> {
    static FRAME_COLUMN_INDEX: OnceLock<HashMap<String, usize>> = OnceLock::new();
    FRAME_COLUMN_INDEX.get_or_init(|| column_index(frame_columns_cached()))
}

fn frame_column_kinds_cached() -> &'static Vec<ColumnKind> {
    static FRAME_COLUMN_KINDS: OnceLock<Vec<ColumnKind>> = OnceLock::new();
    FRAME_COLUMN_KINDS.get_or_init(|| {
        frame_columns_cached()
            .iter()
            .map(|column| column_kind(column))
            .collect()
    })
}

fn column_index(columns: &[String]) -> HashMap<String, usize> {
    columns
        .iter()
        .enumerate()
        .map(|(idx, column)| (column.clone(), idx))
        .collect()
}

fn arrow_data_type(kind: ColumnKind) -> DataType {
    match kind {
        ColumnKind::Utf8 => DataType::Utf8,
        ColumnKind::Int32 => DataType::Int32,
        ColumnKind::Float32 => DataType::Float32,
        ColumnKind::Boolean => DataType::Boolean,
    }
}

fn column_kind(column: &str) -> ColumnKind {
    if boolean_column(column) {
        ColumnKind::Boolean
    } else if int_column(column) {
        ColumnKind::Int32
    } else if utf8_column(column) {
        ColumnKind::Utf8
    } else {
        ColumnKind::Float32
    }
}

fn utf8_column(column: &str) -> bool {
    column == "game_id"
        || column.ends_with("_id")
        || column.ends_with("_name")
        || column.ends_with("_team")
        || column.ends_with("_type")
        || column.ends_with("_platform")
        || column.ends_with("_car_name")
        || column == "event_type"
        || column == "event_team"
        || column == "previous_event_type"
        || column == "kickoff_type"
        || column == "boost_pickup_type"
        || column == "reset_origin"
        || column == "blue_team_name"
        || column == "orange_team_name"
}

fn boolean_column(column: &str) -> bool {
    column == "controlled"
        || column == "frame_has_event"
        || (column.starts_with("official_") && !column.ends_with("_count"))
        || column.starts_with("off_")
        || column.ends_with("_active")
        || column.ends_with("_cam")
        || column.ends_with("_handbrake")
        || column.ends_with("_jumped")
        || column.ends_with("_flipped")
        || column.ends_with("_supersonic")
        || column.ends_with("_is_bot")
        || column.ends_with("_pro_player")
        || column.ends_with("_demolished")
        || column.ends_with("_projected_inside_net")
        || column.ends_with("_on_field")
        || column.ends_with("_powersliding")
        || column.ends_with("_boosting")
        || column.ends_with("_flipping")
        || column.ends_with("_intercept_possible")
        || column.ends_with("_intercept_requires_aerial")
        || matches!(
            column,
            "off_pass"
                | "off_fake"
                | "off_whiff"
                | "off_rotation_cut"
                | "aerialing"
                | "air_dribble"
                | "ground_dribble"
                | "flick_shot"
                | "rebound"
                | "double_tap"
                | "flip-reset"
                | "previous_event_entry"
                | "previous_event_exit"
        )
}

fn int_column(column: &str) -> bool {
    column == "team_size"
        || column == "event_number"
        || column == "frame_number"
        || column == "observed_frame_number"
        || column == "recorded_frame_number"
        || column == "blue_score"
        || column == "orange_score"
        || column == "goal_number"
        || column == "boost_pickup_amount"
        || column == "frame_event_count"
        || (column.starts_with("official_") && column.ends_with("_count"))
        || column.ends_with("_frame_number")
        || column.ends_with("_raw")
        || column.ends_with("_boost")
        || column.ends_with("_collect")
        || column.ends_with("_throttle")
        || column.ends_with("_steer")
        || column.ends_with("_rotation_role")
        || column.ends_with("_title_id")
        || column.ends_with("_score")
        || column.ends_with("_decal_id")
        || column.ends_with("_wheels_id")
        || column.ends_with("_antenna_id")
        || column.ends_with("_topper_id")
        || column.ends_with("_engine_audio_id")
        || column.ends_with("_trail_id")
        || column.ends_with("_goal_explosion_id")
        || column.ends_with("_primary_paint_finish_id")
        || column.ends_with("_accent_paint_finish_id")
        || column.ends_with("_first_frame_in_game")
        || column.ends_with("_time_in_game")
        || column.ends_with("_air_activate_count")
        || column.ends_with("_dodges_refreshed_counter")
}

fn set_row_value<T: ToString>(
    row: &mut [Option<CellValue>],
    column_index: &HashMap<String, usize>,
    column: &str,
    value: T,
) {
    if let Some(idx) = column_index.get(column) {
        row[*idx] = Some(CellValue::Utf8(value.to_string()));
    }
}

fn set_row_i32(
    row: &mut [Option<CellValue>],
    column_index: &HashMap<String, usize>,
    column: &str,
    value: i32,
) {
    if let Some(idx) = column_index.get(column) {
        row[*idx] = Some(CellValue::Int32(value));
    }
}

fn parse_cell_value(kind: ColumnKind, value: &str) -> Option<CellValue> {
    if value.is_empty() {
        return None;
    }
    match kind {
        ColumnKind::Utf8 => Some(CellValue::Utf8(value.to_string())),
        ColumnKind::Int32 => value.parse::<i32>().ok().map(CellValue::Int32),
        ColumnKind::Float32 => value.parse::<f32>().ok().map(CellValue::Float32),
        ColumnKind::Boolean => match value.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" => Some(CellValue::Boolean(true)),
            "false" | "0" | "no" | "n" => Some(CellValue::Boolean(false)),
            _ => None,
        },
    }
}

fn overlay_event_values(row: &mut [Option<CellValue>], values: &RowValues) {
    for (idx, value) in values.as_slice().iter().enumerate() {
        if idx < row.len() && value.is_some() {
            row[idx] = value.clone();
        }
    }
}

#[derive(Clone, Debug, Default)]
struct EntityColumnIndexes {
    pos_x: Option<usize>,
    pos_y: Option<usize>,
    pos_z: Option<usize>,
    vel_x: Option<usize>,
    vel_y: Option<usize>,
    vel_z: Option<usize>,
    ang_vel_x: Option<usize>,
    ang_vel_y: Option<usize>,
    ang_vel_z: Option<usize>,
    rot_x: Option<usize>,
    rot_y: Option<usize>,
    rot_z: Option<usize>,
}

#[derive(Clone, Debug, Default)]
struct PlayerColumnIndexes {
    entity: EntityColumnIndexes,
    boost_raw: Option<usize>,
    boost: Option<usize>,
    boost_active: Option<usize>,
    boost_collect: Option<usize>,
    throttle: Option<usize>,
    steer: Option<usize>,
    handbrake: Option<usize>,
    ball_cam: Option<usize>,
    dodge_active: Option<usize>,
    jump_active: Option<usize>,
    double_jump_active: Option<usize>,
    jumped: Option<usize>,
    flipped: Option<usize>,
    jump_air_activate_count: Option<usize>,
    double_jump_air_activate_count: Option<usize>,
    dodge_air_activate_count: Option<usize>,
    dodges_refreshed_counter: Option<usize>,
    supersonic: Option<usize>,
    distance_to_ball: Option<usize>,
    angle_to_ball: Option<usize>,
    distance_to_own_net: Option<usize>,
    angle_to_own_net: Option<usize>,
    distance_to_opp_net: Option<usize>,
    angle_to_opp_net: Option<usize>,
    rotation_role: Option<usize>,
    distance_to_players: Vec<Option<usize>>,
}

#[derive(Clone, Debug)]
struct FrameColumnIndexes {
    frame_number: Option<usize>,
    observed_frame_number: Option<usize>,
    seconds_elapsed: Option<usize>,
    frame_has_event: Option<usize>,
    frame_event_count: Option<usize>,
    ball: EntityColumnIndexes,
    players: Vec<PlayerColumnIndexes>,
}

fn frame_row_indexes(
    column_index: &HashMap<String, usize>,
    players: &[PlayerInfo],
) -> FrameColumnIndexes {
    FrameColumnIndexes {
        frame_number: column_index.get("frame_number").copied(),
        observed_frame_number: column_index.get("observed_frame_number").copied(),
        seconds_elapsed: column_index.get("seconds_elapsed").copied(),
        frame_has_event: column_index.get("frame_has_event").copied(),
        frame_event_count: column_index.get("frame_event_count").copied(),
        ball: entity_column_indexes(column_index, "ball"),
        players: players
            .iter()
            .map(|player| player_column_indexes(column_index, players, player))
            .collect(),
    }
}

fn entity_column_indexes(
    column_index: &HashMap<String, usize>,
    prefix: &str,
) -> EntityColumnIndexes {
    EntityColumnIndexes {
        pos_x: column_index.get(&format!("{prefix}_pos_x")).copied(),
        pos_y: column_index.get(&format!("{prefix}_pos_y")).copied(),
        pos_z: column_index.get(&format!("{prefix}_pos_z")).copied(),
        vel_x: column_index.get(&format!("{prefix}_vel_x")).copied(),
        vel_y: column_index.get(&format!("{prefix}_vel_y")).copied(),
        vel_z: column_index.get(&format!("{prefix}_vel_z")).copied(),
        ang_vel_x: column_index.get(&format!("{prefix}_ang_vel_x")).copied(),
        ang_vel_y: column_index.get(&format!("{prefix}_ang_vel_y")).copied(),
        ang_vel_z: column_index.get(&format!("{prefix}_ang_vel_z")).copied(),
        rot_x: column_index.get(&format!("{prefix}_rot_x")).copied(),
        rot_y: column_index.get(&format!("{prefix}_rot_y")).copied(),
        rot_z: column_index.get(&format!("{prefix}_rot_z")).copied(),
    }
}

fn player_column_indexes(
    column_index: &HashMap<String, usize>,
    players: &[PlayerInfo],
    player: &PlayerInfo,
) -> PlayerColumnIndexes {
    let slot = &player.slot;
    PlayerColumnIndexes {
        entity: entity_column_indexes(column_index, slot),
        boost_raw: column_index.get(&format!("{slot}_boost_raw")).copied(),
        boost: column_index.get(&format!("{slot}_boost")).copied(),
        boost_active: column_index.get(&format!("{slot}_boost_active")).copied(),
        boost_collect: column_index.get(&format!("{slot}_boost_collect")).copied(),
        throttle: column_index.get(&format!("{slot}_throttle")).copied(),
        steer: column_index.get(&format!("{slot}_steer")).copied(),
        handbrake: column_index.get(&format!("{slot}_handbrake")).copied(),
        ball_cam: column_index.get(&format!("{slot}_ball_cam")).copied(),
        dodge_active: column_index.get(&format!("{slot}_dodge_active")).copied(),
        jump_active: column_index.get(&format!("{slot}_jump_active")).copied(),
        double_jump_active: column_index
            .get(&format!("{slot}_double_jump_active"))
            .copied(),
        jumped: column_index.get(&format!("{slot}_jumped")).copied(),
        flipped: column_index.get(&format!("{slot}_flipped")).copied(),
        jump_air_activate_count: column_index
            .get(&format!("{slot}_jump_air_activate_count"))
            .copied(),
        double_jump_air_activate_count: column_index
            .get(&format!("{slot}_double_jump_air_activate_count"))
            .copied(),
        dodge_air_activate_count: column_index
            .get(&format!("{slot}_dodge_air_activate_count"))
            .copied(),
        dodges_refreshed_counter: column_index
            .get(&format!("{slot}_dodges_refreshed_counter"))
            .copied(),
        supersonic: column_index.get(&format!("{slot}_supersonic")).copied(),
        distance_to_ball: column_index
            .get(&format!("{slot}_distance_to_ball"))
            .copied(),
        angle_to_ball: column_index.get(&format!("{slot}_angle_to_ball")).copied(),
        distance_to_own_net: column_index
            .get(&format!("{slot}_distance_to_own_net"))
            .copied(),
        angle_to_own_net: column_index
            .get(&format!("{slot}_angle_to_own_net"))
            .copied(),
        distance_to_opp_net: column_index
            .get(&format!("{slot}_distance_to_opp_net"))
            .copied(),
        angle_to_opp_net: column_index
            .get(&format!("{slot}_angle_to_opp_net"))
            .copied(),
        rotation_role: column_index.get(&format!("{slot}_rotation_role")).copied(),
        distance_to_players: players
            .iter()
            .map(|target| {
                if target.slot == *slot {
                    None
                } else {
                    column_index
                        .get(&format!("{slot}_distance_to_{}", target.slot))
                        .copied()
                }
            })
            .collect(),
    }
}

fn set_idx_i32(row: &mut [Option<CellValue>], idx: Option<usize>, value: i32) {
    if let Some(idx) = idx {
        row[idx] = Some(CellValue::Int32(value));
    }
}

fn set_idx_f32(row: &mut [Option<CellValue>], idx: Option<usize>, value: f32) {
    if let Some(idx) = idx {
        if value.is_finite() {
            row[idx] = Some(CellValue::Float32(value));
        }
    }
}

fn set_idx_opt_i32(row: &mut [Option<CellValue>], idx: Option<usize>, value: Option<i32>) {
    if let Some(value) = value {
        set_idx_i32(row, idx, value);
    }
}

fn set_idx_opt_f32(row: &mut [Option<CellValue>], idx: Option<usize>, value: Option<f32>) {
    if let Some(value) = value {
        set_idx_f32(row, idx, value);
    }
}

fn set_idx_bool(row: &mut [Option<CellValue>], idx: Option<usize>, value: bool) {
    if let Some(idx) = idx {
        row[idx] = Some(CellValue::Boolean(value));
    }
}

fn add_entity_state_row_indexed(
    row: &mut [Option<CellValue>],
    columns: &EntityColumnIndexes,
    state: EntityState,
) {
    if !state.has_pos {
        return;
    }
    set_idx_f32(row, columns.pos_x, state.pos.x);
    set_idx_f32(row, columns.pos_y, state.pos.y);
    set_idx_f32(row, columns.pos_z, state.pos.z);
    set_idx_f32(row, columns.vel_x, state.vel.x);
    set_idx_f32(row, columns.vel_y, state.vel.y);
    set_idx_f32(row, columns.vel_z, state.vel.z);
    set_idx_f32(row, columns.ang_vel_x, state.ang_vel.x);
    set_idx_f32(row, columns.ang_vel_y, state.ang_vel.y);
    set_idx_f32(row, columns.ang_vel_z, state.ang_vel.z);
    set_idx_f32(row, columns.rot_x, state.rot.x);
    set_idx_f32(row, columns.rot_y, state.rot.y);
    set_idx_f32(row, columns.rot_z, state.rot.z);
}

fn add_frame_state_values_row_indexed(
    row: &mut [Option<CellValue>],
    indexes: &FrameColumnIndexes,
    snapshot: &FrameSnapshot,
) {
    if let Some(ball) = snapshot.ball {
        add_entity_state_row_indexed(row, &indexes.ball, ball);
    }
    for (idx, player_indexes) in indexes.players.iter().enumerate() {
        if let Some(state) = snapshot.players.get(idx).and_then(Option::as_ref) {
            add_entity_state_row_indexed(row, &player_indexes.entity, state.entity);
            set_idx_opt_i32(row, player_indexes.boost_raw, state.boost.map(i32::from));
            set_idx_opt_i32(
                row,
                player_indexes.boost,
                state.boost.map(i32::from).map(boost_units),
            );
            set_idx_bool(row, player_indexes.boost_active, state.boost_active);
            set_idx_opt_i32(
                row,
                player_indexes.boost_collect,
                state.boost_collect.map(i32::from),
            );
            set_idx_opt_i32(row, player_indexes.throttle, state.throttle);
            set_idx_opt_i32(row, player_indexes.steer, state.steer);
            set_idx_bool(row, player_indexes.handbrake, state.handbrake);
            set_idx_bool(row, player_indexes.ball_cam, state.ball_cam);
            set_idx_bool(row, player_indexes.dodge_active, state.dodge_active);
            set_idx_bool(row, player_indexes.jump_active, state.jump_active);
            set_idx_bool(
                row,
                player_indexes.double_jump_active,
                state.double_jump_active,
            );
            set_idx_bool(row, player_indexes.jumped, state.jumped);
            set_idx_bool(row, player_indexes.flipped, state.flipped);
            set_idx_opt_i32(
                row,
                player_indexes.jump_air_activate_count,
                state.jump_air_activate_count,
            );
            set_idx_opt_i32(
                row,
                player_indexes.double_jump_air_activate_count,
                state.double_jump_air_activate_count,
            );
            set_idx_opt_i32(
                row,
                player_indexes.dodge_air_activate_count,
                state.dodge_air_activate_count,
            );
            set_idx_opt_i32(
                row,
                player_indexes.dodges_refreshed_counter,
                state.dodges_refreshed_counter,
            );
            set_idx_bool(row, player_indexes.supersonic, state.supersonic);
        }
    }
}

fn add_spatial_features_row_indexed(
    row: &mut [Option<CellValue>],
    indexes: &FrameColumnIndexes,
    snapshot: &FrameSnapshot,
    players: &[PlayerInfo],
) {
    let ball = snapshot
        .ball
        .and_then(|state| if state.has_pos { Some(state.pos) } else { None });
    let positions = players
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            snapshot
                .players
                .get(idx)
                .and_then(Option::as_ref)
                .map(|state| state.entity)
                .and_then(|state| if state.has_pos { Some(state.pos) } else { None })
        })
        .collect::<Vec<_>>();
    for (idx, player) in players.iter().enumerate() {
        let player_indexes = &indexes.players[idx];
        let pos = positions[idx];
        let own_net = defensive_net(player.team);
        let opp_net = offensive_net(player.team);
        set_idx_opt_f32(
            row,
            player_indexes.distance_to_ball,
            distance_opt(pos, ball),
        );
        set_idx_opt_f32(row, player_indexes.angle_to_ball, angle_opt(pos, ball));
        set_idx_opt_f32(
            row,
            player_indexes.distance_to_own_net,
            distance_opt(pos, own_net),
        );
        set_idx_opt_f32(
            row,
            player_indexes.angle_to_own_net,
            angle_opt(pos, own_net),
        );
        set_idx_opt_f32(
            row,
            player_indexes.distance_to_opp_net,
            distance_opt(pos, opp_net),
        );
        set_idx_opt_f32(
            row,
            player_indexes.angle_to_opp_net,
            angle_opt(pos, opp_net),
        );
        if let (Some(pos), Some(ball)) = (pos, ball) {
            let player_ball_distance = vec_distance(pos, ball);
            let mut closer_teammates = 0;
            for (other_idx, other) in players.iter().enumerate() {
                if other.slot == player.slot || other.team != player.team {
                    continue;
                }
                let Some(other_pos) = positions[other_idx] else {
                    continue;
                };
                if vec_distance(other_pos, ball) < player_ball_distance {
                    closer_teammates += 1;
                }
            }
            set_idx_i32(row, player_indexes.rotation_role, closer_teammates + 1);
        }
    }
    for (source_idx, source_indexes) in indexes.players.iter().enumerate() {
        for (target_idx, column) in source_indexes.distance_to_players.iter().enumerate() {
            if source_idx == target_idx {
                continue;
            }
            set_idx_opt_f32(
                row,
                *column,
                distance_opt(positions[source_idx], positions[target_idx]),
            );
        }
    }
}

fn write_arrow_batch<W: Write + Send>(
    writer: &mut ArrowWriter<W>,
    schema: Arc<Schema>,
    column_kinds: &[ColumnKind],
    rows: &[Vec<Option<CellValue>>],
    static_defaults: &[Option<CellValue>],
) -> Result<()> {
    let column_count = schema.fields().len();
    let arrays = (0..column_count)
        .into_par_iter()
        .map(|column_idx| match column_kinds[column_idx] {
            ColumnKind::Utf8 => {
                let values = rows
                    .iter()
                    .map(|row| cell_utf8(cell_at(row, static_defaults, column_idx)))
                    .collect::<Vec<_>>();
                Arc::new(StringArray::from(values)) as ArrayRef
            }
            ColumnKind::Int32 => {
                let values = rows
                    .iter()
                    .map(|row| cell_i32(cell_at(row, static_defaults, column_idx)))
                    .collect::<Vec<_>>();
                Arc::new(Int32Array::from(values)) as ArrayRef
            }
            ColumnKind::Float32 => {
                let values = rows
                    .iter()
                    .map(|row| cell_f32(cell_at(row, static_defaults, column_idx)))
                    .collect::<Vec<_>>();
                Arc::new(Float32Array::from(values)) as ArrayRef
            }
            ColumnKind::Boolean => {
                let values = rows
                    .iter()
                    .map(|row| cell_bool(cell_at(row, static_defaults, column_idx)))
                    .collect::<Vec<_>>();
                Arc::new(BooleanArray::from(values)) as ArrayRef
            }
        })
        .collect::<Vec<_>>();
    let batch = RecordBatch::try_new(schema, arrays)?;
    writer.write(&batch)?;
    Ok(())
}

fn cell_at<'a>(
    row: &'a [Option<CellValue>],
    static_defaults: &'a [Option<CellValue>],
    idx: usize,
) -> Option<&'a CellValue> {
    row.get(idx)
        .and_then(Option::as_ref)
        .or_else(|| static_defaults.get(idx).and_then(Option::as_ref))
}

fn cell_utf8(value: Option<&CellValue>) -> Option<&str> {
    match value {
        Some(CellValue::Utf8(value)) if !value.is_empty() => Some(value.as_str()),
        _ => None,
    }
}

fn cell_i32(value: Option<&CellValue>) -> Option<i32> {
    match value {
        Some(CellValue::Int32(value)) => Some(*value),
        Some(CellValue::Float32(value)) => Some(*value as i32),
        _ => None,
    }
}

fn cell_f32(value: Option<&CellValue>) -> Option<f32> {
    match value {
        Some(CellValue::Float32(value)) => Some(*value),
        Some(CellValue::Int32(value)) => Some(*value as f32),
        _ => None,
    }
}

fn cell_bool(value: Option<&CellValue>) -> Option<bool> {
    match value {
        Some(CellValue::Boolean(value)) => Some(*value),
        _ => None,
    }
}

fn header_player_id(player: &[(String, HeaderProp)], index: usize) -> String {
    let online_id = prop_string(player, "OnlineID").unwrap_or_default();
    if !online_id.is_empty() && online_id != "0" {
        return online_id;
    }

    let name = prop_string(player, "Name")
        .or_else(|| prop_string(player, "PlayerName"))
        .unwrap_or_else(|| format!("player_{index}"));
    let team = prop_i32(player, "Team")
        .or_else(|| prop_i32(player, "PlayerTeam"))
        .unwrap_or(0);
    format!("{team}:{index}:{name}")
}

fn unique_id_to_network_id(unique_id: &UniqueId) -> String {
    match &unique_id.remote_id {
        RemoteId::PlayStation(value) => format!("playstation:{}", value.online_id),
        RemoteId::PsyNet(value) => format!("psynet:{}", value.online_id),
        RemoteId::SplitScreen(value) => format!("splitscreen:{value}"),
        RemoteId::Steam(value) => format!("steam:{value}"),
        RemoteId::Switch(value) => format!("switch:{}", value.online_id),
        RemoteId::Xbox(value) => format!("xbox:{value}"),
        RemoteId::QQ(value) => format!("qq:{value}"),
        RemoteId::Epic(value) => format!("epic:{value}"),
    }
}

fn unique_id_to_player_id(unique_id: &UniqueId) -> String {
    match &unique_id.remote_id {
        RemoteId::PlayStation(value) => value.online_id.to_string(),
        RemoteId::PsyNet(value) => value.online_id.to_string(),
        RemoteId::SplitScreen(value) => value.to_string(),
        RemoteId::Steam(value) => value.to_string(),
        RemoteId::Switch(value) => value.online_id.to_string(),
        RemoteId::Xbox(value) => value.to_string(),
        RemoteId::QQ(value) => value.to_string(),
        RemoteId::Epic(value) => value.to_string(),
    }
}

fn pbp_players(replay: &Replay) -> Vec<PlayerInfo> {
    let mut players_by_team = Vec::new();
    if let Some(player_stats) = header_array(replay, "PlayerStats") {
        for (idx, player) in player_stats.iter().enumerate() {
            let Some(team @ 0..=1) =
                prop_i32(player, "Team").or_else(|| prop_i32(player, "PlayerTeam"))
            else {
                continue;
            };
            let online_id = prop_string(player, "OnlineID").unwrap_or_default();
            players_by_team.push(PlayerInfo {
                id: header_player_id(player, idx),
                actor_id: String::new(),
                network_id: online_id,
                name: prop_string(player, "Name")
                    .or_else(|| prop_string(player, "PlayerName"))
                    .unwrap_or_default(),
                team,
                slot: String::new(),
                platform: prop_string(player, "Platform").unwrap_or_default(),
                is_bot: prop_bool(player, "bBot")
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                score: prop_i32(player, "Score")
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                title_id: String::new(),
                first_frame_in_game: String::new(),
                time_in_game: String::new(),
                car_id: String::new(),
                car_name: String::new(),
                decal_id: String::new(),
                wheels_id: String::new(),
                boost_id: String::new(),
                antenna_id: String::new(),
                topper_id: String::new(),
                engine_audio_id: String::new(),
                trail_id: String::new(),
                goal_explosion_id: String::new(),
                primary_paint_finish_id: String::new(),
                accent_paint_finish_id: String::new(),
                camera_settings: None,
            });
        }
    }
    players_by_team.sort_by(|left, right| {
        left.team
            .cmp(&right.team)
            .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
    });
    let mut players = Vec::new();
    let mut blue_count = 0;
    let mut orange_count = 0;
    for mut player in players_by_team {
        player.slot = if player.team == 1 {
            orange_count += 1;
            format!("orange_player_{orange_count}")
        } else {
            blue_count += 1;
            format!("blue_player_{blue_count}")
        };
        players.push(player);
    }
    players
}

fn frame_context(replay: &Replay) -> PbpContext {
    replay_context(replay, ContextMode::FramesOnly)
}

fn pbp_context(replay: &Replay) -> PbpContext {
    replay_context(replay, ContextMode::Full)
}

fn replay_context(replay: &Replay, mode: ContextMode) -> PbpContext {
    let include_events = mode == ContextMode::Full;
    let mut context = PbpContext {
        playlist: header_string(replay, "Playlist").unwrap_or_default(),
        players: pbp_players(replay),
        ..PbpContext::default()
    };
    let mut pri_name: HashMap<i32, String> = HashMap::new();
    let mut pri_team_actor: HashMap<i32, i32> = HashMap::new();
    let mut team_actor_number: HashMap<i32, i32> = HashMap::new();
    let mut car_pri: HashMap<i32, i32> = HashMap::new();
    let mut camera_actor_pri: HashMap<i32, i32> = HashMap::new();
    let mut pending_camera_settings: HashMap<i32, PlayerCameraSettings> = HashMap::new();
    let mut camera_settings_by_pri: HashMap<i32, PlayerCameraSettings> = HashMap::new();
    let mut car_player_name: HashMap<i32, String> = HashMap::new();
    let mut component_vehicle: HashMap<i32, i32> = HashMap::new();
    let mut actor_object_name: HashMap<i32, String> = HashMap::new();
    let mut last_demo_frame: HashMap<(String, String), i32> = HashMap::new();
    let mut last_demo_victim_frame: HashMap<String, i32> = HashMap::new();
    let mut ball_actors: HashMap<i32, bool> = HashMap::new();
    let mut previous_ball_ang_vel: Option<Vec3> = None;
    let mut latest_ball_state: Option<EntityState> = None;
    let mut latest_car_states: HashMap<i32, EntityState> = HashMap::new();
    let mut latest_player_states: Vec<Option<PlayerFrameState>> = vec![None; context.players.len()];
    let mut latest_seconds_remaining: Option<i32> = None;
    let mut latest_hit_team: Option<i32> = None;
    let mut latest_match_stats: HashMap<(i32, &'static str), i32> = HashMap::new();
    let mut pending_official_stats: Vec<PendingOfficialStatEvent> = Vec::new();
    let mut active_player_names: HashSet<String> = HashSet::new();
    let mut hit_candidates: Vec<HitCandidate> = Vec::new();
    let goal_frames = header_goal_frames(replay);
    let player_team_by_name = context
        .players
        .iter()
        .map(|player| (player.name.clone(), player.team))
        .collect::<HashMap<_, _>>();
    let player_index_by_name = context
        .players
        .iter()
        .enumerate()
        .map(|(idx, player)| (player.name.clone(), idx))
        .collect::<HashMap<_, _>>();

    if let Some(network_frames) = &replay.network_frames {
        for (frame_number, frame) in network_frames.frames.iter().enumerate() {
            for new_actor in &frame.new_actors {
                let object = object_name(replay, new_actor.object_id.0);
                actor_object_name.insert(new_actor.actor_id.0, object.clone());
                if object.contains("Ball") {
                    ball_actors.insert(new_actor.actor_id.0, true);
                }
            }
            for updated_actor in &frame.updated_actors {
                let name = object_name(replay, updated_actor.object_id.0);
                match (&name[..], &updated_actor.attribute) {
                    ("Engine.PlayerReplicationInfo:PlayerName", Attribute::String(value)) => {
                        pri_name.insert(updated_actor.actor_id.0, value.clone());
                        if let Some(player) = context_player_mut(
                            &mut context.players,
                            updated_actor.actor_id.0,
                            &pri_name,
                        ) {
                            player.actor_id = updated_actor.actor_id.0.to_string();
                            player.camera_settings = camera_settings_by_pri
                                .get(&updated_actor.actor_id.0)
                                .copied();
                        }
                        let mut car_actors = car_pri.keys().copied().collect::<Vec<_>>();
                        car_actors.sort_unstable();
                        for car_actor in car_actors {
                            if car_pri.get(&car_actor).copied() == Some(updated_actor.actor_id.0) {
                                car_player_name.insert(car_actor, value.clone());
                            }
                        }
                        flush_pending_official_stats(
                            &mut context.official_stats,
                            &mut pending_official_stats,
                            &pri_name,
                            Some(updated_actor.actor_id.0),
                        );
                    }
                    ("Engine.PlayerReplicationInfo:Team", Attribute::ActiveActor(value)) => {
                        if value.active {
                            pri_team_actor.insert(updated_actor.actor_id.0, value.actor.0);
                        } else {
                            pri_team_actor.remove(&updated_actor.actor_id.0);
                        }
                        if let Some(player_name) = pri_name.get(&updated_actor.actor_id.0).cloned()
                        {
                            let frame = i32::try_from(frame_number).unwrap_or(i32::MAX);
                            if value.active {
                                if active_player_names.insert(player_name.clone()) {
                                    context.game_presence_events.push(GamePresenceEvent {
                                        frame_number: frame,
                                        player_name,
                                        event_type: "game-join",
                                    });
                                }
                            } else if active_player_names.remove(&player_name) {
                                context.game_presence_events.push(GamePresenceEvent {
                                    frame_number: frame,
                                    player_name: player_name.clone(),
                                    event_type: "game-leave",
                                });
                                if let Some(idx) = player_index_by_name.get(&player_name) {
                                    if let Some(slot) = latest_player_states.get_mut(*idx) {
                                        *slot = None;
                                    }
                                }
                            }
                        }
                    }
                    ("Engine.PlayerReplicationInfo:UniqueId", Attribute::UniqueId(value)) => {
                        if let Some(player) = context_player_mut(
                            &mut context.players,
                            updated_actor.actor_id.0,
                            &pri_name,
                        ) {
                            player.actor_id = updated_actor.actor_id.0.to_string();
                            player.network_id = unique_id_to_network_id(value);
                            player.id = unique_id_to_player_id(value);
                        }
                    }
                    ("TAGame.PRI_TA:CameraSettings", Attribute::CamSettings(value)) => {
                        if let Some(player) = context_player_mut(
                            &mut context.players,
                            updated_actor.actor_id.0,
                            &pri_name,
                        ) {
                            let settings = PlayerCameraSettings::from(**value);
                            player.camera_settings = Some(settings);
                            camera_settings_by_pri.insert(updated_actor.actor_id.0, settings);
                        }
                    }
                    ("TAGame.CameraSettingsActor_TA:PRI", Attribute::ActiveActor(value)) => {
                        if value.active {
                            camera_actor_pri.insert(updated_actor.actor_id.0, value.actor.0);
                            if let Some(settings) =
                                pending_camera_settings.remove(&updated_actor.actor_id.0)
                            {
                                camera_settings_by_pri.insert(value.actor.0, settings);
                                if let Some(player) = context_player_mut(
                                    &mut context.players,
                                    value.actor.0,
                                    &pri_name,
                                ) {
                                    player.camera_settings = Some(settings);
                                }
                            }
                        }
                    }
                    (
                        "TAGame.CameraSettingsActor_TA:ProfileSettings",
                        Attribute::CamSettings(value),
                    ) => {
                        let settings = PlayerCameraSettings::from(**value);
                        if let Some(pri_actor_id) =
                            camera_actor_pri.get(&updated_actor.actor_id.0).copied()
                        {
                            camera_settings_by_pri.insert(pri_actor_id, settings);
                            if let Some(player) =
                                context_player_mut(&mut context.players, pri_actor_id, &pri_name)
                            {
                                player.camera_settings = Some(settings);
                            }
                        } else {
                            pending_camera_settings.insert(updated_actor.actor_id.0, settings);
                        }
                    }
                    ("Engine.Pawn:PlayerReplicationInfo", Attribute::ActiveActor(value)) => {
                        car_pri.insert(updated_actor.actor_id.0, value.actor.0);
                        if let Some(player_name) = pri_name.get(&value.actor.0) {
                            car_player_name.insert(updated_actor.actor_id.0, player_name.clone());
                        }
                    }
                    ("TAGame.CarComponent_TA:Vehicle", Attribute::ActiveActor(value)) => {
                        if value.active {
                            component_vehicle.insert(updated_actor.actor_id.0, value.actor.0);
                        }
                    }
                    ("TAGame.Ball_TA:GameEvent", Attribute::ActiveActor(value)) => {
                        if value.active {
                            ball_actors.insert(updated_actor.actor_id.0, true);
                        }
                    }
                    ("TAGame.Ball_TA:HitTeamNum", Attribute::Byte(value)) => {
                        latest_hit_team = Some(i32::from(*value));
                    }
                    ("TAGame.Ball_TA:HitTeamNum", Attribute::Int(value)) => {
                        latest_hit_team = Some(*value);
                    }
                    ("TAGame.RBActor_TA:ReplicatedRBState", attribute) => {
                        if let Some(state) = entity_state_from_attribute(attribute) {
                            if ball_actors.contains_key(&updated_actor.actor_id.0) {
                                latest_ball_state = Some(state);
                                if include_events {
                                    if let Some(previous) = previous_ball_ang_vel {
                                        if vec_changed(previous, state.ang_vel) {
                                            if let Some(candidate) = closest_ball_hit_candidate(
                                                i32::try_from(frame_number).unwrap_or(i32::MAX),
                                                state,
                                                latest_hit_team,
                                                &context.players,
                                                &pri_name,
                                                &car_pri,
                                                &latest_car_states,
                                                &goal_frames,
                                            ) {
                                                hit_candidates.push(candidate);
                                            }
                                        }
                                    }
                                    previous_ball_ang_vel = Some(state.ang_vel);
                                }
                            } else {
                                latest_car_states.insert(updated_actor.actor_id.0, state);
                                if let Some(name) = player_name_for_car(
                                    updated_actor.actor_id.0,
                                    &car_pri,
                                    &pri_name,
                                ) {
                                    let player_state = match player_state_mut(
                                        &mut latest_player_states,
                                        &player_index_by_name,
                                        &name,
                                    ) {
                                        Some(value) => value,
                                        None => continue,
                                    };
                                    player_state.entity = state;
                                    player_state.supersonic =
                                        vec_norm(state.vel) >= SUPERSONIC_THRESHOLD;
                                }
                            }
                        }
                    }
                    (
                        "TAGame.CarComponent_Boost_TA:ReplicatedBoost",
                        Attribute::ReplicatedBoost(value),
                    ) => {
                        if let Some(name) = component_vehicle
                            .get(&updated_actor.actor_id.0)
                            .and_then(|car_actor| {
                                player_name_for_car(*car_actor, &car_pri, &pri_name)
                            })
                        {
                            if let Some(player_state) = player_state_mut(
                                &mut latest_player_states,
                                &player_index_by_name,
                                &name,
                            ) {
                                player_state.boost = Some(value.boost_amount);
                                player_state.boost_updated_frame =
                                    Some(i32::try_from(frame_number).unwrap_or(i32::MAX));
                                player_state.boost_collect = Some(value.grant_count);
                            }
                        }
                    }
                    (
                        "TAGame.CarComponent_Boost_TA:ReplicatedBoostAmount",
                        Attribute::Byte(value),
                    ) => {
                        if let Some(name) = component_vehicle
                            .get(&updated_actor.actor_id.0)
                            .and_then(|car_actor| {
                                player_name_for_car(*car_actor, &car_pri, &pri_name)
                            })
                        {
                            if let Some(player_state) = player_state_mut(
                                &mut latest_player_states,
                                &player_index_by_name,
                                &name,
                            ) {
                                player_state.boost = Some(*value);
                                player_state.boost_updated_frame =
                                    Some(i32::try_from(frame_number).unwrap_or(i32::MAX));
                            }
                        }
                    }
                    ("TAGame.Vehicle_TA:ReplicatedThrottle", Attribute::Byte(value)) => {
                        if let Some(name) =
                            player_name_for_car(updated_actor.actor_id.0, &car_pri, &pri_name)
                        {
                            if let Some(player_state) = player_state_mut(
                                &mut latest_player_states,
                                &player_index_by_name,
                                &name,
                            ) {
                                player_state.throttle = Some(i32::from(*value));
                            }
                        }
                    }
                    ("TAGame.Vehicle_TA:ReplicatedSteer", Attribute::Byte(value)) => {
                        if let Some(name) =
                            player_name_for_car(updated_actor.actor_id.0, &car_pri, &pri_name)
                        {
                            if let Some(player_state) = player_state_mut(
                                &mut latest_player_states,
                                &player_index_by_name,
                                &name,
                            ) {
                                player_state.steer = Some(i32::from(*value));
                            }
                        }
                    }
                    ("TAGame.Vehicle_TA:bReplicatedHandbrake", Attribute::Boolean(value)) => {
                        if let Some(name) =
                            player_name_for_car(updated_actor.actor_id.0, &car_pri, &pri_name)
                        {
                            if let Some(player_state) = player_state_mut(
                                &mut latest_player_states,
                                &player_index_by_name,
                                &name,
                            ) {
                                player_state.handbrake = *value;
                            }
                        }
                    }
                    ("TAGame.CarComponent_TA:ReplicatedActive", Attribute::Byte(value)) => {
                        if let Some(car_actor) =
                            component_vehicle.get(&updated_actor.actor_id.0).copied()
                        {
                            if let Some(name) = player_name_for_car(car_actor, &car_pri, &pri_name)
                            {
                                let component_name = actor_object_name
                                    .get(&updated_actor.actor_id.0)
                                    .map(String::as_str)
                                    .unwrap_or_default();
                                let active = *value != 0;
                                let player_state = match player_state_mut(
                                    &mut latest_player_states,
                                    &player_index_by_name,
                                    &name,
                                ) {
                                    Some(value) => value,
                                    None => continue,
                                };
                                player_state.boost_active =
                                    component_name.contains("Boost") && active;
                                if component_name.contains("Jump")
                                    && !component_name.contains("Double")
                                {
                                    player_state.jump_active = active;
                                    player_state.jumped = active;
                                }
                                if component_name.contains("DoubleJump") {
                                    player_state.double_jump_active = active;
                                }
                                if component_name.contains("Dodge") {
                                    player_state.dodge_active = active;
                                    player_state.flipped = active;
                                }
                            }
                        }
                    }
                    ("TAGame.CarComponent_Dodge_TA:DodgeTorque", _) => {
                        if let Some(car_actor) =
                            component_vehicle.get(&updated_actor.actor_id.0).copied()
                        {
                            if let Some(name) = player_name_for_car(car_actor, &car_pri, &pri_name)
                            {
                                if let Some(player_state) = player_state_mut(
                                    &mut latest_player_states,
                                    &player_index_by_name,
                                    &name,
                                ) {
                                    player_state.dodge_active = true;
                                    player_state.flipped = true;
                                }
                            }
                        }
                    }
                    (
                        "TAGame.CarComponent_AirActivate_TA:AirActivateCount",
                        Attribute::Int(value),
                    ) => {
                        if let Some(car_actor) =
                            component_vehicle.get(&updated_actor.actor_id.0).copied()
                        {
                            if let Some(name) = player_name_for_car(car_actor, &car_pri, &pri_name)
                            {
                                let component_name = actor_object_name
                                    .get(&updated_actor.actor_id.0)
                                    .map(String::as_str)
                                    .unwrap_or_default();
                                if let Some(player_state) = player_state_mut(
                                    &mut latest_player_states,
                                    &player_index_by_name,
                                    &name,
                                ) {
                                    if component_name.contains("Dodge") {
                                        player_state.dodge_air_activate_count = Some(*value);
                                        player_state.flip_available = *value == 0;
                                    } else if component_name.contains("DoubleJump") {
                                        player_state.double_jump_air_activate_count = Some(*value);
                                    } else if component_name.contains("Jump") {
                                        player_state.jump_air_activate_count = Some(*value);
                                    }
                                }
                            }
                        }
                    }
                    ("TAGame.Car_TA:DodgesRefreshedCounter", Attribute::Int(value)) => {
                        if let Some(name) =
                            player_name_for_car(updated_actor.actor_id.0, &car_pri, &pri_name)
                        {
                            if let Some(player_state) = player_state_mut(
                                &mut latest_player_states,
                                &player_index_by_name,
                                &name,
                            ) {
                                player_state.dodges_refreshed_counter = Some(*value);
                            }
                        }
                    }
                    ("Engine.GameReplicationInfo:ServerName", Attribute::String(value)) => {
                        context.server_name = value.clone();
                    }
                    ("ProjectX.GRI_X:GameServerID", Attribute::Int(value)) => {
                        context.game_server_id = value.to_string();
                    }
                    ("ProjectX.GRI_X:GameServerID", Attribute::String(value)) => {
                        context.game_server_id = value.clone();
                    }
                    ("ProjectX.GRI_X:ReplicatedGamePlaylist", Attribute::Int(value)) => {
                        context.playlist = playlist_name(*value).to_string();
                    }
                    ("TAGame.GameEvent_Soccar_TA:SecondsRemaining", Attribute::Int(value)) => {
                        latest_seconds_remaining = Some(*value);
                    }
                    ("TAGame.GameEvent_Soccar_TA:bBallHasBeenHit", Attribute::Boolean(value)) => {
                        if include_events && *value {
                            if let Some(state) = latest_ball_state {
                                if let Some(candidate) = closest_ball_hit_candidate(
                                    i32::try_from(frame_number).unwrap_or(i32::MAX),
                                    state,
                                    latest_hit_team,
                                    &context.players,
                                    &pri_name,
                                    &car_pri,
                                    &latest_car_states,
                                    &goal_frames,
                                ) {
                                    hit_candidates.push(candidate);
                                }
                            }
                        }
                    }
                    ("TAGame.PRI_TA:MatchShots", Attribute::Int(value)) => {
                        record_official_stat(
                            &mut context.official_stats,
                            &mut pending_official_stats,
                            &mut latest_match_stats,
                            &pri_name,
                            updated_actor.actor_id.0,
                            "shot",
                            *value,
                            i32::try_from(frame_number).unwrap_or(i32::MAX),
                        );
                    }
                    ("TAGame.PRI_TA:MatchGoals", Attribute::Int(value)) => {
                        record_official_stat(
                            &mut context.official_stats,
                            &mut pending_official_stats,
                            &mut latest_match_stats,
                            &pri_name,
                            updated_actor.actor_id.0,
                            "goal",
                            *value,
                            i32::try_from(frame_number).unwrap_or(i32::MAX),
                        );
                    }
                    ("TAGame.PRI_TA:MatchAssists", Attribute::Int(value)) => {
                        record_official_stat(
                            &mut context.official_stats,
                            &mut pending_official_stats,
                            &mut latest_match_stats,
                            &pri_name,
                            updated_actor.actor_id.0,
                            "assist",
                            *value,
                            i32::try_from(frame_number).unwrap_or(i32::MAX),
                        );
                    }
                    ("TAGame.PRI_TA:MatchSaves", Attribute::Int(value)) => {
                        record_official_stat(
                            &mut context.official_stats,
                            &mut pending_official_stats,
                            &mut latest_match_stats,
                            &pri_name,
                            updated_actor.actor_id.0,
                            "save",
                            *value,
                            i32::try_from(frame_number).unwrap_or(i32::MAX),
                        );
                    }
                    ("TAGame.Team_TA:CustomTeamName", Attribute::String(value)) => {
                        let team_number = team_actor_number
                            .get(&updated_actor.actor_id.0)
                            .copied()
                            .or_else(|| {
                                infer_team_number(
                                    updated_actor.actor_id.0,
                                    &pri_name,
                                    &pri_team_actor,
                                    &player_team_by_name,
                                )
                            });
                        if team_number == Some(0) {
                            context.blue_team_name = value.clone();
                        } else if team_number == Some(1) {
                            context.orange_team_name = value.clone();
                        }
                    }
                    ("TAGame.PRI_TA:Title", Attribute::Int(value)) => {
                        if let Some(player) = context_player_mut(
                            &mut context.players,
                            updated_actor.actor_id.0,
                            &pri_name,
                        ) {
                            player.title_id = value.to_string();
                        }
                    }
                    ("TAGame.PRI_TA:TotalGameTimePlayed", Attribute::Float(value)) => {
                        if let Some(player) = context_player_mut(
                            &mut context.players,
                            updated_actor.actor_id.0,
                            &pri_name,
                        ) {
                            player.time_in_game = value.to_string();
                            if player.first_frame_in_game.is_empty() {
                                player.first_frame_in_game = (frame_number + 1).to_string();
                            }
                        }
                    }
                    ("TAGame.PRI_TA:ClientLoadouts", attribute) => {
                        if let Some(player) = context_player_mut(
                            &mut context.players,
                            updated_actor.actor_id.0,
                            &pri_name,
                        ) {
                            if let Ok(value) = serde_json::to_value(attribute) {
                                assign_player_loadout(player, &value);
                            }
                        }
                    }
                    (
                        "TAGame.Car_TA:ReplicatedDemolishExtended",
                        Attribute::DemolishExtended(value),
                    ) => {
                        if !include_events {
                            continue;
                        }
                        let attacker_name = pri_name
                            .get(&value.attacker_pri.actor.0)
                            .or_else(|| {
                                car_pri
                                    .get(&value.attacker.actor.0)
                                    .and_then(|pri_actor| pri_name.get(pri_actor))
                            })
                            .cloned();
                        let victim_name = car_pri
                            .get(&value.victim.actor.0)
                            .and_then(|pri_actor| pri_name.get(pri_actor))
                            .or_else(|| car_player_name.get(&value.victim.actor.0))
                            .cloned();
                        if let (Some(attacker_name), Some(victim_name)) =
                            (attacker_name, victim_name)
                        {
                            record_demo_event(
                                &mut context,
                                &mut last_demo_frame,
                                &mut last_demo_victim_frame,
                                attacker_name,
                                victim_name,
                                i32::try_from(frame_number).unwrap_or(i32::MAX),
                            );
                        }
                    }
                    _ => {}
                }

                if include_events && attribute_type(&updated_actor.attribute) == "demolish_extended"
                {
                    if let Ok(value) = serde_json::to_value(&updated_actor.attribute) {
                        let attacker_pri = value
                            .get("DemolishExtended")
                            .and_then(|value| value.get("attacker_pri"))
                            .and_then(|value| value.get("actor"))
                            .and_then(Value::as_i64)
                            .and_then(|value| i32::try_from(value).ok());
                        let attacker_car = value
                            .get("DemolishExtended")
                            .and_then(|value| value.get("attacker"))
                            .and_then(|value| value.get("actor"))
                            .and_then(Value::as_i64)
                            .and_then(|value| i32::try_from(value).ok());
                        let victim_car = value
                            .get("DemolishExtended")
                            .and_then(|value| value.get("victim"))
                            .and_then(|value| value.get("actor"))
                            .and_then(Value::as_i64)
                            .and_then(|value| i32::try_from(value).ok());
                        let attacker_name = attacker_pri
                            .and_then(|pri_actor| pri_name.get(&pri_actor))
                            .or_else(|| {
                                attacker_car
                                    .and_then(|car_actor| car_pri.get(&car_actor))
                                    .and_then(|pri_actor| pri_name.get(pri_actor))
                            })
                            .cloned();
                        let victim_name = victim_car
                            .and_then(|car_actor| car_pri.get(&car_actor))
                            .and_then(|pri_actor| pri_name.get(pri_actor))
                            .cloned()
                            .or_else(|| {
                                victim_car
                                    .and_then(|car_actor| car_player_name.get(&car_actor))
                                    .cloned()
                            });
                        if let (Some(attacker_name), Some(victim_name)) =
                            (attacker_name, victim_name)
                        {
                            record_demo_event(
                                &mut context,
                                &mut last_demo_frame,
                                &mut last_demo_victim_frame,
                                attacker_name,
                                victim_name,
                                i32::try_from(frame_number).unwrap_or(i32::MAX),
                            );
                        }
                    }
                }

                if name == "Engine.PlayerReplicationInfo:Team" {
                    if let Some(player_name) = pri_name.get(&updated_actor.actor_id.0) {
                        if let Some(team_number) = player_team_by_name.get(player_name) {
                            if let Attribute::ActiveActor(value) = &updated_actor.attribute {
                                if value.active {
                                    team_actor_number.insert(value.actor.0, *team_number);
                                }
                            }
                        }
                    }
                }
            }
            context.frame_states.push(FrameSnapshot {
                frame_number: i32::try_from(frame_number).unwrap_or(i32::MAX),
                seconds_remaining: latest_seconds_remaining,
                seconds_elapsed: None,
                ball: latest_ball_state,
                players: latest_player_states.clone(),
            });
        }
    }
    add_frame_seconds_elapsed(&mut context.frame_states);
    let kickoff_starts = kickoff_start_frames_from_resets(&context.frame_states);
    apply_kickoff_boost_resets(&mut context.frame_states, &kickoff_starts);
    add_inferred_initial_game_joins(&mut context);
    if include_events {
        add_demo_respawn_events(&mut context);
    }

    for player in &mut context.players {
        if player.first_frame_in_game.is_empty() {
            player.first_frame_in_game = "1".to_string();
        }
    }
    if include_events {
        flush_pending_official_stats(
            &mut context.official_stats,
            &mut pending_official_stats,
            &pri_name,
            None,
        );
        dedupe_official_stats(&mut context.official_stats);
        context.ball_events = classify_ball_events(
            filter_duplicate_hits(hit_candidates),
            &context.players,
            &goal_frames,
            &kickoff_starts,
        );
    }
    context
}

fn record_demo_event(
    context: &mut PbpContext,
    last_demo_frame: &mut HashMap<(String, String), i32>,
    last_demo_victim_frame: &mut HashMap<String, i32>,
    attacker_name: String,
    victim_name: String,
    frame_number: i32,
) {
    if attacker_name.is_empty() || victim_name.is_empty() || attacker_name == victim_name {
        return;
    }
    let key = (attacker_name.clone(), victim_name.clone());
    let duplicate_pair = last_demo_frame
        .get(&key)
        .map(|prior_frame| frame_number - *prior_frame <= DEMO_EVENT_COOLDOWN_FRAMES)
        .unwrap_or(false);
    let duplicate_victim = last_demo_victim_frame
        .get(&victim_name)
        .map(|prior_frame| frame_number - *prior_frame <= DEMO_EVENT_COOLDOWN_FRAMES)
        .unwrap_or(false);
    if duplicate_pair || duplicate_victim {
        return;
    }
    context.demo_events.push(CarContactEvent {
        frame_number,
        event_type: "demo".to_string(),
        player_1_name: attacker_name.clone(),
        player_2_name: victim_name.clone(),
        car_contact_distance: 0.0,
        relative_speed: 0.0,
        event_player_1_speed: 0.0,
        event_player_2_speed: 0.0,
        event_player_1_demolished: false,
        event_player_2_demolished: true,
    });
    last_demo_frame.insert(key, frame_number);
    last_demo_victim_frame.insert(victim_name, frame_number);
}

fn add_game_presence_events(
    rows: &mut Vec<PbpEventRecord>,
    context: &PbpContext,
    players: &[PlayerInfo],
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    team_size: Option<i32>,
    game_time: &str,
) {
    let mut seen = HashSet::new();
    for event in &context.game_presence_events {
        if event.player_name.is_empty()
            || !seen.insert((
                event.frame_number,
                event.player_name.clone(),
                event.event_type,
            ))
        {
            continue;
        }

        let mut values = pbp_base_values(
            game_id,
            match_guid,
            replay_name,
            map_id,
            context,
            team_size,
            game_time,
        );
        values.insert("event_type".to_string(), event.event_type.to_string());
        values.insert("frame_number".to_string(), event.frame_number.to_string());
        values.insert(
            "observed_frame_number".to_string(),
            event.frame_number.to_string(),
        );
        insert_seconds_elapsed(&mut values, context, event.frame_number);
        add_event_player(&mut values, players, 1, &event.player_name);
        if let Some(player) = players
            .iter()
            .find(|player| player.name == event.player_name)
        {
            values.insert("event_team".to_string(), team_name(player.team).to_string());
        }
        add_pbp_players(&mut values, player_static_values);
        add_frame_state_values(&mut values, context, event.frame_number, players);
        rows.push(PbpEventRecord {
            frame_number: Some(event.frame_number),
            event_type: event.event_type.to_string(),
            values,
        });
    }
}

fn dedupe_official_stats(official_stats: &mut Vec<OfficialStatEvent>) {
    let mut seen = std::collections::HashSet::new();
    official_stats
        .sort_by_key(|stat| (stat.frame_number, stat.player_name.clone(), stat.stat_type));
    official_stats
        .retain(|stat| seen.insert((stat.player_name.clone(), stat.stat_type, stat.stat_number)));
}

fn add_inferred_initial_game_joins(context: &mut PbpContext) {
    let mut joined = context
        .game_presence_events
        .iter()
        .filter(|event| event.event_type == "game-join")
        .map(|event| event.player_name.clone())
        .collect::<HashSet<_>>();

    for (idx, player) in context.players.iter().enumerate() {
        if joined.contains(&player.name) {
            continue;
        }
        if let Some(snapshot) = context
            .frame_states
            .iter()
            .find(|snapshot| snapshot.players.get(idx).and_then(Option::as_ref).is_some())
        {
            context.game_presence_events.push(GamePresenceEvent {
                frame_number: snapshot.frame_number,
                player_name: player.name.clone(),
                event_type: "game-join",
            });
            joined.insert(player.name.clone());
        }
    }
}

fn add_demo_respawn_events(context: &mut PbpContext) {
    let mut seen = context
        .game_presence_events
        .iter()
        .map(|event| {
            (
                event.frame_number,
                event.player_name.clone(),
                event.event_type,
            )
        })
        .collect::<HashSet<_>>();

    for demo in &context.demo_events {
        if demo.player_2_name.is_empty() {
            continue;
        }
        let target_frame = demo.frame_number + DEMO_RESPAWN_FRAMES;
        let frame_number = context
            .frame_states
            .iter()
            .find(|snapshot| snapshot.frame_number >= target_frame)
            .map(|snapshot| snapshot.frame_number)
            .unwrap_or(target_frame);
        let key = (frame_number, demo.player_2_name.clone(), "respawn");
        if seen.insert(key) {
            context.game_presence_events.push(GamePresenceEvent {
                frame_number,
                player_name: demo.player_2_name.clone(),
                event_type: "respawn",
            });
        }
    }
}

fn context_player_mut<'a>(
    players: &'a mut [PlayerInfo],
    pri_actor_id: i32,
    pri_name: &HashMap<i32, String>,
) -> Option<&'a mut PlayerInfo> {
    let name = pri_name.get(&pri_actor_id)?;
    players.iter_mut().find(|player| &player.name == name)
}

fn record_official_stat(
    official_stats: &mut Vec<OfficialStatEvent>,
    pending_official_stats: &mut Vec<PendingOfficialStatEvent>,
    latest_match_stats: &mut HashMap<(i32, &'static str), i32>,
    pri_name: &HashMap<i32, String>,
    pri_actor_id: i32,
    stat_type: &'static str,
    value: i32,
    frame_number: i32,
) {
    let key = (pri_actor_id, stat_type);
    let prior = latest_match_stats.get(&key).copied().unwrap_or(0);
    if value <= prior {
        return;
    }
    latest_match_stats.insert(key, value);
    let player_name = pri_name
        .get(&pri_actor_id)
        .filter(|value| !value.is_empty())
        .cloned();
    for stat_number in (prior + 1)..=value {
        if let Some(player_name) = &player_name {
            official_stats.push(OfficialStatEvent {
                pri_actor_id: Some(pri_actor_id),
                frame_number,
                player_name: player_name.clone(),
                stat_type,
                stat_number,
            });
        } else {
            pending_official_stats.push(PendingOfficialStatEvent {
                pri_actor_id,
                frame_number,
                stat_type,
                stat_number,
            });
        }
    }
}

fn flush_pending_official_stats(
    official_stats: &mut Vec<OfficialStatEvent>,
    pending_official_stats: &mut Vec<PendingOfficialStatEvent>,
    pri_name: &HashMap<i32, String>,
    pri_actor_id: Option<i32>,
) {
    let mut remaining = Vec::new();
    for pending in pending_official_stats.drain(..) {
        if pri_actor_id
            .map(|actor_id| pending.pri_actor_id != actor_id)
            .unwrap_or(false)
        {
            remaining.push(pending);
            continue;
        }
        match pri_name
            .get(&pending.pri_actor_id)
            .filter(|value| !value.is_empty())
        {
            Some(player_name) => official_stats.push(OfficialStatEvent {
                pri_actor_id: Some(pending.pri_actor_id),
                frame_number: pending.frame_number,
                player_name: player_name.clone(),
                stat_type: pending.stat_type,
                stat_number: pending.stat_number,
            }),
            None => remaining.push(pending),
        }
    }
    *pending_official_stats = remaining;
}

fn player_name_for_car(
    car_actor_id: i32,
    car_pri: &HashMap<i32, i32>,
    pri_name: &HashMap<i32, String>,
) -> Option<String> {
    car_pri
        .get(&car_actor_id)
        .and_then(|pri_actor| pri_name.get(pri_actor))
        .cloned()
}

fn player_state_mut<'a>(
    states: &'a mut [Option<PlayerFrameState>],
    player_index_by_name: &HashMap<String, usize>,
    player_name: &str,
) -> Option<&'a mut PlayerFrameState> {
    let idx = *player_index_by_name.get(player_name)?;
    let slot = states.get_mut(idx)?;
    if slot.is_none() {
        *slot = Some(PlayerFrameState::default());
    }
    slot.as_mut()
}

fn entity_state_from_attribute(attribute: &Attribute) -> Option<EntityState> {
    let value = serde_json::to_value(attribute).ok()?;
    let body = value.get("RigidBody")?;
    let location = body.get("location")?;
    let rotation = body.get("rotation")?;
    let linear_velocity = body.get("linear_velocity");
    let angular_velocity = body.get("angular_velocity");
    Some(EntityState {
        pos: Vec3 {
            x: json_f32(location, "x"),
            y: json_f32(location, "y"),
            z: json_f32(location, "z"),
        },
        vel: linear_velocity
            .map(|value| Vec3 {
                x: json_f32(value, "x"),
                y: json_f32(value, "y"),
                z: json_f32(value, "z"),
            })
            .unwrap_or_default(),
        ang_vel: angular_velocity
            .map(|value| Vec3 {
                x: json_f32(value, "x"),
                y: json_f32(value, "y"),
                z: json_f32(value, "z"),
            })
            .unwrap_or_default(),
        rot: Quat {
            x: json_f32(rotation, "x"),
            y: json_f32(rotation, "y"),
            z: json_f32(rotation, "z"),
            w: json_f32(rotation, "w"),
        },
        has_pos: true,
    })
}

fn json_f32(value: &Value, key: &str) -> f32 {
    value.get(key).and_then(Value::as_f64).unwrap_or(0.0) as f32
}

fn vec_changed(left: Vec3, right: Vec3) -> bool {
    (left.x - right.x).abs() > f32::EPSILON
        || (left.y - right.y).abs() > f32::EPSILON
        || (left.z - right.z).abs() > f32::EPSILON
}

fn vec_norm(value: Vec3) -> f32 {
    (value.x * value.x + value.y * value.y + value.z * value.z).sqrt()
}

fn closest_ball_hit_candidate(
    frame_number: i32,
    ball_state: EntityState,
    hit_team: Option<i32>,
    players: &[PlayerInfo],
    pri_name: &HashMap<i32, String>,
    car_pri: &HashMap<i32, i32>,
    car_states: &HashMap<i32, EntityState>,
    goal_frames: &[(i32, String)],
) -> Option<HitCandidate> {
    let mut best_name = String::new();
    let mut best_distance = f32::MAX;
    let mut player_positions = vec![None; players.len()];
    let mut car_actors = car_pri.keys().copied().collect::<Vec<_>>();
    car_actors.sort_unstable();
    for car_actor in car_actors {
        let Some(pri_actor) = car_pri.get(&car_actor) else {
            continue;
        };
        let name = match pri_name.get(pri_actor) {
            Some(value) => value,
            None => continue,
        };
        let (player_idx, player) = match players
            .iter()
            .enumerate()
            .find(|(_, player)| &player.name == name)
        {
            Some(value) => value,
            None => continue,
        };
        if hit_team.map(|team| team != player.team).unwrap_or(false) {
            continue;
        }
        let car_state = match car_states.get(&car_actor) {
            Some(value) if value.has_pos => *value,
            _ => continue,
        };
        player_positions[player_idx] = Some(car_state.pos);
        let distance = ball_collision_distance(
            ball_state.pos,
            car_state,
            player.car_id.parse().unwrap_or(23),
        );
        if distance < best_distance {
            best_distance = distance;
            best_name = name.clone();
        }
    }
    if best_distance < 300.0 {
        Some(HitCandidate {
            frame_number,
            player_name: best_name,
            collision_distance: best_distance,
            ball_state,
            player_positions,
            goal_number: goal_number_for_frame(frame_number, goal_frames),
        })
    } else {
        None
    }
}

fn filter_duplicate_hits(mut hits: Vec<HitCandidate>) -> Vec<HitCandidate> {
    hits.sort_by(|left, right| {
        left.frame_number
            .cmp(&right.frame_number)
            .then_with(|| left.player_name.cmp(&right.player_name))
            .then_with(|| left.collision_distance.total_cmp(&right.collision_distance))
    });
    let mut output = Vec::new();
    let mut idx = 0;
    while idx < hits.len() {
        let start = idx;
        let mut end = idx + 1;
        while end < hits.len()
            && hits[end].player_name == hits[start].player_name
            && hits[end].frame_number - hits[start].frame_number <= 10
        {
            end += 1;
        }
        let best = hits[start..end]
            .iter()
            .min_by(|left, right| left.collision_distance.total_cmp(&right.collision_distance))
            .cloned();
        if let Some(hit) = best {
            output.push(hit);
        }
        idx = end;
    }
    output
}

fn classify_ball_events(
    hits: Vec<HitCandidate>,
    players: &[PlayerInfo],
    goal_frames: &[(i32, String)],
    kickoff_starts: &[i32],
) -> Vec<BallEvent> {
    let mut events = hits
        .into_iter()
        .map(|hit| BallEvent {
            frame_number: hit.frame_number,
            event_type: "touch".to_string(),
            player_name: hit.player_name,
            player_2_name: String::new(),
            player_3_name: String::new(),
            collision_distance: hit.collision_distance,
            distance: 0.0,
            distance_to_goal: 0.0,
            previous_hit_frame_number: None,
            next_hit_frame_number: None,
            goal_number: hit.goal_number,
            ball_state: hit.ball_state,
            player_positions: hit.player_positions,
            goal: false,
            shot: false,
            missed_shot: false,
            missed_pass: false,
            pass_: false,
            clear: false,
            save: false,
            assist: false,
        })
        .collect::<Vec<_>>();

    for (goal_frame, scorer) in goal_frames {
        if let Some(event) = events.iter_mut().rev().find(|event| {
            event.frame_number <= *goal_frame
                && *goal_frame - event.frame_number <= 120
                && &event.player_name == scorer
        }) {
            event.goal = true;
        }
    }

    let mut last_passing_idx: Option<usize> = None;
    let mut last_pass_pair_frame: HashMap<(String, String), i32> = HashMap::new();
    for idx in 0..events.len() {
        let next_idx =
            if idx + 1 < events.len() && events[idx + 1].goal_number == events[idx].goal_number {
                Some(idx + 1)
            } else {
                None
            };
        if idx > 0 && events[idx - 1].goal_number == events[idx].goal_number {
            events[idx].previous_hit_frame_number = Some(events[idx - 1].frame_number);
        }
        if let Some(next_idx) = next_idx {
            events[idx].next_hit_frame_number = Some(events[next_idx].frame_number);
            events[idx].distance =
                vec_distance(events[idx].ball_state.pos, events[next_idx].ball_state.pos);
            if player_team(&events[idx].player_name, players)
                == player_team(&events[next_idx].player_name, players)
                && events[idx].player_name != events[next_idx].player_name
            {
                let key = (
                    events[idx].player_name.clone(),
                    events[next_idx].player_name.clone(),
                );
                let duplicate = last_pass_pair_frame
                    .get(&key)
                    .map(|frame| events[idx].frame_number - *frame <= 20)
                    .unwrap_or(false);
                if !duplicate {
                    events[idx].pass_ = true;
                    events[idx].player_2_name = events[next_idx].player_name.clone();
                    last_passing_idx = Some(idx);
                    last_pass_pair_frame.insert(key, events[idx].frame_number);
                }
            } else if events[idx].player_name != events[next_idx].player_name {
                last_passing_idx = None;
            }
            events[idx].clear = is_clear(&events[idx], Some(&events[next_idx]), players);
        } else {
            events[idx].clear = is_clear(&events[idx], None, players);
        }
        events[idx].distance_to_goal = distance_to_goal(&events[idx], players);
        events[idx].shot = is_shot(&events[idx], players) || events[idx].goal;
        events[idx].missed_shot = !events[idx].shot && is_missed_shot(&events[idx], players);
        if !events[idx].pass_ && !events[idx].shot && !events[idx].missed_shot {
            if let Some(target_name) = missed_pass_target(&events[idx], players) {
                events[idx].missed_pass = true;
                events[idx].player_2_name = target_name;
            }
        }
        if events[idx].shot {
            if let Some(pass_idx) = last_passing_idx {
                events[idx].player_2_name = events[pass_idx].player_name.clone();
                events[pass_idx].assist = true;
            }
        }
    }

    for idx in 1..events.len() {
        let previous = events[idx - 1].clone();
        if previous.shot
            && !previous.goal
            && player_team(&previous.player_name, players)
                != player_team(&events[idx].player_name, players)
        {
            events[idx].save = true;
        }
    }

    let kickoff_event_indices = kickoff_touch_event_indices(&events, kickoff_starts, goal_frames);
    let shooter_by_frame = events
        .iter()
        .map(|event| (event.frame_number, event.player_name.clone()))
        .collect::<HashMap<_, _>>();
    for (idx, event) in events.iter_mut().enumerate() {
        let is_kickoff_touch = kickoff_event_indices.contains(&idx);
        event.event_type = if is_kickoff_touch {
            "kickoff".to_string()
        } else if event.goal {
            "goal".to_string()
        } else if event.shot || event.save {
            "shot".to_string()
        } else if event.missed_shot {
            "missed-shot".to_string()
        } else if event.missed_pass {
            "missed-pass".to_string()
        } else if event.clear {
            "exit".to_string()
        } else if event.pass_ {
            "pass".to_string()
        } else {
            "touch".to_string()
        };
        if event.save {
            event.player_3_name = event.player_name.clone();
            if let Some(prev_frame) = event.previous_hit_frame_number {
                if let Some(previous_player_name) = shooter_by_frame.get(&prev_frame) {
                    event.player_name = previous_player_name.clone();
                }
            }
        }
    }
    events
}

fn player_team(name: &str, players: &[PlayerInfo]) -> Option<i32> {
    players
        .iter()
        .find(|player| player.name == name)
        .map(|player| player.team)
}

fn team_name(team: i32) -> &'static str {
    if team == 1 {
        "orange"
    } else {
        "blue"
    }
}

fn actual_team_size(players: &[PlayerInfo]) -> Option<i32> {
    let blue = players.iter().filter(|player| player.team == 0).count();
    let orange = players.iter().filter(|player| player.team == 1).count();
    let size = blue.max(orange);
    (size > 0).then_some(size as i32)
}

fn header_actual_team_size(replay: &Replay) -> Option<i32> {
    let players = pbp_players(replay);
    actual_team_size(&players)
}

fn kickoff_touch_event_indices(
    events: &[BallEvent],
    kickoff_starts: &[i32],
    goal_frames: &[(i32, String)],
) -> HashSet<usize> {
    let mut starts = std::iter::once(0)
        .chain(kickoff_starts.iter().copied())
        .chain(goal_frames.iter().map(|(frame, _)| frame.saturating_add(1)))
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();

    let mut selected = HashSet::new();
    let mut used_touch_frames = HashSet::new();
    for start in starts {
        let first_touch = events
            .iter()
            .enumerate()
            .filter(|(_, event)| {
                event.frame_number >= start
                    && event.frame_number - start <= POST_GOAL_KICKOFF_WINDOW_FRAMES
            })
            .min_by_key(|(_, event)| event.frame_number);
        if let Some((idx, event)) = first_touch {
            //Carball treats kickoffs as the first ball touch after each kickoff reset.
            //If multiple hit candidates land on the same frame, keep one.
            if used_touch_frames.insert(event.frame_number) {
                selected.insert(idx);
            }
        }
    }
    selected
}

fn kickoff_start_frames_from_resets(frames: &[FrameSnapshot]) -> Vec<i32> {
    let mut starts = Vec::new();
    let mut in_center_reset = false;
    let mut last_start = -10_000;
    for snapshot in frames {
        let ball = match snapshot.ball {
            Some(value) => value,
            None => continue,
        };
        let centered = ball.pos.x.abs() <= 25.0
            && ball.pos.y.abs() <= 25.0
            && (80.0..=110.0).contains(&ball.pos.z)
            && vec_norm(ball.vel) <= 150.0;
        if centered && !in_center_reset && snapshot.frame_number - last_start > 300 {
            starts.push(snapshot.frame_number);
            last_start = snapshot.frame_number;
        }
        in_center_reset = centered;
    }
    starts
}

fn apply_kickoff_boost_resets(frames: &mut [FrameSnapshot], kickoff_starts: &[i32]) {
    for (kickoff_idx, start_frame) in kickoff_starts.iter().copied().enumerate() {
        let next_start = kickoff_starts
            .get(kickoff_idx + 1)
            .copied()
            .unwrap_or(i32::MAX);
        let player_count = frames
            .iter()
            .find(|snapshot| snapshot.frame_number >= start_frame)
            .map(|snapshot| snapshot.players.len())
            .unwrap_or(0);
        let mut resetting = vec![true; player_count];

        for snapshot in frames.iter_mut().filter(|snapshot| {
            snapshot.frame_number >= start_frame && snapshot.frame_number < next_start
        }) {
            for (idx, player_state) in snapshot.players.iter_mut().enumerate() {
                if !resetting.get(idx).copied().unwrap_or(false) {
                    continue;
                }
                let Some(state) = player_state.as_mut() else {
                    continue;
                };
                if state
                    .boost_updated_frame
                    .map(|frame| frame > start_frame)
                    .unwrap_or(false)
                {
                    if let Some(slot) = resetting.get_mut(idx) {
                        *slot = false;
                    }
                    continue;
                }
                state.boost = Some(raw_boost_units(33));
                state.boost_collect = None;
                state.boost_updated_frame = Some(start_frame);
            }
        }
    }
}

fn filter_goal_to_kickoff_rows(rows: &mut Vec<PbpEventRecord>, goal_frames: &[(i32, String)]) {
    let kickoff_frames = rows
        .iter()
        .filter(|row| row.event_type == "kickoff")
        .filter_map(|row| row.frame_number)
        .collect::<Vec<_>>();
    let mut cutoff_goal_frames = goal_frames
        .iter()
        .map(|(frame, _)| *frame)
        .collect::<Vec<_>>();
    cutoff_goal_frames.extend(
        rows.iter()
            .filter(|row| row.event_type == "goal")
            .filter_map(|row| row.frame_number),
    );
    cutoff_goal_frames.sort_unstable();
    cutoff_goal_frames.dedup();
    rows.retain(|row| {
        let frame = match row.frame_number {
            Some(value) => value,
            None => return true,
        };
        if row.event_type == "goal" || row.event_type == "kickoff" {
            return true;
        }
        for goal_frame in &cutoff_goal_frames {
            if frame <= *goal_frame {
                continue;
            }
            let next_kickoff = kickoff_frames
                .iter()
                .copied()
                .filter(|kickoff_frame| *kickoff_frame > *goal_frame)
                .min();
            if next_kickoff
                .map(|kickoff_frame| frame < kickoff_frame)
                .unwrap_or(true)
            {
                return false;
            }
        }
        true
    });
}

fn apply_official_stats(
    rows: &mut Vec<PbpEventRecord>,
    official_stats: &[OfficialStatEvent],
    players: &[PlayerInfo],
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    context: &PbpContext,
    team_size: Option<i32>,
    game_time: &str,
) {
    for stat in official_stats {
        match stat.stat_type {
            "shot" | "goal" => {
                rows.push(build_official_stat_row(
                    stat,
                    rows,
                    players,
                    player_static_values,
                    game_id,
                    match_guid,
                    replay_name,
                    map_id,
                    context,
                    team_size,
                    game_time,
                ));
            }
            "assist" => {
                rows.push(build_official_assist_row(
                    stat,
                    rows,
                    players,
                    player_static_values,
                    game_id,
                    match_guid,
                    replay_name,
                    map_id,
                    context,
                    team_size,
                    game_time,
                ));
            }
            "save" => {
                //Save rows keep the recorded save frame while linking back to the observed shot.
                let row = build_official_save_row(
                    stat,
                    rows,
                    players,
                    player_static_values,
                    game_id,
                    match_guid,
                    replay_name,
                    map_id,
                    context,
                    team_size,
                    game_time,
                );
                rows.push(row);
            }
            _ => {}
        }
    }
}

fn build_official_stat_row(
    stat: &OfficialStatEvent,
    _rows: &[PbpEventRecord],
    players: &[PlayerInfo],
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    context: &PbpContext,
    team_size: Option<i32>,
    game_time: &str,
) -> PbpEventRecord {
    let player_name = official_stat_player_name(stat, players);
    let observed_frame =
        observed_frame_from_player_contact(context, players, &player_name, stat.frame_number, None);
    let mut values = pbp_base_values(
        game_id,
        match_guid,
        replay_name,
        map_id,
        context,
        team_size,
        game_time,
    );
    values.insert("event_type".to_string(), stat.stat_type.to_string());
    values.insert("frame_number".to_string(), stat.frame_number.to_string());
    values.insert(
        "observed_frame_number".to_string(),
        observed_frame.to_string(),
    );
    values.insert(
        "recorded_frame_number".to_string(),
        stat.frame_number.to_string(),
    );
    values.insert(format!("official_{}", stat.stat_type), "true".to_string());
    values.insert(
        format!("official_{}_count", stat.stat_type),
        "1".to_string(),
    );
    insert_seconds_elapsed(&mut values, context, observed_frame);
    add_event_player(&mut values, players, 1, &player_name);
    if let Some(player) = players.iter().find(|player| player.name == player_name) {
        values.insert("event_team".to_string(), team_name(player.team).to_string());
    }
    add_pbp_players(&mut values, player_static_values);
    add_frame_state_values(&mut values, context, observed_frame, players);
    PbpEventRecord {
        frame_number: Some(stat.frame_number),
        event_type: stat.stat_type.to_string(),
        values,
    }
}

fn build_official_assist_row(
    stat: &OfficialStatEvent,
    _rows: &[PbpEventRecord],
    players: &[PlayerInfo],
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    context: &PbpContext,
    team_size: Option<i32>,
    game_time: &str,
) -> PbpEventRecord {
    let player_name = official_stat_player_name(stat, players);
    let observed_frame =
        observed_frame_from_player_contact(context, players, &player_name, stat.frame_number, None);
    let mut values = pbp_base_values(
        game_id,
        match_guid,
        replay_name,
        map_id,
        context,
        team_size,
        game_time,
    );
    values.insert("event_type".to_string(), "assist".to_string());
    values.insert("frame_number".to_string(), stat.frame_number.to_string());
    values.insert(
        "observed_frame_number".to_string(),
        observed_frame.to_string(),
    );
    values.insert(
        "recorded_frame_number".to_string(),
        stat.frame_number.to_string(),
    );
    values.insert("official_assist".to_string(), "true".to_string());
    values.insert("official_assist_count".to_string(), "1".to_string());
    insert_seconds_elapsed(&mut values, context, observed_frame);
    add_event_player(&mut values, players, 1, &player_name);
    if let Some(player) = players.iter().find(|player| player.name == player_name) {
        values.insert("event_team".to_string(), team_name(player.team).to_string());
    }
    add_pbp_players(&mut values, player_static_values);
    add_frame_state_values(&mut values, context, observed_frame, players);
    PbpEventRecord {
        frame_number: Some(stat.frame_number),
        event_type: "assist".to_string(),
        values,
    }
}

fn build_official_save_row(
    stat: &OfficialStatEvent,
    rows: &[PbpEventRecord],
    players: &[PlayerInfo],
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    context: &PbpContext,
    team_size: Option<i32>,
    game_time: &str,
) -> PbpEventRecord {
    let player_name = official_stat_player_name(stat, players);
    let save_team = player_team(&player_name, players);
    let linked_shot = linked_shot_row(rows, save_team, stat.frame_number);
    let min_observed_frame = linked_shot
        .and_then(|row| row_i32(&row.values, "observed_frame_number").or(row.frame_number))
        .unwrap_or(i32::MIN);
    let observed_frame = observed_frame_from_player_contact(
        context,
        players,
        &player_name,
        stat.frame_number,
        Some(min_observed_frame),
    );
    let mut values = pbp_base_values(
        game_id,
        match_guid,
        replay_name,
        map_id,
        context,
        team_size,
        game_time,
    );
    values.insert("event_type".to_string(), "save".to_string());
    values.insert("frame_number".to_string(), stat.frame_number.to_string());
    values.insert(
        "observed_frame_number".to_string(),
        observed_frame.to_string(),
    );
    values.insert(
        "recorded_frame_number".to_string(),
        stat.frame_number.to_string(),
    );
    values.insert("official_save".to_string(), "true".to_string());
    values.insert("official_save_count".to_string(), "1".to_string());
    insert_seconds_elapsed(&mut values, context, observed_frame);
    add_event_player(&mut values, players, 1, &player_name);
    if let Some(player) = players.iter().find(|player| player.name == player_name) {
        values.insert("event_team".to_string(), team_name(player.team).to_string());
    }
    if let Some(shot) = linked_shot {
        values.insert(
            "linked_shot_observed_frame_number".to_string(),
            row_i32(&shot.values, "observed_frame_number")
                .or(shot.frame_number)
                .map(|frame| frame.to_string())
                .unwrap_or_default(),
        );
        values.insert(
            "linked_shot_recorded_frame_number".to_string(),
            row_string(&shot.values, "recorded_frame_number"),
        );
        add_event_player(
            &mut values,
            players,
            2,
            &row_string(&shot.values, "event_player_1_name"),
        );
    }
    add_pbp_players(&mut values, player_static_values);
    add_frame_state_values(&mut values, context, observed_frame, players);
    PbpEventRecord {
        frame_number: Some(stat.frame_number),
        event_type: "save".to_string(),
        values,
    }
}

fn official_stat_player_name(stat: &OfficialStatEvent, players: &[PlayerInfo]) -> String {
    stat.pri_actor_id
        .and_then(|actor_id| {
            players
                .iter()
                .find(|player| player.actor_id.parse::<i32>().ok() == Some(actor_id))
        })
        .map(|player| player.name.clone())
        .unwrap_or_else(|| stat.player_name.clone())
}

fn observed_frame_from_player_contact(
    context: &PbpContext,
    players: &[PlayerInfo],
    player_name: &str,
    recorded_frame_number: i32,
    min_frame: Option<i32>,
) -> i32 {
    let Some(player_idx) = players.iter().position(|player| player.name == player_name) else {
        return recorded_frame_number;
    };
    let player = &players[player_idx];
    let car_id = player.car_id.parse().unwrap_or(23);
    context
        .frame_states
        .iter()
        .rev()
        .filter(|snapshot| {
            snapshot.frame_number <= recorded_frame_number
                && min_frame
                    .map(|min_frame| snapshot.frame_number >= min_frame)
                    .unwrap_or(true)
        })
        .find(|snapshot| {
            let Some(ball) = snapshot.ball.filter(|state| state.has_pos) else {
                return false;
            };
            let Some(player_state) = snapshot.players.get(player_idx).and_then(Option::as_ref)
            else {
                return false;
            };
            if !player_state.entity.has_pos {
                return false;
            }
            let collision_distance = ball_collision_distance(ball.pos, player_state.entity, car_id);
            collision_distance <= 300.0
        })
        .map(|snapshot| snapshot.frame_number)
        .unwrap_or(recorded_frame_number)
}

fn linked_shot_row(
    rows: &[PbpEventRecord],
    save_team: Option<i32>,
    recorded_frame_number: i32,
) -> Option<&PbpEventRecord> {
    rows.iter()
        .filter(|row| {
            truthy(row.values.get("official_shot"))
                && match save_team {
                    Some(1) => row_string(&row.values, "event_player_1_team") != "orange",
                    Some(_) => row_string(&row.values, "event_player_1_team") != "blue",
                    None => true,
                }
        })
        .filter_map(|row| {
            let frame = row_i32(&row.values, "recorded_frame_number")?;
            (frame <= recorded_frame_number && recorded_frame_number - frame <= 360)
                .then_some((row, frame))
        })
        .max_by_key(|(_, frame)| *frame)
        .map(|(row, _)| row)
}

fn collapse_duplicate_official_saves(rows: &mut Vec<PbpEventRecord>) {
    let mut indexes: HashMap<(Option<i32>, String, String, String, String), usize> = HashMap::new();
    let mut collapsed: Vec<PbpEventRecord> = Vec::with_capacity(rows.len());

    for row in rows.drain(..) {
        if row.event_type != "save" || !truthy(row.values.get("official_save")) {
            collapsed.push(row);
            continue;
        }
        let key = (
            row.frame_number,
            row_string(&row.values, "recorded_frame_number"),
            row_string(&row.values, "event_player_1_name"),
            row_string(&row.values, "event_player_2_name"),
            row_string(&row.values, "linked_shot_observed_frame_number"),
        );
        if let Some(&idx) = indexes.get(&key) {
            let count = row_i32(&row.values, "official_save_count").unwrap_or(1);
            let prior = row_i32(&collapsed[idx].values, "official_save_count").unwrap_or(1);
            collapsed[idx].values.insert(
                "official_save_count".to_string(),
                (prior + count).to_string(),
            );
            continue;
        }
        indexes.insert(key, collapsed.len());
        collapsed.push(row);
    }

    *rows = collapsed;
}

fn audit_pbp_stats(
    game_id: &str,
    replay: &Replay,
    rows: &[PbpEventRecord],
    context: &PbpContext,
) -> Result<()> {
    if context
        .official_stats
        .iter()
        .any(|stat| stat.stat_type == "goal")
    {
        return Ok(());
    }

    let mut actual: HashMap<String, (i32, i32, i32, i32)> = HashMap::new();
    for row in rows {
        let player_1 = row_string(&row.values, "event_player_1_name");
        if !player_1.is_empty() {
            let entry = actual.entry(player_1).or_default();
            if truthy(row.values.get("official_goal")) {
                entry.0 += row_i32(&row.values, "official_goal_count").unwrap_or(1);
            }
            if truthy(row.values.get("official_save")) {
                entry.2 += row_i32(&row.values, "official_save_count").unwrap_or(1);
            }
            if truthy(row.values.get("official_shot")) {
                entry.3 += row_i32(&row.values, "official_shot_count").unwrap_or(1);
            }
        }

        let player_2 = row_string(&row.values, "event_player_2_name");
        if !player_2.is_empty() && truthy(row.values.get("official_assist")) {
            let entry = actual.entry(player_2).or_default();
            entry.1 += row_i32(&row.values, "official_assist_count").unwrap_or(1);
        }
    }

    let mut mismatches = Vec::new();
    if let Some(players) = header_array(replay, "PlayerStats") {
        for player in players {
            let name = prop_string(player, "Name")
                .or_else(|| prop_string(player, "PlayerName"))
                .unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let (goals, _, _, _) = actual.get(&name).copied().unwrap_or_default();
            if let Some(expected) = prop_i32(player, "Goals") {
                if expected != goals {
                    mismatches.push(format!("{name} goals header={expected} pbp={goals}"));
                }
            }
        }
    }
    if !mismatches.is_empty() {
        //Only header-sourced goals are audited here; PRI stat events are authoritative when recorded.
        eprintln!("PBP stat mismatch {game_id}: {}", mismatches.join("; "));
    }
    Ok(())
}

fn ball_collision_distance(ball_pos: Vec3, car_state: EntityState, car_id: i32) -> f32 {
    let local = inverse_rotate(
        car_state.rot,
        Vec3 {
            x: ball_pos.x - car_state.pos.x,
            y: ball_pos.y - car_state.pos.y,
            z: ball_pos.z - car_state.pos.z,
        },
    );
    let (length, width, height, offset, elevation) = hitbox_dims(car_id);
    let x_lims = (-length / 2.0 + offset, length / 2.0 + offset);
    let y_lims = (-width / 2.0, width / 2.0);
    let z_lims = (-height / 2.0 + elevation, height / 2.0 + elevation);
    let dx = axis_distance(local.x, x_lims);
    let dy = axis_distance(local.y, y_lims);
    let dz = axis_distance(local.z, z_lims);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn hitbox_dims(car_id: i32) -> (f32, f32, f32, f32, f32) {
    match car_id {
        403 => (127.9268, 83.27995, 31.3, 9.0, 15.75),
        _ => (118.0074, 84.2, 36.15907, 13.87566, 20.755),
    }
}

fn axis_distance(value: f32, lims: (f32, f32)) -> f32 {
    if value < lims.0 {
        (lims.0 - value).abs()
    } else if value > lims.1 {
        (lims.1 - value).abs()
    } else {
        0.0
    }
}

fn inverse_rotate(q: Quat, v: Vec3) -> Vec3 {
    rotate(
        Quat {
            x: -q.x,
            y: -q.y,
            z: -q.z,
            w: q.w,
        },
        v,
    )
}

fn rotate(q: Quat, v: Vec3) -> Vec3 {
    let ux = q.x;
    let uy = q.y;
    let uz = q.z;
    let s = q.w;
    let dot_uv = ux * v.x + uy * v.y + uz * v.z;
    let dot_uu = ux * ux + uy * uy + uz * uz;
    Vec3 {
        x: 2.0 * dot_uv * ux + (s * s - dot_uu) * v.x + 2.0 * s * (uy * v.z - uz * v.y),
        y: 2.0 * dot_uv * uy + (s * s - dot_uu) * v.y + 2.0 * s * (uz * v.x - ux * v.z),
        z: 2.0 * dot_uv * uz + (s * s - dot_uu) * v.z + 2.0 * s * (ux * v.y - uy * v.x),
    }
}

fn vec_distance(left: Vec3, right: Vec3) -> f32 {
    ((right.x - left.x).powi(2) + (right.y - left.y).powi(2) + (right.z - left.z).powi(2)).sqrt()
}

fn vec_dot(left: Vec3, right: Vec3) -> f32 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

fn speed_toward_point(velocity: Vec3, from: Vec3, to: Vec3) -> Option<f32> {
    let direction = Vec3 {
        x: to.x - from.x,
        y: to.y - from.y,
        z: to.z - from.z,
    };
    let distance = vec_norm(direction);
    if distance <= f32::EPSILON {
        return None;
    }
    Some(vec_dot(velocity, direction) / distance)
}

fn point_to_segment_distance_2d(point: Vec3, start: Vec3, end: Vec3) -> f32 {
    let segment_x = end.x - start.x;
    let segment_y = end.y - start.y;
    let length_squared = segment_x * segment_x + segment_y * segment_y;
    if length_squared <= f32::EPSILON {
        return ((point.x - start.x).powi(2) + (point.y - start.y).powi(2)).sqrt();
    }
    let t = (((point.x - start.x) * segment_x + (point.y - start.y) * segment_y) / length_squared)
        .clamp(0.0, 1.0);
    let closest_x = start.x + t * segment_x;
    let closest_y = start.y + t * segment_y;
    ((point.x - closest_x).powi(2) + (point.y - closest_y).powi(2)).sqrt()
}

fn whiff_like_miss(
    previous_player_pos: Vec3,
    player_pos: Vec3,
    previous_ball_pos: Vec3,
    ball_pos: Vec3,
) -> bool {
    let previous_distance = vec_distance(previous_player_pos, previous_ball_pos);
    let current_distance = vec_distance(player_pos, ball_pos);
    if previous_distance > WHIFF_PREVIOUS_BALL_DISTANCE && current_distance > WHIFF_BALL_DISTANCE {
        return false;
    }
    let separating_after_approach = current_distance >= previous_distance + 25.0;

    let previous_relative = Vec3 {
        x: previous_player_pos.x - previous_ball_pos.x,
        y: previous_player_pos.y - previous_ball_pos.y,
        z: previous_player_pos.z - previous_ball_pos.z,
    };
    let current_relative = Vec3 {
        x: player_pos.x - ball_pos.x,
        y: player_pos.y - ball_pos.y,
        z: player_pos.z - ball_pos.z,
    };
    let relative_crossed = vec_dot(previous_relative, current_relative) < 0.0
        && current_distance <= WHIFF_BALL_DISTANCE;
    let player_path_missed_ball =
        point_to_segment_distance_2d(ball_pos, previous_player_pos, player_pos)
            <= WHIFF_CROSS_BALL_DISTANCE;
    let ball_path_missed_player =
        point_to_segment_distance_2d(player_pos, previous_ball_pos, ball_pos)
            <= WHIFF_CROSS_BALL_DISTANCE;

    separating_after_approach
        && (relative_crossed || player_path_missed_ball || ball_path_missed_player)
}

fn is_shot(event: &BallEvent, players: &[PlayerInfo]) -> bool {
    let team = player_team(&event.player_name, players).unwrap_or(0);
    let toward_orange_goal = team == 0 && event.ball_state.vel.y > 0.0;
    let toward_blue_goal = team == 1 && event.ball_state.vel.y < 0.0;
    if !(toward_orange_goal || toward_blue_goal) {
        return false;
    }
    let goal_y = if team == 0 { 5140.0 } else { -5140.0 };
    if event.ball_state.vel.y.abs() < 1.0 {
        return false;
    }
    let t = (goal_y - event.ball_state.pos.y) / event.ball_state.vel.y;
    if !(0.0..=3.0).contains(&t) {
        return false;
    }
    let x_at_goal = event.ball_state.pos.x + event.ball_state.vel.x * t;
    let z_at_goal = event.ball_state.pos.z + event.ball_state.vel.z * t - 0.5 * GRAVITY * t * t;
    x_at_goal.abs() <= GOAL_CENTER_TO_POST && z_at_goal > 0.0 && z_at_goal <= GOAL_HEIGHT
}

fn is_missed_shot(event: &BallEvent, players: &[PlayerInfo]) -> bool {
    let team = player_team(&event.player_name, players).unwrap_or(0);
    let toward_orange_goal = team == 0 && event.ball_state.vel.y > 0.0;
    let toward_blue_goal = team == 1 && event.ball_state.vel.y < 0.0;
    if !(toward_orange_goal || toward_blue_goal) || event.ball_state.vel.y.abs() < 1.0 {
        return false;
    }
    let goal_y = if team == 0 { BACK_WALL_Y } else { -BACK_WALL_Y };
    let t = (goal_y - event.ball_state.pos.y) / event.ball_state.vel.y;
    if !(0.0..=3.0).contains(&t) {
        return false;
    }
    let x_at_goal = event.ball_state.pos.x + event.ball_state.vel.x * t;
    let z_at_goal = event.ball_state.pos.z + event.ball_state.vel.z * t - 0.5 * GRAVITY * t * t;
    z_at_goal > 0.0
        && z_at_goal <= MISSED_SHOT_MAX_HEIGHT
        && x_at_goal.abs() <= GOAL_CENTER_TO_POST + MISSED_SHOT_MAX_LATERAL_MISS
        && (x_at_goal.abs() > GOAL_CENTER_TO_POST || z_at_goal > GOAL_HEIGHT)
}

fn missed_pass_target(event: &BallEvent, players: &[PlayerInfo]) -> Option<String> {
    let passer_team = player_team(&event.player_name, players)?;
    let ball_speed = vec_norm(event.ball_state.vel);
    if ball_speed < MISSED_PASS_MIN_SPEED {
        return None;
    }

    let mut best_target: Option<(String, f32, f32)> = None;
    for (player_idx, player) in players.iter().enumerate() {
        if player.team != passer_team || player.name == event.player_name {
            continue;
        }
        let Some(target_pos) = event.player_positions.get(player_idx).copied().flatten() else {
            continue;
        };
        let to_target = Vec3 {
            x: target_pos.x - event.ball_state.pos.x,
            y: target_pos.y - event.ball_state.pos.y,
            z: target_pos.z - event.ball_state.pos.z,
        };
        let target_distance = vec_norm(to_target);
        if target_distance <= f32::EPSILON {
            continue;
        }
        let forward_dot = vec_dot(event.ball_state.vel, to_target) / (ball_speed * target_distance);
        if forward_dot < MISSED_PASS_MIN_FORWARD_DOT {
            continue;
        }
        let miss_distance = projected_ball_target_distance(event, target_pos);
        if miss_distance <= MISSED_PASS_TARGET_RADIUS || miss_distance > MISSED_PASS_MAX_TARGET_MISS
        {
            continue;
        }
        match &best_target {
            Some((_, best_miss, _)) if miss_distance >= *best_miss => {}
            _ => best_target = Some((player.name.clone(), miss_distance, target_distance)),
        }
    }

    best_target.map(|(name, _, _)| name)
}

fn projected_ball_target_distance(event: &BallEvent, target_pos: Vec3) -> f32 {
    let mut best = f32::MAX;
    for sample in 1..=12 {
        let t = MISSED_PASS_PROJECTION_SECONDS * sample as f32 / 12.0;
        let predicted = Vec3 {
            x: event.ball_state.pos.x + event.ball_state.vel.x * t,
            y: event.ball_state.pos.y + event.ball_state.vel.y * t,
            z: (event.ball_state.pos.z + event.ball_state.vel.z * t - 0.5 * GRAVITY * t * t)
                .max(BALL_RADIUS),
        };
        best = best.min(vec_distance(predicted, target_pos));
    }
    best
}

fn is_clear(event: &BallEvent, next_event: Option<&BallEvent>, players: &[PlayerInfo]) -> bool {
    let team = player_team(&event.player_name, players).unwrap_or(0);
    let y = event.ball_state.pos.y;
    let clear_buffer = 400.0;
    let defending = if team == 1 {
        y > BACK_WALL_Y / 3.0 + clear_buffer
    } else {
        y < -BACK_WALL_Y / 3.0 - clear_buffer
    };
    if !defending {
        return false;
    }
    if let Some(next) = next_event {
        if team == 1 {
            next.ball_state.pos.y < BACK_WALL_Y / 3.0 - clear_buffer
        } else {
            next.ball_state.pos.y > -BACK_WALL_Y / 3.0 + clear_buffer
        }
    } else {
        event.distance > clear_buffer
    }
}

fn distance_to_goal(event: &BallEvent, players: &[PlayerInfo]) -> f32 {
    let team = player_team(&event.player_name, players).unwrap_or(0);
    let goal_y = if team == 1 { -BACK_WALL_Y } else { BACK_WALL_Y };
    let goal_x = event
        .ball_state
        .pos
        .x
        .clamp(-GOAL_CENTER_TO_POST, GOAL_CENTER_TO_POST);
    ((event.ball_state.pos.x - goal_x).powi(2) + (event.ball_state.pos.y - goal_y).powi(2)).sqrt()
}

fn header_goal_frames(replay: &Replay) -> Vec<(i32, String)> {
    header_array(replay, "Goals")
        .map(|goals| {
            goals
                .iter()
                .filter_map(|goal| {
                    Some((
                        prop_i32(goal, "frame")
                            .or_else(|| prop_i32(goal, "Frame"))
                            .or_else(|| prop_i32(goal, "Time"))?,
                        prop_string(goal, "PlayerName").unwrap_or_default(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn goal_number_for_frame(frame_number: i32, goal_frames: &[(i32, String)]) -> i32 {
    goal_frames
        .iter()
        .filter(|(goal_frame, _)| frame_number <= *goal_frame)
        .count()
        .try_into()
        .unwrap_or(0)
}

fn infer_team_number(
    team_actor_id: i32,
    pri_name: &HashMap<i32, String>,
    pri_team_actor: &HashMap<i32, i32>,
    player_team_by_name: &HashMap<String, i32>,
) -> Option<i32> {
    pri_team_actor.iter().find_map(|(pri_actor_id, actor_id)| {
        if *actor_id != team_actor_id {
            return None;
        }
        let name = pri_name.get(pri_actor_id)?;
        player_team_by_name.get(name).copied()
    })
}

fn loadout_body_id(value: &Value, team: i32) -> Option<i32> {
    loadout_item_id(value, team, "body")
}

fn loadout_item_id(value: &Value, team: i32, item: &str) -> Option<i32> {
    let side = if team == 1 { "orange" } else { "blue" };
    value
        .get("TeamLoadout")?
        .get(side)?
        .get(item)?
        .as_i64()
        .and_then(|value| i32::try_from(value).ok())
}

fn assign_player_loadout(player: &mut PlayerInfo, value: &Value) {
    if let Some(car_id) = loadout_body_id(value, player.team) {
        player.car_id = car_id.to_string();
        player.car_name = car_name(car_id).to_string();
    }
    for (field, item) in [
        (&mut player.decal_id, "decal"),
        (&mut player.wheels_id, "wheels"),
        (&mut player.boost_id, "boost"),
        (&mut player.antenna_id, "antenna"),
        (&mut player.topper_id, "topper"),
        (&mut player.engine_audio_id, "engine_audio"),
        (&mut player.trail_id, "trail"),
        (&mut player.goal_explosion_id, "goal_explosion"),
        (&mut player.primary_paint_finish_id, "paint_finish"),
        (&mut player.accent_paint_finish_id, "accent_paint_finish"),
    ] {
        if let Some(value) = loadout_item_id(value, player.team, item) {
            *field = value.to_string();
        }
    }
}

fn playlist_name(value: i32) -> &'static str {
    match value {
        1 => "DUEL",
        2 => "DOUBLES",
        3 => "STANDARD",
        4 => "CHAOS",
        6 => "CUSTOM_LOBBY",
        10 => "RANKED_DUEL",
        11 => "RANKED_DOUBLES",
        13 => "RANKED_STANDARD",
        _ => "UNKNOWN",
    }
}

fn car_name(car_id: i32) -> &'static str {
    match car_id {
        23 => "Octane",
        403 => "Dominus",
        4284 => "Fennec",
        _ => "",
    }
}

fn pbp_player_static_values(players: &[PlayerInfo]) -> Vec<(String, String)> {
    let mut values = Vec::with_capacity(players.len() * 28);
    for player in players {
        values.push((format!("{}_id", player.slot), player.id.clone()));
        values.push((format!("{}_actor_id", player.slot), player.actor_id.clone()));
        values.push((
            format!("{}_network_id", player.slot),
            player.network_id.clone(),
        ));
        values.push((format!("{}_name", player.slot), player.name.clone()));
        values.push((format!("{}_platform", player.slot), player.platform.clone()));
        values.push((format!("{}_is_bot", player.slot), player.is_bot.clone()));
        values.push((format!("{}_score", player.slot), player.score.clone()));
        values.push((
            format!("{}_time_in_game", player.slot),
            player.time_in_game.clone(),
        ));
        values.push((format!("{}_car_id", player.slot), player.car_id.clone()));
        values.push((format!("{}_car_name", player.slot), player.car_name.clone()));
        values.push((format!("{}_decal_id", player.slot), player.decal_id.clone()));
        values.push((
            format!("{}_wheels_id", player.slot),
            player.wheels_id.clone(),
        ));
        values.push((format!("{}_boost_id", player.slot), player.boost_id.clone()));
        values.push((
            format!("{}_antenna_id", player.slot),
            player.antenna_id.clone(),
        ));
        values.push((
            format!("{}_topper_id", player.slot),
            player.topper_id.clone(),
        ));
        values.push((
            format!("{}_engine_audio_id", player.slot),
            player.engine_audio_id.clone(),
        ));
        values.push((format!("{}_trail_id", player.slot), player.trail_id.clone()));
        values.push((
            format!("{}_goal_explosion_id", player.slot),
            player.goal_explosion_id.clone(),
        ));
        values.push((
            format!("{}_primary_paint_finish_id", player.slot),
            player.primary_paint_finish_id.clone(),
        ));
        values.push((
            format!("{}_accent_paint_finish_id", player.slot),
            player.accent_paint_finish_id.clone(),
        ));
        if let Some(camera) = player.camera_settings {
            values.push((
                format!("{}_camera_fov", player.slot),
                camera.fov.to_string(),
            ));
            values.push((
                format!("{}_camera_height", player.slot),
                camera.height.to_string(),
            ));
            values.push((
                format!("{}_camera_angle", player.slot),
                camera.angle.to_string(),
            ));
            values.push((
                format!("{}_camera_distance", player.slot),
                camera.distance.to_string(),
            ));
            values.push((
                format!("{}_camera_stiffness", player.slot),
                camera.stiffness.to_string(),
            ));
            values.push((
                format!("{}_camera_swivel", player.slot),
                camera.swivel.to_string(),
            ));
            values.push((
                format!("{}_camera_transition", player.slot),
                camera
                    .transition
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
            ));
        }
    }
    values
}

fn add_pbp_players(values: &mut RowValues, player_static_values: &[(String, String)]) {
    values.extend(player_static_values.iter().cloned());
}

fn add_event_player(
    values: &mut RowValues,
    players: &[PlayerInfo],
    event_player_number: usize,
    player_name: &str,
) {
    values.insert(
        format!("event_player_{event_player_number}_name"),
        player_name.to_string(),
    );
    if let Some(player) = players.iter().find(|player| player.name == player_name) {
        values.insert(
            format!("event_player_{event_player_number}_id"),
            player.id.clone(),
        );
        values.insert(
            format!("event_player_{event_player_number}_actor_id"),
            player.actor_id.clone(),
        );
        values.insert(
            format!("event_player_{event_player_number}_team"),
            team_name(player.team).to_string(),
        );
    }
}

fn pbp_base_values(
    game_id: &str,
    _match_guid: &str,
    _replay_name: &str,
    _map_id: &str,
    context: &PbpContext,
    team_size: Option<i32>,
    _game_time: &str,
) -> RowValues {
    let mut values = RowValues::new();
    values.insert_utf8("game_id", game_id.to_string());
    values.insert_utf8("blue_team_name", context.blue_team_name.clone());
    values.insert_utf8("orange_team_name", context.orange_team_name.clone());
    if let Some(size) = team_size {
        values.insert_i32("team_size", size);
    }
    values
}

fn frame_snapshot(context: &PbpContext, frame_number: i32) -> Option<&FrameSnapshot> {
    context
        .frame_states
        .get(usize::try_from(frame_number).ok()?)
        .filter(|snapshot| snapshot.frame_number == frame_number)
}

fn add_frame_seconds_elapsed(frames: &mut [FrameSnapshot]) {
    let mut saw_regulation_zero = false;
    let mut overtime = false;
    let mut previous_clock: Option<i32> = None;
    for snapshot in frames {
        let Some(clock) = snapshot.seconds_remaining else {
            snapshot.seconds_elapsed = Some(snapshot.frame_number as f32 / 30.0);
            continue;
        };
        if clock <= 0 {
            saw_regulation_zero = true;
            if clock < 0 {
                overtime = true;
            }
        }
        if saw_regulation_zero {
            if let Some(previous) = previous_clock {
                if clock > previous && clock >= 0 && previous <= 5 {
                    overtime = true;
                }
            }
        }
        snapshot.seconds_elapsed = Some(if overtime {
            300.0 + clock.abs() as f32
        } else {
            300.0 - clock as f32
        });
        previous_clock = Some(clock);
    }
}

fn game_seconds_elapsed(context: &PbpContext, frame_number: i32) -> f32 {
    frame_snapshot(context, frame_number)
        .and_then(|snapshot| snapshot.seconds_elapsed)
        .unwrap_or(frame_number as f32 / 30.0)
}

fn insert_seconds_elapsed(values: &mut RowValues, context: &PbpContext, frame_number: i32) {
    values.insert_f32(
        "seconds_elapsed",
        game_seconds_elapsed(context, frame_number),
    );
}

fn add_frame_state_values(
    values: &mut RowValues,
    context: &PbpContext,
    frame_number: i32,
    players: &[PlayerInfo],
) {
    let snapshot = match frame_snapshot(context, frame_number) {
        Some(value) => value,
        None => return,
    };
    if let Some(ball) = snapshot.ball {
        add_entity_state_values(values, "ball", ball);
    }
    for (idx, player) in players.iter().enumerate() {
        if let Some(state) = snapshot.players.get(idx).and_then(Option::as_ref) {
            let slot = &player.slot;
            add_entity_state_values(values, slot, state.entity);
            insert_opt(
                values,
                &format!("{slot}_boost_raw"),
                state.boost.map(i32::from),
            );
            insert_opt(
                values,
                &format!("{slot}_boost"),
                state.boost.map(i32::from).map(boost_units),
            );
            values.insert_bool(&format!("{slot}_boost_active"), state.boost_active);
            insert_opt(
                values,
                &format!("{slot}_boost_collect"),
                state.boost_collect.map(i32::from),
            );
            insert_opt(values, &format!("{slot}_throttle"), state.throttle);
            insert_opt(values, &format!("{slot}_steer"), state.steer);
            values.insert_bool(&format!("{slot}_handbrake"), state.handbrake);
            values.insert_bool(&format!("{slot}_ball_cam"), state.ball_cam);
            values.insert_bool(&format!("{slot}_dodge_active"), state.dodge_active);
            values.insert_bool(&format!("{slot}_jump_active"), state.jump_active);
            values.insert_bool(
                &format!("{slot}_double_jump_active"),
                state.double_jump_active,
            );
            values.insert_bool(&format!("{slot}_jumped"), state.jumped);
            values.insert_bool(&format!("{slot}_flipped"), state.flipped);
            insert_opt(
                values,
                &format!("{slot}_jump_air_activate_count"),
                state.jump_air_activate_count,
            );
            insert_opt(
                values,
                &format!("{slot}_double_jump_air_activate_count"),
                state.double_jump_air_activate_count,
            );
            insert_opt(
                values,
                &format!("{slot}_dodge_air_activate_count"),
                state.dodge_air_activate_count,
            );
            insert_opt(
                values,
                &format!("{slot}_dodges_refreshed_counter"),
                state.dodges_refreshed_counter,
            );
            values.insert_bool(&format!("{slot}_supersonic"), state.supersonic);
            values.insert_bool(&format!("{slot}_flip_available"), state.flip_available);
        }
    }
    add_spatial_features_from_snapshot(values, snapshot, players);
}

fn add_spatial_features_from_snapshot(
    values: &mut RowValues,
    snapshot: &FrameSnapshot,
    players: &[PlayerInfo],
) {
    let ball = snapshot
        .ball
        .filter(|state| state.has_pos)
        .map(|state| state.pos);
    let player_positions = players
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            snapshot
                .players
                .get(idx)
                .and_then(Option::as_ref)
                .filter(|state| state.entity.has_pos)
                .map(|state| state.entity.pos)
        })
        .collect::<Vec<_>>();

    for (player_idx, player) in players.iter().enumerate() {
        let slot = &player.slot;
        let pos = player_positions.get(player_idx).copied().flatten();
        let own_net = defensive_net(player.team);
        let opp_net = offensive_net(player.team);
        set_float(
            values,
            &format!("{slot}_distance_to_ball"),
            distance_opt(pos, ball),
        );
        set_float(
            values,
            &format!("{slot}_angle_to_ball"),
            angle_opt(pos, ball),
        );
        set_float(
            values,
            &format!("{slot}_distance_to_own_net"),
            distance_opt(pos, own_net),
        );
        set_float(
            values,
            &format!("{slot}_angle_to_own_net"),
            angle_opt(pos, own_net),
        );
        set_float(
            values,
            &format!("{slot}_distance_to_opp_net"),
            distance_opt(pos, opp_net),
        );
        set_float(
            values,
            &format!("{slot}_angle_to_opp_net"),
            angle_opt(pos, opp_net),
        );
        if let (Some(pos), Some(ball)) = (pos, ball) {
            let player_ball_distance = vec_distance(pos, ball);
            let closer_teammates = players
                .iter()
                .enumerate()
                .filter(|(teammate_idx, teammate)| {
                    *teammate_idx != player_idx && teammate.team == player.team
                })
                .filter_map(|(teammate_idx, _)| {
                    player_positions.get(teammate_idx).copied().flatten()
                })
                .filter(|teammate_pos| vec_distance(*teammate_pos, ball) < player_ball_distance)
                .count();
            values.insert_i32(
                &format!("{slot}_rotation_role"),
                (closer_teammates + 1) as i32,
            );
        }
    }

    for (source_idx, source) in players.iter().enumerate() {
        let source_pos = player_positions.get(source_idx).copied().flatten();
        for (target_idx, target) in players.iter().enumerate() {
            if source_idx == target_idx {
                continue;
            }
            let target_pos = player_positions.get(target_idx).copied().flatten();
            set_float(
                values,
                &format!("{}_distance_to_{}", source.slot, target.slot),
                distance_opt(source_pos, target_pos),
            );
        }
    }
}

fn add_entity_state_values(values: &mut RowValues, prefix: &str, state: EntityState) {
    if !state.has_pos {
        return;
    }
    values.insert_f32(&format!("{prefix}_pos_x"), state.pos.x);
    values.insert_f32(&format!("{prefix}_pos_y"), state.pos.y);
    values.insert_f32(&format!("{prefix}_pos_z"), state.pos.z);
    values.insert_f32(&format!("{prefix}_vel_x"), state.vel.x);
    values.insert_f32(&format!("{prefix}_vel_y"), state.vel.y);
    values.insert_f32(&format!("{prefix}_vel_z"), state.vel.z);
    values.insert_f32(&format!("{prefix}_ang_vel_x"), state.ang_vel.x);
    values.insert_f32(&format!("{prefix}_ang_vel_y"), state.ang_vel.y);
    values.insert_f32(&format!("{prefix}_ang_vel_z"), state.ang_vel.z);
    values.insert_f32(&format!("{prefix}_rot_x"), state.rot.x);
    values.insert_f32(&format!("{prefix}_rot_y"), state.rot.y);
    values.insert_f32(&format!("{prefix}_rot_z"), state.rot.z);
}

fn insert_opt(values: &mut RowValues, key: &str, value: Option<i32>) {
    if let Some(value) = value {
        values.insert_i32(key, value);
    }
}

trait LookupString {
    fn and_then_lookup(&self, lookup: &HashMap<String, String>) -> Option<String>;
}

impl LookupString for String {
    fn and_then_lookup(&self, lookup: &HashMap<String, String>) -> Option<String> {
        lookup.get(self).cloned()
    }
}

fn cell_to_string(value: &CellValue) -> String {
    match value {
        CellValue::Utf8(value) => value.clone(),
        CellValue::Int32(value) => value.to_string(),
        CellValue::Float32(value) => value.to_string(),
        CellValue::Boolean(value) => value.to_string(),
    }
}

fn row_string(values: &RowValues, key: &str) -> String {
    values.get(key).map(cell_to_string).unwrap_or_default()
}

fn row_f32(values: &RowValues, key: &str) -> Option<f32> {
    parse_f32(values.get(key))
}

fn row_i32(values: &RowValues, key: &str) -> Option<i32> {
    match values.get(key) {
        Some(CellValue::Int32(value)) => Some(*value),
        Some(value) => cell_to_string(value).parse::<i32>().ok(),
        None => None,
    }
}

fn parse_f32(value: Option<&CellValue>) -> Option<f32> {
    match value {
        Some(CellValue::Float32(value)) => Some(*value),
        Some(CellValue::Int32(value)) => Some(*value as f32),
        Some(value) => cell_to_string(value).parse::<f32>().ok(),
        None => None,
    }
}

fn truthy(value: Option<&CellValue>) -> bool {
    match value {
        Some(CellValue::Boolean(value)) => *value,
        Some(CellValue::Int32(value)) => *value != 0,
        Some(value) => matches!(
            cell_to_string(value).to_ascii_lowercase().as_str(),
            "true" | "1" | "yes"
        ),
        None => false,
    }
}

fn row_vec(values: &RowValues, prefix: &str, field_prefix: &str) -> Option<Vec3> {
    Some(Vec3 {
        x: row_f32(values, &format!("{prefix}_{field_prefix}_x"))?,
        y: row_f32(values, &format!("{prefix}_{field_prefix}_y"))?,
        z: row_f32(values, &format!("{prefix}_{field_prefix}_z"))?,
    })
}

fn set_float(values: &mut RowValues, key: &str, value: Option<f32>) {
    if let Some(value) = value {
        values.insert_f32(key, value);
    }
}

fn distance_opt(start: Option<Vec3>, end: Option<Vec3>) -> Option<f32> {
    Some(vec_distance(start?, end?))
}

fn angle_opt(start: Option<Vec3>, end: Option<Vec3>) -> Option<f32> {
    let start = start?;
    let end = end?;
    Some((end.y - start.y).atan2(end.x - start.x))
}

fn angle_delta_opt(previous: Option<f32>, current: Option<f32>) -> Option<f32> {
    let mut delta = current? - previous?;
    while delta > std::f32::consts::PI {
        delta -= 2.0 * std::f32::consts::PI;
    }
    while delta < -std::f32::consts::PI {
        delta += 2.0 * std::f32::consts::PI;
    }
    Some(delta)
}

fn defensive_net(team: i32) -> Option<Vec3> {
    Some(if team == 1 {
        Vec3 {
            x: 0.0,
            y: BACK_NET_Y,
            z: 0.0,
        }
    } else {
        Vec3 {
            x: 0.0,
            y: -BACK_NET_Y,
            z: 0.0,
        }
    })
}

fn offensive_net(team: i32) -> Option<Vec3> {
    Some(if team == 1 {
        Vec3 {
            x: 0.0,
            y: -BACK_NET_Y,
            z: 0.0,
        }
    } else {
        Vec3 {
            x: 0.0,
            y: BACK_NET_Y,
            z: 0.0,
        }
    })
}

fn swap_event_players(values: &mut RowValues) {
    for field in ["id", "name", "team"] {
        let left = format!("event_player_1_{field}");
        let right = format!("event_player_2_{field}");
        let left_value = row_string(values, &left);
        let right_value = row_string(values, &right);
        values.insert(left, right_value);
        values.insert(right, left_value);
    }
}

fn player_flipped(
    values: &RowValues,
    slot_by_id: &HashMap<String, String>,
    player_id: &str,
) -> bool {
    slot_by_id
        .get(player_id)
        .map(|slot| truthy(values.get(&format!("{slot}_flipped"))))
        .unwrap_or(false)
}

fn hood_dribble_control(
    values: &RowValues,
    slot_by_id: &HashMap<String, String>,
    player_id: &str,
) -> bool {
    let slot = match slot_by_id.get(player_id) {
        Some(value) => value,
        None => return false,
    };
    let player = match row_vec(values, slot, "pos") {
        Some(value) => value,
        None => return false,
    };
    let ball = row_vec(values, "ball", "pos").or_else(|| row_vec(values, "event_ball", "pos"));
    let ball = match ball {
        Some(value) => value,
        None => return false,
    };
    let horizontal_distance = ((ball.x - player.x).powi(2) + (ball.y - player.y).powi(2)).sqrt();
    let vertical_separation = ball.z - player.z;
    horizontal_distance <= HOOD_DRIBBLE_HORIZONTAL_DISTANCE
        && (HOOD_DRIBBLE_MIN_VERTICAL_SEPARATION..=HOOD_DRIBBLE_MAX_VERTICAL_SEPARATION)
            .contains(&vertical_separation)
}

fn add_event_location_flags(values: &mut RowValues, slot_by_id: &HashMap<String, String>) {
    let ball_pos = row_vec(values, "ball", "pos").or_else(|| row_vec(values, "event_ball", "pos"));
    let player_pos = row_string(values, "event_player_1_id")
        .and_then_lookup(slot_by_id)
        .and_then(|slot| row_vec(values, &slot, "pos"));
    let off_wall = ball_pos
        .map(|pos| {
            (SIDE_WALL_X - pos.x.abs()).abs() <= WALL_SHOT_DISTANCE
                || (BACK_WALL_Y - pos.y.abs()).abs() <= WALL_SHOT_DISTANCE
        })
        .unwrap_or(false)
        || player_pos
            .map(|pos| {
                (SIDE_WALL_X - pos.x.abs()).abs() <= WALL_SHOT_DISTANCE
                    || (BACK_WALL_Y - pos.y.abs()).abs() <= WALL_SHOT_DISTANCE
            })
            .unwrap_or(false);
    let off_ceiling = ball_pos
        .map(|pos| CEILING_Z - pos.z <= CEILING_SHOT_DISTANCE)
        .unwrap_or(false)
        || player_pos
            .map(|pos| CEILING_Z - pos.z <= CEILING_SHOT_DISTANCE)
            .unwrap_or(false);
    values.insert("off_wall".to_string(), off_wall.to_string());
    values.insert("off_ceiling".to_string(), off_ceiling.to_string());
}

fn pbp_columns() -> Vec<String> {
    let mut columns = vec![
        "game_id",
        "team_size",
        "blue_team_name",
        "orange_team_name",
        "event_number",
        "event_type",
        "frame_number",
        "observed_frame_number",
        "recorded_frame_number",
        "stint_number",
        "rotation_number",
        "seconds_elapsed",
        "event_team",
        "blue_score",
        "orange_score",
        "controlled",
        "official_shot",
        "official_goal",
        "official_assist",
        "official_save",
        "official_demo",
        "official_shot_count",
        "official_goal_count",
        "official_assist_count",
        "official_save_count",
        "official_demo_count",
        "boost_pickup_amount",
        "boost_pickup_type",
        "reset_origin",
        "event_length",
        "event_duration",
        "off_demo",
        "off_kickoff",
        "off_challenge_win",
        "off_bump",
        "off_controlled_entry",
        "off_controlled_exit",
        "off_retrieval",
        "off_uncontrolled_entry",
        "off_uncontrolled_exit",
        "off_air_dribble",
        "off_ground_dribble",
        "off_flick",
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
        "off_double_tap",
        "off_wall",
        "off_ceiling",
        "previous_event_entry",
        "previous_event_exit",
        "ball_contact_x",
        "ball_contact_y",
        "ball_contact_z",
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
        "event_player_1_id",
        "event_player_1_actor_id",
        "event_player_1_name",
        "event_player_1_team",
        "event_player_2_id",
        "event_player_2_actor_id",
        "event_player_2_name",
        "event_player_2_team",
        "event_player_3_id",
        "event_player_3_actor_id",
        "event_player_3_name",
        "event_player_3_team",
        "linked_shot_observed_frame_number",
        "linked_shot_recorded_frame_number",
        "collision_distance",
        "distance",
        "distance_to_goal",
        "previous_hit_frame_number",
        "next_hit_frame_number",
        "goal_number",
        "kickoff_start_frame_number",
        "kickoff_touch_time",
        "kickoff_type",
        "car_contact_distance",
        "relative_speed",
        "event_player_1_speed",
        "event_player_2_speed",
        "event_player_1_demolished",
        "event_player_2_demolished",
        "event_ball_pos_x",
        "event_ball_pos_y",
        "event_ball_pos_z",
        "ball_pos_x",
        "ball_pos_y",
        "ball_pos_z",
        "ball_vel_x",
        "ball_vel_y",
        "ball_vel_z",
        "ball_ang_vel_x",
        "ball_ang_vel_y",
        "ball_ang_vel_z",
        "previous_event_type",
        "seconds_from_last_event",
        "ball_distance_from_last_event",
        "ball_angle_from_last_event",
        "ball_speed_from_last_event",
        "ball_vel_x_change_from_last_event",
        "ball_vel_y_change_from_last_event",
        "ball_vel_z_change_from_last_event",
        "ball_speed_change_from_last_event",
        "ball_angle_change_from_last_event",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();

    let player_fields = [
        "id",
        "actor_id",
        "network_id",
        "name",
        "platform",
        "is_bot",
        "score",
        "time_in_game",
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
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "ang_vel_x",
        "ang_vel_y",
        "ang_vel_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "boost",
        "boost_active",
        "boost_collect",
        "throttle",
        "steer",
        "handbrake",
        "ball_cam",
        "dodge_active",
        "jump_active",
        "double_jump_active",
        "jumped",
        "flipped",
        "supersonic",
        "distance_to_ball",
        "angle_to_ball",
        "rotation_role",
        "distance_to_own_net",
        "angle_to_own_net",
        "distance_to_opp_net",
        "angle_to_opp_net",
        "distance_from_last_event",
    ];
    let slots = [
        "blue_player_1",
        "blue_player_2",
        "blue_player_3",
        "blue_player_4",
        "orange_player_1",
        "orange_player_2",
        "orange_player_3",
        "orange_player_4",
    ];

    for slot in slots {
        for field in player_fields {
            columns.push(format!("{slot}_{field}"));
        }
        for target in slots {
            if target != slot {
                columns.push(format!("{slot}_distance_to_{target}"));
            }
        }
    }

    columns
}

fn header_prop<'a>(replay: &'a Replay, key: &str) -> Option<&'a HeaderProp> {
    replay
        .properties
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, prop)| prop)
}

fn header_array<'a>(replay: &'a Replay, key: &str) -> Option<&'a Vec<Vec<(String, HeaderProp)>>> {
    header_prop(replay, key).and_then(HeaderProp::as_array)
}

fn header_string(replay: &Replay, key: &str) -> Option<String> {
    header_prop(replay, key).and_then(prop_to_string)
}

fn header_i32(replay: &Replay, key: &str) -> Option<i32> {
    header_prop(replay, key).and_then(prop_to_i32)
}

fn prop<'a>(props: &'a [(String, HeaderProp)], key: &str) -> Option<&'a HeaderProp> {
    props
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, prop)| prop)
}

fn prop_string(props: &[(String, HeaderProp)], key: &str) -> Option<String> {
    prop(props, key).and_then(prop_to_string)
}

fn prop_i32(props: &[(String, HeaderProp)], key: &str) -> Option<i32> {
    prop(props, key).and_then(prop_to_i32)
}

fn prop_bool(props: &[(String, HeaderProp)], key: &str) -> Option<bool> {
    prop(props, key).and_then(HeaderProp::as_bool)
}

fn prop_to_string(prop: &HeaderProp) -> Option<String> {
    match prop {
        HeaderProp::Byte { value, .. } => value.clone(),
        HeaderProp::Name(value) | HeaderProp::Str(value) => Some(value.clone()),
        HeaderProp::Int(value) => Some(value.to_string()),
        HeaderProp::QWord(value) => Some(value.to_string()),
        HeaderProp::Float(value) => Some(value.to_string()),
        HeaderProp::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn prop_to_i32(prop: &HeaderProp) -> Option<i32> {
    match prop {
        HeaderProp::Int(value) => Some(*value),
        HeaderProp::QWord(value) => i32::try_from(*value).ok(),
        HeaderProp::Float(value) => Some(*value as i32),
        _ => None,
    }
}
