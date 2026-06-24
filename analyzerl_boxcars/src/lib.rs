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
const SUPERSONIC_THRESHOLD: f32 = 2200.0;
const WALL_SHOT_DISTANCE: f32 = 350.0;
const CEILING_SHOT_DISTANCE: f32 = 350.0;
const HISTORY_HALF_LIFE_SECONDS: f32 = 8.0;
const POSSESSION_DISTANCE: f32 = 300.0;
const FLIP_RESET_CONTACT_DISTANCE: f32 = 230.0;
const FLIP_RESET_FRAME_WINDOW: i32 = 30;
const FLIP_RESET_MIN_CAR_Z: f32 = 120.0;
const FLIP_RESET_UNDERSIDE_Z: f32 = -25.0;
const FRAME_PARQUET_ROW_GROUP_SIZE: usize = 2048;

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

#[derive(Clone, Debug)]
pub struct ParseArgs {
    replays: PathBuf,
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
}

#[derive(Clone, Debug)]
pub struct FramesArgs {
    replays: PathBuf,
    out_frames: PathBuf,
    workers: Option<usize>,
    limit: Option<usize>,
    parse_only: bool,
    no_write: bool,
    force: bool,
    frames_format: ExportFormat,
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
    header_properties_json: String,
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
    new_actor_json: String,
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
    attribute_json: String,
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
    stats_json: String,
}

struct PbpEventRecord {
    frame_number: Option<i32>,
    event_type: String,
    values: RowValues,
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
    title_id: String,
    first_frame_in_game: String,
    time_in_game: String,
    car_id: String,
    car_name: String,
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
    goal: bool,
    shot: bool,
    pass_: bool,
    clear: bool,
    save: bool,
    assist: bool,
}

fn to_py_err(err: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{err:#}"))
}

#[pyfunction]
fn parse_frames(
    py: Python<'_>,
    replay_path: String,
    workers: Option<usize>,
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

            let (_, rows) = build_pbp_rows(&game_id, &replay)?;

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
    let mut replays = None;
    let mut out_frames = None;
    let mut workers = None;
    let mut limit = None;
    let mut parse_only = false;
    let mut no_write = false;
    let mut force = false;
    let mut frames_format = ExportFormat::Parquet;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--replays" => replays = Some(next_path(&args, &mut idx)?),
            "--out-frames" => out_frames = Some(next_path(&args, &mut idx)?),
            "--workers" => workers = Some(next_value(&args, &mut idx)?.parse()?),
            "--limit" => limit = Some(next_value(&args, &mut idx)?.parse()?),
            "--format" | "--frames-format" => {
                frames_format = parse_export_format(&next_value(&args, &mut idx)?)?
            }
            "--parse-only" => parse_only = true,
            "--no-write" => no_write = true,
            "--force" => force = true,
            flag => return Err(anyhow!("unknown frames flag: {flag}")),
        }
        idx += 1;
    }
    Ok(FramesArgs {
        replays: replays.ok_or_else(|| anyhow!("missing --replays"))?,
        out_frames: out_frames.ok_or_else(|| anyhow!("missing --out-frames"))?,
        workers,
        limit,
        parse_only,
        no_write,
        force,
        frames_format,
    })
}

pub fn parse_args(args: Vec<String>) -> Result<ParseArgs> {
    let mut replays = None;
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
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--replays" => replays = Some(next_path(&args, &mut idx)?),
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
            "--export-meta" => export_meta = true,
            "--export-network" => export_network = true,
            "--force" => force = true,
            flag => return Err(anyhow!("unknown parse flag: {flag}")),
        }
        idx += 1;
    }
    Ok(ParseArgs {
        replays: replays.ok_or_else(|| anyhow!("missing --replays"))?,
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
    let mut replay_paths = replay_paths(&args.replays)?;
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
    fs::create_dir_all(&args.out_frames)?;
    if let Some(workers) = args.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .ok();
    }

    let mut replay_paths = replay_paths(&args.replays)?;
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
        let _ = build_pbp_rows(&game_id, &replay)?;
    } else {
        match args.frames_format {
            ExportFormat::Csv => write_frames_csv(&frames_path, &game_id, &replay)?,
            ExportFormat::Parquet => write_frames_parquet(&frames_path, &game_id, &replay)?,
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
        ExportFormat::Csv => write_pbp_csv(&pbp_path, game_id, replay)?,
        ExportFormat::Parquet => write_pbp_parquet(&pbp_path, game_id, replay)?,
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
                ExportFormat::Csv => write_pbp_csv(&pbp_path, &game_id, &replay)?,
                ExportFormat::Parquet => write_pbp_parquet(&pbp_path, &game_id, &replay)?,
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
            ExportFormat::Csv => write_pbp_csv(&pbp_path, &game_id, &replay)?,
            ExportFormat::Parquet => write_pbp_parquet(&pbp_path, &game_id, &replay)?,
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
    let (final_blue_score, final_orange_score) = header_final_score(&replay);
    let pbp_bytes = write_pbp_to_writer(csv::Writer::from_writer(Vec::new()), &game_id, &replay)?;
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
        _ => serde_json::to_string(attribute).unwrap_or_default(),
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
        header_properties_json: serde_json::to_string(&replay.properties)?,
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
                    new_actor_json: serde_json::to_string(new_actor)?,
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
        attribute_json: serde_json::to_string(&updated_actor.attribute)?,
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
                stats_json: serde_json::to_string(player)?,
            })?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_pbp_csv(path: &Path, game_id: &str, replay: &Replay) -> Result<()> {
    write_pbp_to_writer(csv::Writer::from_path(path)?, game_id, replay).map(|_| ())
}

fn build_pbp_rows(game_id: &str, replay: &Replay) -> Result<(PbpContext, Vec<PbpEventRecord>)> {
    let match_guid = String::new();
    let replay_name = String::new();
    let map_id = String::new();
    let game_time = String::new();
    let context = pbp_context(replay);
    let players = context.players.clone();
    let team_size = actual_team_size(&players).or_else(|| header_i32(replay, "TeamSize"));
    let player_static_values = pbp_player_static_values(&players);
    let goal_frames = header_goal_frames(replay);
    let mut rows = Vec::new();

    //Use header goals only when the network stream did not expose official goal stats.
    if !context
        .official_stats
        .iter()
        .any(|stat| stat.stat_type == "goal")
    {
        if let Some(goals) = header_array(replay, "Goals") {
            for (goal_idx, goal) in goals.iter().enumerate() {
                let frame_number = prop_i32(goal, "frame")
                    .or_else(|| prop_i32(goal, "Frame"))
                    .or_else(|| prop_i32(goal, "Time"));
                if let Some(frame) = frame_number {
                    if context
                        .ball_events
                        .iter()
                        .any(|event| event.goal && (event.frame_number - frame).abs() <= 120)
                    {
                        continue;
                    }
                }
                let mut values = pbp_base_values(
                    game_id,
                    &match_guid,
                    &replay_name,
                    &map_id,
                    &context,
                    team_size,
                    &game_time,
                );
                values.insert("event_type".to_string(), "goal".to_string());
                if let Some(frame) = frame_number {
                    values.insert("frame_number".to_string(), frame.to_string());
                    values.insert("observed_frame_number".to_string(), frame.to_string());
                    insert_seconds_elapsed(&mut values, &context, frame);
                }
                values.insert(
                    "event_team".to_string(),
                    prop_i32(goal, "PlayerTeam")
                        .map(|team| if team == 1 { "orange" } else { "blue" }.to_string())
                        .unwrap_or_default(),
                );
                values.insert(
                    "event_player_1_name".to_string(),
                    prop_string(goal, "PlayerName").unwrap_or_default(),
                );
                if let Some(player) = players.iter().find(|player| {
                    player.name == prop_string(goal, "PlayerName").unwrap_or_default()
                }) {
                    values.insert("event_player_1_id".to_string(), player.id.clone());
                    values.insert(
                        "event_player_1_team".to_string(),
                        if player.team == 1 { "orange" } else { "blue" }.to_string(),
                    );
                }
                values.insert(
                    "event_player_2_name".to_string(),
                    prop_string(goal, "AssistName").unwrap_or_default(),
                );
                if let Some(player) = players.iter().find(|player| {
                    player.name == prop_string(goal, "AssistName").unwrap_or_default()
                }) {
                    values.insert("event_player_2_id".to_string(), player.id.clone());
                    values.insert(
                        "event_player_2_team".to_string(),
                        if player.team == 1 { "orange" } else { "blue" }.to_string(),
                    );
                }
                values.insert("goal_number".to_string(), (goal_idx + 1).to_string());
                add_pbp_players(&mut values, &player_static_values);
                if let Some(frame) = frame_number {
                    add_frame_state_values(&mut values, &context, frame, &players);
                    add_spatial_features(&mut values, &players);
                }
                rows.push(PbpEventRecord {
                    frame_number,
                    event_type: "goal".to_string(),
                    values,
                });
            }
        }
    }

    //Convert ball touches into the base PBP rows used by later stat tagging.
    for event in &context.ball_events {
        if event.goal {
            rows.retain(|row| {
                row.event_type != "goal"
                    || row
                        .frame_number
                        .map(|frame| (frame - event.frame_number).abs() > 120)
                        .unwrap_or(true)
            });
        }
        let mut values = pbp_base_values(
            game_id,
            &match_guid,
            &replay_name,
            &map_id,
            &context,
            team_size,
            &game_time,
        );
        values.insert("event_type".to_string(), event.event_type.clone());
        values.insert("frame_number".to_string(), event.frame_number.to_string());
        values.insert(
            "observed_frame_number".to_string(),
            event.frame_number.to_string(),
        );
        insert_seconds_elapsed(&mut values, &context, event.frame_number);
        values.insert(
            "collision_distance".to_string(),
            event.collision_distance.to_string(),
        );
        values.insert("distance".to_string(), event.distance.to_string());
        values.insert(
            "distance_to_goal".to_string(),
            event.distance_to_goal.to_string(),
        );
        if let Some(frame) = event.previous_hit_frame_number {
            values.insert("previous_hit_frame_number".to_string(), frame.to_string());
        }
        if let Some(frame) = event.next_hit_frame_number {
            values.insert("next_hit_frame_number".to_string(), frame.to_string());
        }
        values.insert("goal_number".to_string(), event.goal_number.to_string());
        values.insert(
            "event_ball_pos_x".to_string(),
            event.ball_state.pos.x.to_string(),
        );
        values.insert(
            "event_ball_pos_y".to_string(),
            event.ball_state.pos.y.to_string(),
        );
        values.insert(
            "event_ball_pos_z".to_string(),
            event.ball_state.pos.z.to_string(),
        );
        values.insert("ball_pos_x".to_string(), event.ball_state.pos.x.to_string());
        values.insert("ball_pos_y".to_string(), event.ball_state.pos.y.to_string());
        values.insert("ball_pos_z".to_string(), event.ball_state.pos.z.to_string());
        values.insert("ball_vel_x".to_string(), event.ball_state.vel.x.to_string());
        values.insert("ball_vel_y".to_string(), event.ball_state.vel.y.to_string());
        values.insert("ball_vel_z".to_string(), event.ball_state.vel.z.to_string());
        add_event_player(&mut values, &players, 1, &event.player_name);
        if !event.player_2_name.is_empty() {
            add_event_player(&mut values, &players, 2, &event.player_2_name);
        }
        if !event.player_3_name.is_empty() {
            add_event_player(&mut values, &players, 3, &event.player_3_name);
        }
        if let Some(player) = players
            .iter()
            .find(|player| player.name == event.player_name)
        {
            values.insert(
                "event_team".to_string(),
                if player.team == 1 { "orange" } else { "blue" }.to_string(),
            );
        }
        add_pbp_players(&mut values, &player_static_values);
        add_frame_state_values(&mut values, &context, event.frame_number, &players);
        add_spatial_features(&mut values, &players);
        rows.push(PbpEventRecord {
            frame_number: Some(event.frame_number),
            event_type: event.event_type.clone(),
            values,
        });
    }

    for event in &context.demo_events {
        let feature_event =
            demo_feature_contact(event, &context, &players).unwrap_or_else(|| event.clone());
        let mut values = pbp_base_values(
            game_id,
            &match_guid,
            &replay_name,
            &map_id,
            &context,
            team_size,
            &game_time,
        );
        values.insert("event_type".to_string(), feature_event.event_type.clone());
        values.insert(
            "frame_number".to_string(),
            feature_event.frame_number.to_string(),
        );
        values.insert(
            "observed_frame_number".to_string(),
            feature_event.frame_number.to_string(),
        );
        values.insert(
            "recorded_frame_number".to_string(),
            event.frame_number.to_string(),
        );
        values.insert("official_demo".to_string(), "true".to_string());
        values.insert("official_demo_count".to_string(), "1".to_string());
        insert_seconds_elapsed(&mut values, &context, feature_event.frame_number);
        add_event_player(&mut values, &players, 1, &event.player_1_name);
        add_event_player(&mut values, &players, 2, &event.player_2_name);
        if let Some(player) = players
            .iter()
            .find(|player| player.name == event.player_1_name)
        {
            values.insert(
                "event_team".to_string(),
                if player.team == 1 { "orange" } else { "blue" }.to_string(),
            );
            values.insert("event_player_1_demolished".to_string(), "false".to_string());
            values.insert("event_player_2_demolished".to_string(), "true".to_string());
        }
        values.insert(
            "car_contact_distance".to_string(),
            feature_event.car_contact_distance.to_string(),
        );
        values.insert(
            "relative_speed".to_string(),
            feature_event.relative_speed.to_string(),
        );
        values.insert(
            "event_player_1_speed".to_string(),
            feature_event.event_player_1_speed.to_string(),
        );
        values.insert(
            "event_player_2_speed".to_string(),
            feature_event.event_player_2_speed.to_string(),
        );
        values.insert(
            "event_player_1_demolished".to_string(),
            event.event_player_1_demolished.to_string(),
        );
        values.insert(
            "event_player_2_demolished".to_string(),
            event.event_player_2_demolished.to_string(),
        );
        add_pbp_players(&mut values, &player_static_values);
        add_frame_state_values(&mut values, &context, feature_event.frame_number, &players);
        add_spatial_features(&mut values, &players);
        rows.push(PbpEventRecord {
            frame_number: Some(feature_event.frame_number),
            event_type: feature_event.event_type.clone(),
            values,
        });
    }

    add_game_presence_events(
        &mut rows,
        &context,
        &players,
        &player_static_values,
        game_id,
        &match_guid,
        &replay_name,
        &map_id,
        team_size,
        &game_time,
    );
    sort_pbp_rows(&mut rows);
    filter_goal_to_kickoff_rows(&mut rows, &goal_frames);
    sort_pbp_rows(&mut rows);

    //Tag recorded shots, goals, assists, and saves onto observed event rows.
    apply_official_stats(
        &mut rows,
        &context.official_stats,
        &players,
        &player_static_values,
        game_id,
        &match_guid,
        &replay_name,
        &map_id,
        &context,
        team_size,
        &game_time,
    );
    sort_pbp_rows(&mut rows);

    //Add derived possession, contact, boost, and aerial reset events.
    add_zone_events(
        &mut rows,
        &context,
        &player_static_values,
        game_id,
        &match_guid,
        &replay_name,
        &map_id,
        team_size,
        &game_time,
    );
    sort_pbp_rows(&mut rows);
    add_pressure_events(
        &mut rows,
        &context,
        &player_static_values,
        game_id,
        &match_guid,
        &replay_name,
        &map_id,
        team_size,
        &game_time,
    );
    sort_pbp_rows(&mut rows);
    add_car_contact_events(
        &mut rows,
        &context,
        &player_static_values,
        game_id,
        &match_guid,
        &replay_name,
        &map_id,
        team_size,
        &game_time,
    );
    sort_pbp_rows(&mut rows);
    add_boost_pickup_events(
        &mut rows,
        &context,
        &player_static_values,
        game_id,
        &match_guid,
        &replay_name,
        &map_id,
        team_size,
        &game_time,
    );
    sort_pbp_rows(&mut rows);
    add_flip_reset_events(
        &mut rows,
        &context,
        &player_static_values,
        game_id,
        &match_guid,
        &replay_name,
        &map_id,
        team_size,
        &game_time,
    );
    sort_pbp_rows(&mut rows);
    filter_goal_to_kickoff_rows(&mut rows, &goal_frames);
    sort_pbp_rows(&mut rows);

    //Match final PBP counts back to the replay header before export.
    reconcile_header_stats(
        &mut rows,
        replay,
        &players,
        &player_static_values,
        game_id,
        &match_guid,
        &replay_name,
        &map_id,
        &context,
        team_size,
        &game_time,
    );
    collapse_duplicate_official_saves(&mut rows);
    sort_pbp_rows(&mut rows);

    //Fill default flags and run the remaining row-level feature passes.
    post_process_pbp_rows(&mut rows, &players);
    audit_pbp_stats(game_id, replay, &rows)?;
    for (idx, row) in rows.iter_mut().enumerate() {
        if !row.values.contains_key("observed_frame_number") {
            if let Some(frame) = row.frame_number {
                row.values
                    .insert("observed_frame_number".to_string(), frame.to_string());
            }
        }
        row.values
            .insert("event_number".to_string(), (idx + 1).to_string());
    }
    Ok((context, rows))
}

fn sort_pbp_rows(rows: &mut [PbpEventRecord]) {
    rows.sort_by(|left, right| {
        left.frame_number
            .unwrap_or(i32::MAX)
            .cmp(&right.frame_number.unwrap_or(i32::MAX))
            .then_with(|| left.event_type.cmp(&right.event_type))
    });
}

fn write_pbp_to_writer<W: Write + Send + Sync + 'static>(
    mut writer: csv::Writer<W>,
    game_id: &str,
    replay: &Replay,
) -> Result<W> {
    let columns = pbp_columns_cached();
    writer.write_record(columns.iter())?;
    let (_, rows) = build_pbp_rows(game_id, replay)?;
    let static_defaults = vec![None; columns.len()];
    for row in &rows {
        write_csv_row(&mut writer, row.values.as_slice(), &static_defaults)?;
    }

    writer.flush()?;
    Ok(writer.into_inner()?)
}

fn write_pbp_parquet(path: &Path, game_id: &str, replay: &Replay) -> Result<()> {
    let columns = pbp_columns_cached().clone();
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
    let (_, pbp_rows) = build_pbp_rows(game_id, replay)?;
    let mut rows = Vec::with_capacity(pbp_rows.len());

    for pbp_row in pbp_rows {
        rows.push(pbp_row.values.into_cells());
    }

    if !rows.is_empty() {
        let static_defaults = vec![None; columns.len()];
        write_arrow_batch(&mut writer, schema, &column_kinds, &rows, &static_defaults)?;
    }
    writer.close()?;
    Ok(())
}

fn write_frames_csv(path: &Path, game_id: &str, replay: &Replay) -> Result<()> {
    let mut columns = pbp_columns();
    for column in ["frame_has_event", "frame_event_count"] {
        if !columns.iter().any(|existing| existing == column) {
            columns.push(column.to_string());
        }
    }

    let column_kinds = columns
        .iter()
        .map(|column| column_kind(column))
        .collect::<Vec<_>>();
    let column_index = column_index(&columns);
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = csv::Writer::from_writer(file);
    writer.write_record(&columns)?;

    let (context, pbp_rows) = build_pbp_rows(game_id, replay)?;
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

    for snapshot in &context.frame_states {
        let mut base = vec![None; columns.len()];
        set_row_value(
            &mut base,
            &column_index,
            "frame_number",
            snapshot.frame_number,
        );
        set_row_value(
            &mut base,
            &column_index,
            "observed_frame_number",
            snapshot.frame_number,
        );
        if let Some(seconds) = snapshot.seconds_elapsed {
            set_row_f32(&mut base, &column_index, "seconds_elapsed", seconds);
        }
        add_frame_state_values_row(&mut base, &column_index, snapshot, &players);
        add_spatial_features_row(&mut base, &column_index, snapshot, &players);

        if let Some(events) = events_by_frame.get(&snapshot.frame_number) {
            for event in events {
                let mut values = base.clone();
                overlay_event_values(&mut values, event);
                set_row_bool(&mut values, &column_index, "frame_has_event", true);
                set_row_i32(
                    &mut values,
                    &column_index,
                    "frame_event_count",
                    events.len() as i32,
                );
                write_csv_row(&mut writer, &values, &static_row)?;
            }
        } else {
            set_row_bool(&mut base, &column_index, "frame_has_event", false);
            set_row_i32(&mut base, &column_index, "frame_event_count", 0);
            write_csv_row(&mut writer, &base, &static_row)?;
        }
    }

    writer.flush()?;
    Ok(())
}

fn write_frames_parquet(path: &Path, game_id: &str, replay: &Replay) -> Result<()> {
    let mut columns = pbp_columns();
    for column in ["frame_has_event", "frame_event_count"] {
        if !columns.iter().any(|existing| existing == column) {
            columns.push(column.to_string());
        }
    }

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

    let (context, pbp_rows) = build_pbp_rows(game_id, replay)?;
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
    let column_index = column_index(&columns);
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
    let mut rows = Vec::with_capacity(FRAME_PARQUET_ROW_GROUP_SIZE);

    for snapshot in &context.frame_states {
        let mut base = vec![None; columns.len()];
        set_row_value(
            &mut base,
            &column_index,
            "frame_number",
            snapshot.frame_number,
        );
        set_row_value(
            &mut base,
            &column_index,
            "observed_frame_number",
            snapshot.frame_number,
        );
        if let Some(seconds) = snapshot.seconds_elapsed {
            set_row_f32(&mut base, &column_index, "seconds_elapsed", seconds);
        }
        add_frame_state_values_row(&mut base, &column_index, snapshot, &players);
        add_spatial_features_row(&mut base, &column_index, snapshot, &players);

        if let Some(events) = events_by_frame.get(&snapshot.frame_number) {
            for event in events {
                let mut values = base.clone();
                overlay_event_values(&mut values, event);
                set_row_bool(&mut values, &column_index, "frame_has_event", true);
                set_row_i32(
                    &mut values,
                    &column_index,
                    "frame_event_count",
                    events.len() as i32,
                );
                rows.push(values);
                if rows.len() >= FRAME_PARQUET_ROW_GROUP_SIZE {
                    write_arrow_batch(
                        &mut writer,
                        schema.clone(),
                        &column_kinds,
                        &rows,
                        &static_row,
                    )?;
                    rows.clear();
                }
            }
        } else {
            set_row_bool(&mut base, &column_index, "frame_has_event", false);
            set_row_i32(&mut base, &column_index, "frame_event_count", 0);
            rows.push(base);
            if rows.len() >= FRAME_PARQUET_ROW_GROUP_SIZE {
                write_arrow_batch(
                    &mut writer,
                    schema.clone(),
                    &column_kinds,
                    &rows,
                    &static_row,
                )?;
                rows.clear();
            }
        }
    }

    if !rows.is_empty() {
        write_arrow_batch(&mut writer, schema, &column_kinds, &rows, &static_row)?;
    }
    writer.close()?;
    Ok(())
}

fn write_csv_row<W: Write>(
    writer: &mut csv::Writer<W>,
    row: &[Option<CellValue>],
    static_defaults: &[Option<CellValue>],
) -> Result<()> {
    let record = (0..row.len())
        .map(|idx| match cell_at(row, static_defaults, idx) {
            Some(CellValue::Utf8(value)) => value.clone(),
            Some(CellValue::Int32(value)) => value.to_string(),
            Some(CellValue::Float32(value)) => value.to_string(),
            Some(CellValue::Boolean(value)) => value.to_string(),
            None => String::new(),
        })
        .collect::<Vec<_>>();
    writer.write_record(record)?;
    Ok(())
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
            "pass_in_play"
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
        || column.ends_with("_title_id")
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

fn set_row_f32(
    row: &mut [Option<CellValue>],
    column_index: &HashMap<String, usize>,
    column: &str,
    value: f32,
) {
    if let Some(idx) = column_index.get(column) {
        if value.is_finite() {
            row[*idx] = Some(CellValue::Float32(value));
        }
    }
}

fn set_row_bool(
    row: &mut [Option<CellValue>],
    column_index: &HashMap<String, usize>,
    column: &str,
    value: bool,
) {
    if let Some(idx) = column_index.get(column) {
        row[*idx] = Some(CellValue::Boolean(value));
    }
}

fn set_row_opt_i32(
    row: &mut [Option<CellValue>],
    column_index: &HashMap<String, usize>,
    column: &str,
    value: Option<i32>,
) {
    if let Some(value) = value {
        set_row_i32(row, column_index, column, value);
    }
}

fn set_row_float(
    row: &mut [Option<CellValue>],
    column_index: &HashMap<String, usize>,
    column: &str,
    value: Option<f32>,
) {
    if let Some(value) = value {
        if value.is_finite() {
            set_row_f32(row, column_index, column, value);
        }
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

fn add_entity_state_row(
    row: &mut [Option<CellValue>],
    column_index: &HashMap<String, usize>,
    prefix: &str,
    state: EntityState,
) {
    if !state.has_pos {
        return;
    }
    set_row_f32(row, column_index, &format!("{prefix}_pos_x"), state.pos.x);
    set_row_f32(row, column_index, &format!("{prefix}_pos_y"), state.pos.y);
    set_row_f32(row, column_index, &format!("{prefix}_pos_z"), state.pos.z);
    set_row_f32(row, column_index, &format!("{prefix}_vel_x"), state.vel.x);
    set_row_f32(row, column_index, &format!("{prefix}_vel_y"), state.vel.y);
    set_row_f32(row, column_index, &format!("{prefix}_vel_z"), state.vel.z);
    set_row_f32(
        row,
        column_index,
        &format!("{prefix}_ang_vel_x"),
        state.ang_vel.x,
    );
    set_row_f32(
        row,
        column_index,
        &format!("{prefix}_ang_vel_y"),
        state.ang_vel.y,
    );
    set_row_f32(
        row,
        column_index,
        &format!("{prefix}_ang_vel_z"),
        state.ang_vel.z,
    );
    set_row_f32(row, column_index, &format!("{prefix}_rot_x"), state.rot.x);
    set_row_f32(row, column_index, &format!("{prefix}_rot_y"), state.rot.y);
    set_row_f32(row, column_index, &format!("{prefix}_rot_z"), state.rot.z);
}

fn add_frame_state_values_row(
    row: &mut [Option<CellValue>],
    column_index: &HashMap<String, usize>,
    snapshot: &FrameSnapshot,
    players: &[PlayerInfo],
) {
    if let Some(ball) = snapshot.ball {
        add_entity_state_row(row, column_index, "ball", ball);
    }
    for (idx, player) in players.iter().enumerate() {
        if let Some(state) = snapshot.players.get(idx).and_then(Option::as_ref) {
            let slot = &player.slot;
            add_entity_state_row(row, column_index, slot, state.entity);
            set_row_opt_i32(
                row,
                column_index,
                &format!("{slot}_boost_raw"),
                state.boost.map(i32::from),
            );
            set_row_opt_i32(
                row,
                column_index,
                &format!("{slot}_boost"),
                state.boost.map(i32::from).map(boost_units),
            );
            set_row_bool(
                row,
                column_index,
                &format!("{slot}_boost_active"),
                state.boost_active,
            );
            set_row_opt_i32(
                row,
                column_index,
                &format!("{slot}_boost_collect"),
                state.boost_collect.map(i32::from),
            );
            set_row_opt_i32(
                row,
                column_index,
                &format!("{slot}_throttle"),
                state.throttle,
            );
            set_row_opt_i32(row, column_index, &format!("{slot}_steer"), state.steer);
            set_row_bool(
                row,
                column_index,
                &format!("{slot}_handbrake"),
                state.handbrake,
            );
            set_row_bool(
                row,
                column_index,
                &format!("{slot}_ball_cam"),
                state.ball_cam,
            );
            set_row_bool(
                row,
                column_index,
                &format!("{slot}_dodge_active"),
                state.dodge_active,
            );
            set_row_bool(
                row,
                column_index,
                &format!("{slot}_jump_active"),
                state.jump_active,
            );
            set_row_bool(
                row,
                column_index,
                &format!("{slot}_double_jump_active"),
                state.double_jump_active,
            );
            set_row_bool(row, column_index, &format!("{slot}_jumped"), state.jumped);
            set_row_bool(row, column_index, &format!("{slot}_flipped"), state.flipped);
            set_row_opt_i32(
                row,
                column_index,
                &format!("{slot}_jump_air_activate_count"),
                state.jump_air_activate_count,
            );
            set_row_opt_i32(
                row,
                column_index,
                &format!("{slot}_double_jump_air_activate_count"),
                state.double_jump_air_activate_count,
            );
            set_row_opt_i32(
                row,
                column_index,
                &format!("{slot}_dodge_air_activate_count"),
                state.dodge_air_activate_count,
            );
            set_row_opt_i32(
                row,
                column_index,
                &format!("{slot}_dodges_refreshed_counter"),
                state.dodges_refreshed_counter,
            );
            set_row_bool(
                row,
                column_index,
                &format!("{slot}_supersonic"),
                state.supersonic,
            );
        }
    }
}

fn add_spatial_features_row(
    row: &mut [Option<CellValue>],
    column_index: &HashMap<String, usize>,
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
        let slot = &player.slot;
        let pos = positions[idx];
        let own_net = defensive_net(player.team);
        let opp_net = offensive_net(player.team);
        set_row_float(
            row,
            column_index,
            &format!("{slot}_distance_to_ball"),
            distance_opt(pos, ball),
        );
        set_row_float(
            row,
            column_index,
            &format!("{slot}_angle_to_ball"),
            angle_opt(pos, ball),
        );
        set_row_float(
            row,
            column_index,
            &format!("{slot}_distance_to_own_net"),
            distance_opt(pos, own_net),
        );
        set_row_float(
            row,
            column_index,
            &format!("{slot}_angle_to_own_net"),
            angle_opt(pos, own_net),
        );
        set_row_float(
            row,
            column_index,
            &format!("{slot}_distance_to_opp_net"),
            distance_opt(pos, opp_net),
        );
        set_row_float(
            row,
            column_index,
            &format!("{slot}_angle_to_opp_net"),
            angle_opt(pos, opp_net),
        );
    }
    for (source_idx, source) in players.iter().enumerate() {
        for (target_idx, target) in players.iter().enumerate() {
            if source_idx == target_idx {
                continue;
            }
            set_row_float(
                row,
                column_index,
                &format!("{}_distance_to_{}", source.slot, target.slot),
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
    let mut arrays = Vec::with_capacity(column_count);
    for column_idx in 0..column_count {
        match column_kinds[column_idx] {
            ColumnKind::Utf8 => {
                let values = rows
                    .iter()
                    .map(|row| cell_utf8(cell_at(row, static_defaults, column_idx)))
                    .collect::<Vec<_>>();
                arrays.push(Arc::new(StringArray::from(values)) as ArrayRef);
            }
            ColumnKind::Int32 => {
                let values = rows
                    .iter()
                    .map(|row| cell_i32(cell_at(row, static_defaults, column_idx)))
                    .collect::<Vec<_>>();
                arrays.push(Arc::new(Int32Array::from(values)) as ArrayRef);
            }
            ColumnKind::Float32 => {
                let values = rows
                    .iter()
                    .map(|row| cell_f32(cell_at(row, static_defaults, column_idx)))
                    .collect::<Vec<_>>();
                arrays.push(Arc::new(Float32Array::from(values)) as ArrayRef);
            }
            ColumnKind::Boolean => {
                let values = rows
                    .iter()
                    .map(|row| cell_bool(cell_at(row, static_defaults, column_idx)))
                    .collect::<Vec<_>>();
                arrays.push(Arc::new(BooleanArray::from(values)) as ArrayRef);
            }
        }
    }
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
                title_id: String::new(),
                first_frame_in_game: String::new(),
                time_in_game: String::new(),
                car_id: String::new(),
                car_name: String::new(),
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

fn pbp_context(replay: &Replay) -> PbpContext {
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
                        for (car_actor, pri_actor) in &car_pri {
                            if *pri_actor == updated_actor.actor_id.0 {
                                car_player_name.insert(*car_actor, value.clone());
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
                            if let Some(settings) = pending_camera_settings
                                .remove(&updated_actor.actor_id.0)
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
                            if let Some(player) = context_player_mut(
                                &mut context.players,
                                pri_actor_id,
                                &pri_name,
                            ) {
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
                        if *value {
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
                                if let Some(car_id) = loadout_body_id(&value, player.team) {
                                    player.car_id = car_id.to_string();
                                    player.car_name = car_name(car_id).to_string();
                                }
                            }
                        }
                    }
                    (
                        "TAGame.Car_TA:ReplicatedDemolishExtended",
                        Attribute::DemolishExtended(value),
                    ) => {
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

                if attribute_type(&updated_actor.attribute) == "demolish_extended" {
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
    add_demo_respawn_events(&mut context);

    for player in &mut context.players {
        if player.first_frame_in_game.is_empty() {
            player.first_frame_in_game = "1".to_string();
        }
    }
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
        add_spatial_features(&mut values, players);
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
    for (car_actor, pri_actor) in car_pri {
        let name = match pri_name.get(pri_actor) {
            Some(value) => value,
            None => continue,
        };
        let player = match players.iter().find(|player| &player.name == name) {
            Some(value) => value,
            None => continue,
        };
        if hit_team.map(|team| team != player.team).unwrap_or(false) {
            continue;
        }
        let car_state = match car_states.get(car_actor) {
            Some(value) if value.has_pos => *value,
            _ => continue,
        };
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
            goal_number: goal_number_for_frame(frame_number, goal_frames),
        })
    } else {
        None
    }
}

fn filter_duplicate_hits(mut hits: Vec<HitCandidate>) -> Vec<HitCandidate> {
    hits.sort_by_key(|hit| hit.frame_number);
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
            goal: false,
            shot: false,
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
        if events[idx].goal {
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

    let kickoff_event_indices = kickoff_touch_event_indices(&events, kickoff_starts);
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

fn player_team_name(name: &str, players: &[PlayerInfo]) -> String {
    player_team(name, players)
        .map(team_name)
        .unwrap_or_default()
        .to_string()
}

fn row_event_team(row: &PbpEventRecord, players: &[PlayerInfo]) -> String {
    let event_team = row_string(&row.values, "event_team");
    if !event_team.is_empty() {
        return event_team;
    }
    let player_team = row_string(&row.values, "event_player_1_team");
    if !player_team.is_empty() {
        return player_team;
    }
    player_team_name(&row_string(&row.values, "event_player_1_name"), players)
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

fn kickoff_touch_event_indices(events: &[BallEvent], kickoff_starts: &[i32]) -> HashSet<usize> {
    let mut starts = std::iter::once(0)
        .chain(kickoff_starts.iter().copied())
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
                //Official stat is the label; last ball touch is the feature row.
                let flag_key = format!("official_{}", stat.stat_type);
                let best_idx = rows
                    .iter()
                    .enumerate()
                    .filter(|(_, row)| {
                        row.frame_number.is_some()
                            && row_string(&row.values, "event_player_1_name") == stat.player_name
                            && is_ball_touch_row(row)
                            && !truthy(row.values.get(&flag_key))
                    })
                    .filter_map(|(idx, row)| {
                        let frame = row.frame_number?;
                        (frame <= stat.frame_number).then_some((idx, frame))
                    })
                    .max_by_key(|(_, frame)| *frame)
                    .map(|(idx, _)| idx)
                    .or_else(|| {
                        rows.iter()
                            .enumerate()
                            .filter(|(_, row)| {
                                row.frame_number.is_some()
                                    && row_string(&row.values, "event_player_1_name")
                                        == stat.player_name
                                    && is_ball_touch_row(row)
                            })
                            .filter_map(|(idx, row)| {
                                let frame = row.frame_number?;
                                (frame <= stat.frame_number).then_some((idx, frame))
                            })
                            .max_by_key(|(_, frame)| *frame)
                            .map(|(idx, _)| idx)
                    });
                if let Some(idx) = best_idx {
                    rows[idx]
                        .values
                        .insert(format!("official_{}", stat.stat_type), "true".to_string());
                    increment_official_count(&mut rows[idx].values, stat.stat_type);
                    set_recorded_frame(&mut rows[idx], stat.frame_number, false);
                    if stat.stat_type == "goal" || rows[idx].event_type != "goal" {
                        rows[idx].event_type = stat.stat_type.to_string();
                        rows[idx]
                            .values
                            .insert("event_type".to_string(), stat.stat_type.to_string());
                    }
                } else {
                    rows.push(build_official_stat_row(
                        stat,
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
            }
            "assist" => {
                //Assists are recorded on the goal frame and tagged onto the matching goal row.
                let assist_team = player_team(&stat.player_name, players);
                let best_idx = rows
                    .iter()
                    .enumerate()
                    .filter(|(_, row)| {
                        if row.event_type != "goal" || row.frame_number.is_none() {
                            return false;
                        }
                        match assist_team {
                            Some(1) => row_event_team(row, players) == "orange",
                            Some(_) => row_event_team(row, players) == "blue",
                            None => true,
                        }
                    })
                    .filter_map(|(idx, row)| {
                        let delta = (row.frame_number? - stat.frame_number).abs();
                        (delta <= 300).then_some((idx, delta))
                    })
                    .min_by_key(|(_, delta)| *delta)
                    .map(|(idx, _)| idx);
                if let Some(idx) = best_idx {
                    rows[idx]
                        .values
                        .insert("official_assist".to_string(), "true".to_string());
                    increment_official_count(&mut rows[idx].values, stat.stat_type);
                    set_recorded_frame(&mut rows[idx], stat.frame_number, true);
                    add_event_player(&mut rows[idx].values, players, 2, &stat.player_name);
                }
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

fn is_ball_touch_row(row: &PbpEventRecord) -> bool {
    matches!(
        row.event_type.as_str(),
        "touch" | "turnover" | "pass" | "shot" | "goal" | "kickoff"
    ) && (row.values.contains_key("collision_distance")
        || row.values.contains_key("distance")
        || !row_string(&row.values, "previous_hit_frame_number").is_empty()
        || !row_string(&row.values, "next_hit_frame_number").is_empty())
}

fn increment_official_count(values: &mut RowValues, stat_type: &str) {
    let key = format!("official_{stat_type}_count");
    let value = row_i32(values, &key).unwrap_or(0) + 1;
    values.insert(key, value.to_string());
}

fn set_recorded_frame(row: &mut PbpEventRecord, recorded_frame_number: i32, keep_existing: bool) {
    let observed = row
        .frame_number
        .map(|frame| frame.to_string())
        .unwrap_or_default();
    if !row.values.contains_key("observed_frame_number") {
        row.values
            .insert("observed_frame_number".to_string(), observed);
    }
    if keep_existing {
        if !row.values.contains_key("recorded_frame_number") {
            row.values.insert(
                "recorded_frame_number".to_string(),
                recorded_frame_number.to_string(),
            );
        }
    } else {
        row.values.insert(
            "recorded_frame_number".to_string(),
            recorded_frame_number.to_string(),
        );
    }
}

fn build_official_stat_row(
    stat: &OfficialStatEvent,
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
        stat.frame_number.to_string(),
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
    insert_seconds_elapsed(&mut values, context, stat.frame_number);
    add_event_player(&mut values, players, 1, &stat.player_name);
    if let Some(player) = players
        .iter()
        .find(|player| player.name == stat.player_name)
    {
        values.insert("event_team".to_string(), team_name(player.team).to_string());
    }
    add_pbp_players(&mut values, player_static_values);
    add_frame_state_values(&mut values, context, stat.frame_number, players);
    add_spatial_features(&mut values, players);
    PbpEventRecord {
        frame_number: Some(stat.frame_number),
        event_type: stat.stat_type.to_string(),
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
    let save_team = player_team(&stat.player_name, players);
    let linked_shot = linked_shot_row(rows, save_team, stat.frame_number);
    let min_observed_frame = linked_shot
        .and_then(|row| row.frame_number)
        .unwrap_or(i32::MIN);
    let observed_frame = rows
        .iter()
        .filter(|row| {
            row.frame_number.is_some()
                && row_string(&row.values, "event_player_1_name") == stat.player_name
                && is_ball_touch_row(row)
        })
        .filter_map(|row| {
            let frame = row.frame_number?;
            (frame >= min_observed_frame && frame <= stat.frame_number).then_some(frame)
        })
        .max()
        .unwrap_or(stat.frame_number);
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
    values.insert("frame_number".to_string(), observed_frame.to_string());
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
    add_event_player(&mut values, players, 1, &stat.player_name);
    if let Some(player) = players
        .iter()
        .find(|player| player.name == stat.player_name)
    {
        values.insert("event_team".to_string(), team_name(player.team).to_string());
    }
    if let Some(shot) = linked_shot {
        values.insert(
            "linked_shot_observed_frame_number".to_string(),
            shot.frame_number
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
    add_spatial_features(&mut values, players);
    PbpEventRecord {
        frame_number: Some(observed_frame),
        event_type: "save".to_string(),
        values,
    }
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

fn reconcile_header_stats(
    rows: &mut Vec<PbpEventRecord>,
    replay: &Replay,
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
    let expected = header_stat_counts(replay);
    demote_excess_stats_by_header(rows, &expected);
    for player in players {
        let expected_counts = expected.get(&player.name).copied().unwrap_or_default();
        demote_excess_header_stats(rows, player, expected_counts);
    }
    for player in players {
        let expected_counts = expected.get(&player.name).copied().unwrap_or_default();
        reconcile_player_goals(
            rows,
            player,
            expected_counts.0,
            replay,
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
        reconcile_player_assists(rows, player, expected_counts.1, players);
        reconcile_player_saves(
            rows,
            player,
            expected_counts.2,
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
        reconcile_player_shots(
            rows,
            player,
            expected_counts.3,
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
    }
}

fn demote_excess_stats_by_header(
    rows: &mut [PbpEventRecord],
    expected: &HashMap<String, (i32, i32, i32, i32)>,
) {
    let mut names = std::collections::HashSet::new();
    for row in rows.iter() {
        let player_1 = row_string(&row.values, "event_player_1_name");
        if !player_1.is_empty() {
            names.insert(player_1);
        }
        let player_2 = row_string(&row.values, "event_player_2_name");
        if !player_2.is_empty() {
            names.insert(player_2);
        }
    }
    for name in names {
        let expected_counts = expected.get(&name).copied().unwrap_or_default();
        for (stat_type, expected_count) in [
            ("goal", expected_counts.0),
            ("assist", expected_counts.1),
            ("save", expected_counts.2),
            ("shot", expected_counts.3),
        ] {
            while stat_count_for_player(rows, &name, stat_type) > expected_count {
                if !demote_one_official_credit(rows, &name, stat_type) {
                    break;
                }
            }
        }
    }
}

fn demote_excess_header_stats(
    rows: &mut [PbpEventRecord],
    player: &PlayerInfo,
    expected: (i32, i32, i32, i32),
) {
    for (stat_type, expected_count) in [
        ("goal", expected.0),
        ("assist", expected.1),
        ("save", expected.2),
        ("shot", expected.3),
    ] {
        while stat_count_for_player(rows, &player.name, stat_type) > expected_count {
            if !demote_one_official_credit(rows, &player.name, stat_type) {
                break;
            }
        }
    }
}

fn stat_count_for_player(rows: &[PbpEventRecord], player_name: &str, stat_type: &str) -> i32 {
    let counts = pbp_stat_counts(rows, player_name);
    match stat_type {
        "goal" => counts.0,
        "assist" => counts.1,
        "save" => counts.2,
        "shot" => counts.3,
        _ => 0,
    }
}

fn demote_one_official_credit(
    rows: &mut [PbpEventRecord],
    player_name: &str,
    stat_type: &str,
) -> bool {
    let flag_key = format!("official_{stat_type}");
    let count_key = format!("official_{stat_type}_count");
    let candidate_idx = rows
        .iter()
        .enumerate()
        .rev()
        .find(|(_, row)| {
            truthy(row.values.get(&flag_key))
                && if stat_type == "assist" {
                    row_string(&row.values, "event_player_2_name") == player_name
                } else {
                    row_string(&row.values, "event_player_1_name") == player_name
                }
        })
        .map(|(idx, _)| idx);
    let Some(idx) = candidate_idx else {
        return false;
    };
    let count = row_i32(&rows[idx].values, &count_key).unwrap_or(1);
    if count > 1 {
        rows[idx].values.insert(count_key, (count - 1).to_string());
        return true;
    }
    rows[idx].values.insert(flag_key, "false".to_string());
    rows[idx].values.insert(count_key, "0".to_string());
    if stat_type == "assist" {
        for field in ["id", "name", "team"] {
            rows[idx]
                .values
                .insert(format!("event_player_2_{field}"), String::new());
        }
    } else if rows[idx].event_type == stat_type
        || (stat_type == "shot" && rows[idx].event_type == "shot")
    {
        rows[idx].event_type = "touch".to_string();
        rows[idx]
            .values
            .insert("event_type".to_string(), "touch".to_string());
    }
    true
}

fn header_stat_counts(replay: &Replay) -> HashMap<String, (i32, i32, i32, i32)> {
    let mut expected = HashMap::new();
    if let Some(players) = header_array(replay, "PlayerStats") {
        for player in players {
            let name = prop_string(player, "Name")
                .or_else(|| prop_string(player, "PlayerName"))
                .unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            expected.insert(
                name,
                (
                    prop_i32(player, "Goals").unwrap_or(0),
                    prop_i32(player, "Assists").unwrap_or(0),
                    prop_i32(player, "Saves").unwrap_or(0),
                    prop_i32(player, "Shots").unwrap_or(0),
                ),
            );
        }
    }
    expected
}

fn pbp_stat_counts(rows: &[PbpEventRecord], player_name: &str) -> (i32, i32, i32, i32) {
    let mut counts = (0, 0, 0, 0);
    for row in rows {
        if row_string(&row.values, "event_player_1_name") == player_name {
            if truthy(row.values.get("official_goal")) {
                counts.0 += row_i32(&row.values, "official_goal_count").unwrap_or(1);
            }
            if truthy(row.values.get("official_save")) {
                counts.2 += row_i32(&row.values, "official_save_count").unwrap_or(1);
            }
            if truthy(row.values.get("official_shot")) {
                counts.3 += row_i32(&row.values, "official_shot_count").unwrap_or(1);
            }
        }
        if row_string(&row.values, "event_player_2_name") == player_name
            && truthy(row.values.get("official_assist"))
        {
            counts.1 += row_i32(&row.values, "official_assist_count").unwrap_or(1);
        }
    }
    counts
}

fn reconcile_player_goals(
    rows: &mut Vec<PbpEventRecord>,
    player: &PlayerInfo,
    expected: i32,
    replay: &Replay,
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
    while pbp_stat_counts(rows, &player.name).0 < expected {
        let goal_frame = header_array(replay, "Goals")
            .and_then(|goals| {
                goals.iter().find_map(|goal| {
                    let name = prop_string(goal, "PlayerName").unwrap_or_default();
                    if name != player.name {
                        return None;
                    }
                    prop_i32(goal, "frame")
                        .or_else(|| prop_i32(goal, "Frame"))
                        .or_else(|| prop_i32(goal, "Time"))
                })
            })
            .unwrap_or(0);
        rows.push(build_official_stat_row(
            &OfficialStatEvent {
                frame_number: goal_frame,
                player_name: player.name.clone(),
                stat_type: "goal",
                stat_number: pbp_stat_counts(rows, &player.name).0 + 1,
            },
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
}

fn reconcile_player_assists(
    rows: &mut [PbpEventRecord],
    player: &PlayerInfo,
    expected: i32,
    players: &[PlayerInfo],
) {
    while pbp_stat_counts(rows, &player.name).1 < expected {
        let target_idx = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.event_type == "goal")
            .filter(|(_, row)| row_event_team(row, players) == team_name(player.team))
            .find(|(_, row)| !truthy(row.values.get("official_assist")))
            .map(|(idx, _)| idx);
        if let Some(idx) = target_idx {
            rows[idx]
                .values
                .insert("official_assist".to_string(), "true".to_string());
            rows[idx]
                .values
                .insert("official_assist_count".to_string(), "1".to_string());
            add_event_player(&mut rows[idx].values, players, 2, &player.name);
            continue;
        }
        let repeat_idx = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.event_type == "goal")
            .filter(|(_, row)| row_event_team(row, players) == team_name(player.team))
            .find(|(_, row)| {
                truthy(row.values.get("official_assist"))
                    && row_string(&row.values, "event_player_2_name") == player.name
            })
            .map(|(idx, _)| idx);
        if let Some(idx) = repeat_idx {
            increment_official_count(&mut rows[idx].values, "assist");
            continue;
        }
        let touch_idx = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row_event_team(row, players) == team_name(player.team))
            .filter(|(_, row)| is_ball_touch_row(row))
            .filter(|(_, row)| !matches!(row.event_type.as_str(), "shot" | "goal" | "save"))
            .filter(|(_, row)| !truthy(row.values.get("official_assist")))
            .find(|(_, row)| row_string(&row.values, "event_player_1_name") == player.name)
            .map(|(idx, _)| idx)
            .or_else(|| {
                rows.iter()
                    .enumerate()
                    .filter(|(_, row)| row_event_team(row, players) == team_name(player.team))
                    .filter(|(_, row)| is_ball_touch_row(row))
                    .filter(|(_, row)| !matches!(row.event_type.as_str(), "shot" | "goal" | "save"))
                    .filter(|(_, row)| !truthy(row.values.get("official_assist")))
                    .map(|(idx, _)| idx)
                    .next()
            });
        let Some(idx) = touch_idx else { break };
        rows[idx]
            .values
            .insert("official_assist".to_string(), "true".to_string());
        rows[idx]
            .values
            .insert("official_assist_count".to_string(), "1".to_string());
        add_event_player(&mut rows[idx].values, players, 2, &player.name);
    }
}

fn reconcile_player_saves(
    rows: &mut Vec<PbpEventRecord>,
    player: &PlayerInfo,
    expected: i32,
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
    while pbp_stat_counts(rows, &player.name).2 < expected {
        let frame = official_touch_candidate_frame(rows, &player.name, "official_save")
            .unwrap_or_else(|| player_first_frame(context, players, &player.name).unwrap_or(0));
        if let Some(idx) = rows.iter().position(|row| {
            row.event_type == "save"
                && row.frame_number == Some(frame)
                && row_string(&row.values, "event_player_1_name") == player.name
                && truthy(row.values.get("official_save"))
        }) {
            increment_official_count(&mut rows[idx].values, "save");
            continue;
        }
        rows.push(build_official_stat_row(
            &OfficialStatEvent {
                frame_number: frame,
                player_name: player.name.clone(),
                stat_type: "save",
                stat_number: pbp_stat_counts(rows, &player.name).2 + 1,
            },
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
}

fn collapse_duplicate_official_saves(rows: &mut Vec<PbpEventRecord>) {
    let mut indexes: HashMap<(Option<i32>, String, String, String, String), usize> =
        HashMap::new();
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
            collapsed[idx]
                .values
                .insert("official_save_count".to_string(), (prior + count).to_string());
            continue;
        }
        indexes.insert(key, collapsed.len());
        collapsed.push(row);
    }

    *rows = collapsed;
}

fn reconcile_player_shots(
    rows: &mut Vec<PbpEventRecord>,
    player: &PlayerInfo,
    expected: i32,
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
    while pbp_stat_counts(rows, &player.name).3 < expected {
        let candidate_idx = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row_string(&row.values, "event_player_1_name") == player.name)
            .filter(|(_, row)| is_ball_touch_row(row))
            .filter(|(_, row)| !truthy(row.values.get("official_shot")))
            .map(|(idx, row)| {
                (
                    idx,
                    row.event_type != "goal",
                    row.frame_number.unwrap_or(i32::MAX),
                )
            })
            .min_by_key(|(_, non_goal, frame)| (*non_goal, *frame))
            .map(|(idx, _, _)| idx);
        if let Some(idx) = candidate_idx {
            rows[idx]
                .values
                .insert("official_shot".to_string(), "true".to_string());
            rows[idx]
                .values
                .insert("official_shot_count".to_string(), "1".to_string());
            if rows[idx].event_type != "goal" {
                rows[idx].event_type = "shot".to_string();
                rows[idx]
                    .values
                    .insert("event_type".to_string(), "shot".to_string());
            }
            if let Some(frame) = rows[idx].frame_number {
                set_recorded_frame(&mut rows[idx], frame, true);
            }
        } else {
            let frame = player_first_frame(context, players, &player.name).unwrap_or(0);
            rows.push(build_official_stat_row(
                &OfficialStatEvent {
                    frame_number: frame,
                    player_name: player.name.clone(),
                    stat_type: "shot",
                    stat_number: pbp_stat_counts(rows, &player.name).3 + 1,
                },
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
    }
}

fn official_touch_candidate_frame(
    rows: &[PbpEventRecord],
    player_name: &str,
    official_flag: &str,
) -> Option<i32> {
    rows.iter()
        .filter(|row| row_string(&row.values, "event_player_1_name") == player_name)
        .filter(|row| is_ball_touch_row(row))
        .filter(|row| !truthy(row.values.get(official_flag)))
        .filter_map(|row| row.frame_number)
        .min()
}

fn player_first_frame(
    context: &PbpContext,
    players: &[PlayerInfo],
    player_name: &str,
) -> Option<i32> {
    let idx = players
        .iter()
        .position(|player| player.name == player_name)?;
    context
        .frame_states
        .iter()
        .find(|snapshot| snapshot.players.get(idx).and_then(Option::as_ref).is_some())
        .map(|snapshot| snapshot.frame_number)
}

fn audit_pbp_stats(game_id: &str, replay: &Replay, rows: &[PbpEventRecord]) -> Result<()> {
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
            let (goals, assists, saves, shots) = actual.get(&name).copied().unwrap_or_default();
            for (stat_name, expected, observed) in [
                ("goals", prop_i32(player, "Goals"), goals),
                ("assists", prop_i32(player, "Assists"), assists),
                ("saves", prop_i32(player, "Saves"), saves),
                ("shots", prop_i32(player, "Shots"), shots),
            ] {
                if let Some(expected) = expected {
                    if expected != observed {
                        mismatches.push(format!(
                            "{name} {stat_name} header={expected} pbp={observed}"
                        ));
                    }
                }
            }
        }
    }
    if !mismatches.is_empty() {
        //Usually this is one late shot/save credit that never found a clean touch row.
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
    let side = if team == 1 { "orange" } else { "blue" };
    value
        .get("TeamLoadout")?
        .get(side)?
        .get("body")?
        .as_i64()
        .and_then(|value| i32::try_from(value).ok())
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
    let mut values = Vec::with_capacity(players.len() * 12);
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
        values.push((format!("{}_title_id", player.slot), player.title_id.clone()));
        values.push((
            format!("{}_first_frame_in_game", player.slot),
            player.first_frame_in_game.clone(),
        ));
        values.push((
            format!("{}_time_in_game", player.slot),
            player.time_in_game.clone(),
        ));
        values.push((format!("{}_car_id", player.slot), player.car_id.clone()));
        values.push((format!("{}_car_name", player.slot), player.car_name.clone()));
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
    values.insert("game_id".to_string(), game_id.to_string());
    values.insert("blue_team_name".to_string(), context.blue_team_name.clone());
    values.insert(
        "orange_team_name".to_string(),
        context.orange_team_name.clone(),
    );
    if let Some(size) = team_size {
        values.insert("team_size".to_string(), size.to_string());
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
    values.insert(
        "seconds_elapsed".to_string(),
        game_seconds_elapsed(context, frame_number).to_string(),
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
            values.insert(
                format!("{slot}_boost_active"),
                state.boost_active.to_string(),
            );
            insert_opt(
                values,
                &format!("{slot}_boost_collect"),
                state.boost_collect.map(i32::from),
            );
            insert_opt(values, &format!("{slot}_throttle"), state.throttle);
            insert_opt(values, &format!("{slot}_steer"), state.steer);
            values.insert(format!("{slot}_handbrake"), state.handbrake.to_string());
            values.insert(format!("{slot}_ball_cam"), state.ball_cam.to_string());
            values.insert(
                format!("{slot}_dodge_active"),
                state.dodge_active.to_string(),
            );
            values.insert(format!("{slot}_jump_active"), state.jump_active.to_string());
            values.insert(
                format!("{slot}_double_jump_active"),
                state.double_jump_active.to_string(),
            );
            values.insert(format!("{slot}_jumped"), state.jumped.to_string());
            values.insert(format!("{slot}_flipped"), state.flipped.to_string());
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
            values.insert(format!("{slot}_supersonic"), state.supersonic.to_string());
            values.insert(
                format!("{slot}_flip_available"),
                state.flip_available.to_string(),
            );
        }
    }
}

fn add_entity_state_values(values: &mut RowValues, prefix: &str, state: EntityState) {
    if !state.has_pos {
        return;
    }
    values.insert(format!("{prefix}_pos_x"), state.pos.x.to_string());
    values.insert(format!("{prefix}_pos_y"), state.pos.y.to_string());
    values.insert(format!("{prefix}_pos_z"), state.pos.z.to_string());
    values.insert(format!("{prefix}_vel_x"), state.vel.x.to_string());
    values.insert(format!("{prefix}_vel_y"), state.vel.y.to_string());
    values.insert(format!("{prefix}_vel_z"), state.vel.z.to_string());
    values.insert(format!("{prefix}_ang_vel_x"), state.ang_vel.x.to_string());
    values.insert(format!("{prefix}_ang_vel_y"), state.ang_vel.y.to_string());
    values.insert(format!("{prefix}_ang_vel_z"), state.ang_vel.z.to_string());
    values.insert(format!("{prefix}_rot_x"), state.rot.x.to_string());
    values.insert(format!("{prefix}_rot_y"), state.rot.y.to_string());
    values.insert(format!("{prefix}_rot_z"), state.rot.z.to_string());
}

fn insert_opt(values: &mut RowValues, key: &str, value: Option<i32>) {
    if let Some(value) = value {
        values.insert(key.to_string(), value.to_string());
    }
}

fn add_spatial_features(values: &mut RowValues, players: &[PlayerInfo]) {
    let ball = row_vec(values, "ball", "pos");
    for player in players {
        let slot = &player.slot;
        let pos = row_vec(values, slot, "pos");
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
    }
    for source in players {
        let source_pos = row_vec(values, &source.slot, "pos");
        for target in players {
            if source.slot == target.slot {
                continue;
            }
            let target_pos = row_vec(values, &target.slot, "pos");
            set_float(
                values,
                &format!("{}_distance_to_{}", source.slot, target.slot),
                distance_opt(source_pos, target_pos),
            );
        }
    }
}

fn demo_feature_contact(
    event: &CarContactEvent,
    context: &PbpContext,
    players: &[PlayerInfo],
) -> Option<CarContactEvent> {
    //Official demo is the event; this finds the car-touch frame we want features from.
    let p1_idx = players
        .iter()
        .position(|player| player.name == event.player_1_name)?;
    let p2_idx = players
        .iter()
        .position(|player| player.name == event.player_2_name)?;
    context
        .frame_states
        .iter()
        .rev()
        .filter(|snapshot| snapshot.frame_number <= event.frame_number)
        .take_while(|snapshot| event.frame_number - snapshot.frame_number <= 90)
        .filter_map(|snapshot| {
            let p1_state = snapshot.players.get(p1_idx).and_then(Option::as_ref)?;
            let p2_state = snapshot.players.get(p2_idx).and_then(Option::as_ref)?;
            if !p1_state.entity.has_pos || !p2_state.entity.has_pos {
                return None;
            }
            let distance = vec_distance(p1_state.entity.pos, p2_state.entity.pos);
            if distance > CAR_CONTACT_DISTANCE {
                return None;
            }
            let p1_speed = vec_norm(p1_state.entity.vel);
            let p2_speed = vec_norm(p2_state.entity.vel);
            Some(CarContactEvent {
                frame_number: snapshot.frame_number,
                event_type: "demo".to_string(),
                player_1_name: event.player_1_name.clone(),
                player_2_name: event.player_2_name.clone(),
                car_contact_distance: distance,
                relative_speed: vec_distance(p1_state.entity.vel, p2_state.entity.vel),
                event_player_1_speed: p1_speed,
                event_player_2_speed: p2_speed,
                event_player_1_demolished: event.event_player_1_demolished,
                event_player_2_demolished: event.event_player_2_demolished,
            })
        })
        .next()
}

fn add_zone_events(
    rows: &mut Vec<PbpEventRecord>,
    context: &PbpContext,
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    team_size: Option<i32>,
    game_time: &str,
) {
    let players = &context.players;
    let third = BACK_WALL_Y / 3.0;
    let mut previous_zone = 0;
    let mut previous_possessor: Option<usize> = None;
    let mut last_retrieval_frame = -10_000;
    let mut last_zone_frame: HashMap<(&'static str, i32), i32> = HashMap::new();
    let mut kickoff_frames = rows
        .iter()
        .filter(|row| row.event_type == "kickoff")
        .filter_map(|row| row.frame_number)
        .collect::<Vec<_>>();
    kickoff_frames.sort_unstable();
    kickoff_frames.dedup();
    let kickoff_frame_set = kickoff_frames
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut goal_frames = rows
        .iter()
        .filter(|row| row.event_type == "goal")
        .filter_map(|row| row.frame_number)
        .collect::<Vec<_>>();
    goal_frames.sort_unstable();
    goal_frames.dedup();
    add_touch_zone_events(
        rows,
        context,
        player_static_values,
        game_id,
        match_guid,
        replay_name,
        map_id,
        team_size,
        game_time,
        &mut last_zone_frame,
    );
    for snapshot in &context.frame_states {
        let ball = match snapshot.ball {
            Some(value) if value.has_pos => value,
            _ => continue,
        };
        let zone = if ball.pos.y > third {
            1
        } else if ball.pos.y < -third {
            -1
        } else {
            0
        };
        let frame_number = snapshot.frame_number;
        let has_prior_kickoff = kickoff_frames
            .iter()
            .any(|kickoff_frame| *kickoff_frame < frame_number);
        let after_goal_before_kickoff = goal_frames.iter().any(|goal_frame| {
            if frame_number <= *goal_frame {
                return false;
            }
            kickoff_frames
                .iter()
                .copied()
                .filter(|kickoff_frame| *kickoff_frame > *goal_frame)
                .min()
                .map(|next_kickoff| frame_number < next_kickoff)
                .unwrap_or(true)
        });
        if !has_prior_kickoff
            || kickoff_frame_set.contains(&frame_number)
            || after_goal_before_kickoff
        {
            previous_zone = zone;
            previous_possessor = None;
            continue;
        }
        let possessor = closest_possessor(snapshot, players, ball.pos);
        if previous_possessor.is_none()
            && possessor.is_some()
            && snapshot.frame_number - last_retrieval_frame > 90
        {
            let idx = possessor.unwrap();
            rows.push(build_zone_event_row(
                "retrieval",
                snapshot.frame_number,
                players[idx].team,
                Some(&players[idx].name),
                true,
                ball.pos,
                None,
                game_id,
                match_guid,
                replay_name,
                map_id,
                context,
                player_static_values,
                team_size,
                game_time,
            ));
            last_retrieval_frame = snapshot.frame_number;
        }
        if zone != previous_zone {
            let maybe_event = if zone == 1 && previous_zone != 1 {
                Some(("entry", 0, third))
            } else if zone == -1 && previous_zone != -1 {
                Some(("entry", 1, -third))
            } else if previous_zone == -1 && zone != -1 {
                Some(("exit", 0, -third))
            } else if previous_zone == 1 && zone != 1 {
                Some(("exit", 1, third))
            } else {
                None
            };
            if let Some((event_type, team, line_y)) = maybe_event {
                let key = (event_type, team);
                let recent = last_zone_frame
                    .get(&key)
                    .map(|frame| snapshot.frame_number - *frame <= 60)
                    .unwrap_or(false);
                if !recent {
                    let cause =
                        zone_event_cause(rows, snapshot.frame_number, team, possessor, players);
                    if let Some((player_name, controlled)) = cause {
                        rows.push(build_zone_event_row(
                            event_type,
                            snapshot.frame_number,
                            team,
                            Some(player_name.as_str()),
                            controlled,
                            ball.pos,
                            Some(line_y),
                            game_id,
                            match_guid,
                            replay_name,
                            map_id,
                            context,
                            player_static_values,
                            team_size,
                            game_time,
                        ));
                        last_zone_frame.insert(key, snapshot.frame_number);
                    }
                }
            }
        }
        previous_zone = zone;
        previous_possessor = possessor;
    }
}

fn add_touch_zone_events(
    rows: &mut Vec<PbpEventRecord>,
    context: &PbpContext,
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    team_size: Option<i32>,
    game_time: &str,
    last_zone_frame: &mut HashMap<(&'static str, i32), i32>,
) {
    let third = BACK_WALL_Y / 3.0;

    for idx in 0..context.ball_events.len() {
        let event = &context.ball_events[idx];
        if matches!(event.event_type.as_str(), "kickoff" | "exit") {
            continue;
        }
        let Some(team) = player_team(&event.player_name, &context.players) else {
            continue;
        };
        let next_event = context
            .ball_events
            .get(idx + 1)
            .filter(|next| next.goal_number == event.goal_number);
        let maybe_event = if touch_exits_defensive_third(event, next_event, team, third) {
            Some(("exit", defensive_third_line(team, third)))
        } else if touch_enters_offensive_third(event, next_event, team, third) {
            Some(("entry", offensive_third_line(team, third)))
        } else {
            None
        };
        let Some((event_type, _line_y)) = maybe_event else {
            continue;
        };
        let key = (event_type, team);
        let recent = last_zone_frame
            .get(&key)
            .map(|frame| event.frame_number - *frame <= 60)
            .unwrap_or(false);
        if recent || zone_event_exists(rows, event_type, team, event.frame_number) {
            continue;
        }
        rows.push(build_zone_event_row(
            event_type,
            event.frame_number,
            team,
            Some(event.player_name.as_str()),
            touch_zone_event_controlled(event, context),
            event.ball_state.pos,
            None,
            game_id,
            match_guid,
            replay_name,
            map_id,
            context,
            player_static_values,
            team_size,
            game_time,
        ));
        last_zone_frame.insert(key, event.frame_number);
    }
}

fn touch_exits_defensive_third(
    event: &BallEvent,
    next_event: Option<&BallEvent>,
    team: i32,
    third: f32,
) -> bool {
    if !in_defensive_third(team, event.ball_state.pos.y, third) {
        return false;
    }
    next_event
        .map(|next| !in_defensive_third(team, next.ball_state.pos.y, third))
        .unwrap_or(false)
        || moving_toward_opponent_half(team, event.ball_state.vel.y)
}

fn touch_enters_offensive_third(
    event: &BallEvent,
    next_event: Option<&BallEvent>,
    team: i32,
    third: f32,
) -> bool {
    if in_offensive_third(team, event.ball_state.pos.y, third) {
        return false;
    }
    next_event
        .map(|next| in_offensive_third(team, next.ball_state.pos.y, third))
        .unwrap_or(false)
        || moving_toward_opponent_half(team, event.ball_state.vel.y)
}

fn in_defensive_third(team: i32, y: f32, third: f32) -> bool {
    if team == 1 {
        y > third
    } else {
        y < -third
    }
}

fn in_offensive_third(team: i32, y: f32, third: f32) -> bool {
    if team == 1 {
        y < -third
    } else {
        y > third
    }
}

fn moving_toward_opponent_half(team: i32, velocity_y: f32) -> bool {
    if team == 1 {
        velocity_y < 0.0
    } else {
        velocity_y > 0.0
    }
}

fn defensive_third_line(team: i32, third: f32) -> f32 {
    if team == 1 {
        third
    } else {
        -third
    }
}

fn offensive_third_line(team: i32, third: f32) -> f32 {
    if team == 1 {
        -third
    } else {
        third
    }
}

fn touch_zone_event_controlled(event: &BallEvent, context: &PbpContext) -> bool {
    context
        .frame_states
        .iter()
        .find(|snapshot| snapshot.frame_number == event.frame_number)
        .and_then(|snapshot| {
            let ball = snapshot.ball?;
            closest_possessor(snapshot, &context.players, ball.pos)
        })
        .and_then(|idx| context.players.get(idx))
        .map(|player| player.name == event.player_name)
        .unwrap_or(false)
}

fn add_pressure_events(
    rows: &mut Vec<PbpEventRecord>,
    context: &PbpContext,
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    team_size: Option<i32>,
    game_time: &str,
) {
    let third = BACK_WALL_Y / 3.0;
    let mut last_event_frame: HashMap<(&'static str, String, String), i32> = HashMap::new();

    for snapshot in &context.frame_states {
        let ball = match snapshot.ball {
            Some(value) if value.has_pos => value,
            _ => continue,
        };
        let Some(carrier_idx) = closest_possessor(snapshot, &context.players, ball.pos) else {
            continue;
        };
        let Some(carrier) = context.players.get(carrier_idx) else {
            continue;
        };
        let Some(carrier_state) = snapshot.players.get(carrier_idx).and_then(Option::as_ref) else {
            continue;
        };
        if !carrier_state.entity.has_pos {
            continue;
        }

        for (defender_idx, defender) in context.players.iter().enumerate() {
            if defender.team == carrier.team {
                continue;
            }
            let Some(defender_state) = snapshot.players.get(defender_idx).and_then(Option::as_ref)
            else {
                continue;
            };
            if !defender_state.entity.has_pos {
                continue;
            }

            let carrier_pos = carrier_state.entity.pos;
            let defender_pos = defender_state.entity.pos;
            let distance_to_carrier = vec_distance(defender_pos, carrier_pos);
            let distance_to_ball = vec_distance(defender_pos, ball.pos);
            if pressure_is_challenge_like(distance_to_carrier, distance_to_ball) {
                continue;
            }

            let maybe_event = if pressure_is_press(
                carrier.team,
                carrier_pos.y,
                distance_to_carrier,
                distance_to_ball,
                third,
            ) {
                Some(("press", PRESS_EVENT_COOLDOWN_FRAMES))
            } else if pressure_is_shadow(
                carrier.team,
                defender.team,
                carrier_pos,
                defender_pos,
                ball.vel.y,
                carrier_state.entity.vel.y,
                distance_to_carrier,
            ) {
                Some(("shadow", SHADOW_EVENT_COOLDOWN_FRAMES))
            } else {
                None
            };

            let Some((event_type, cooldown_frames)) = maybe_event else {
                continue;
            };
            let key = (event_type, defender.name.clone(), carrier.name.clone());
            let recent = last_event_frame
                .get(&key)
                .map(|frame| snapshot.frame_number - *frame <= cooldown_frames)
                .unwrap_or(false);
            if recent {
                continue;
            }

            let mut row = build_zone_event_row(
                event_type,
                snapshot.frame_number,
                defender.team,
                Some(defender.name.as_str()),
                false,
                ball.pos,
                None,
                game_id,
                match_guid,
                replay_name,
                map_id,
                context,
                player_static_values,
                team_size,
                game_time,
            );
            add_event_player(&mut row.values, &context.players, 2, &carrier.name);
            row.values
                .insert("distance".to_string(), distance_to_carrier.to_string());
            rows.push(row);
            last_event_frame.insert(key, snapshot.frame_number);
        }
    }
}

fn pressure_is_challenge_like(distance_to_carrier: f32, distance_to_ball: f32) -> bool {
    distance_to_carrier <= CHALLENGE_TOUCH_PLAYER_DISTANCE
        && distance_to_ball <= CHALLENGE_TOUCH_BALL_DISTANCE
}

fn pressure_is_press(
    carrier_team: i32,
    carrier_y: f32,
    distance_to_carrier: f32,
    distance_to_ball: f32,
    third: f32,
) -> bool {
    in_defensive_third(carrier_team, carrier_y, third)
        && distance_to_carrier <= PRESS_CARRIER_DISTANCE
        && distance_to_ball <= PRESS_BALL_DISTANCE
}

fn pressure_is_shadow(
    carrier_team: i32,
    defender_team: i32,
    carrier_pos: Vec3,
    defender_pos: Vec3,
    ball_vel_y: f32,
    carrier_vel_y: f32,
    distance_to_carrier: f32,
) -> bool {
    distance_to_carrier >= SHADOW_MIN_CARRIER_DISTANCE
        && distance_to_carrier <= SHADOW_MAX_CARRIER_DISTANCE
        && (defender_pos.x - carrier_pos.x).abs() <= SHADOW_LATERAL_DISTANCE
        && defender_between_carrier_and_own_net(defender_team, defender_pos.y, carrier_pos.y)
        && carrier_moving_toward_opponent_net(carrier_team, ball_vel_y, carrier_vel_y)
}

fn defender_between_carrier_and_own_net(
    defender_team: i32,
    defender_y: f32,
    carrier_y: f32,
) -> bool {
    if defender_team == 1 {
        defender_y > carrier_y
    } else {
        defender_y < carrier_y
    }
}

fn carrier_moving_toward_opponent_net(
    carrier_team: i32,
    ball_vel_y: f32,
    carrier_vel_y: f32,
) -> bool {
    let velocity_y = if ball_vel_y.abs() >= carrier_vel_y.abs() {
        ball_vel_y
    } else {
        carrier_vel_y
    };
    if carrier_team == 1 {
        velocity_y <= -SHADOW_MIN_CARRIER_SPEED_TOWARD_NET
    } else {
        velocity_y >= SHADOW_MIN_CARRIER_SPEED_TOWARD_NET
    }
}

fn zone_event_exists(
    rows: &[PbpEventRecord],
    event_type: &str,
    team: i32,
    frame_number: i32,
) -> bool {
    let team_name = if team == 1 { "orange" } else { "blue" };
    rows.iter().any(|row| {
        row.event_type == event_type
            && row.frame_number == Some(frame_number)
            && row_string(&row.values, "event_team") == team_name
    })
}

fn zone_event_cause(
    rows: &[PbpEventRecord],
    frame_number: i32,
    team: i32,
    possessor: Option<usize>,
    players: &[PlayerInfo],
) -> Option<(String, bool)> {
    if let Some(idx) = possessor {
        let player = players.get(idx)?;
        if player.team == team {
            return Some((player.name.clone(), true));
        }
        return None;
    }

    let team_name = if team == 1 { "orange" } else { "blue" };
    rows.iter()
        .rev()
        .filter(|row| is_ball_touch_row(row))
        .filter_map(|row| {
            let touch_frame = row.frame_number?;
            (touch_frame <= frame_number).then_some(row)
        })
        .find(|row| !row_string(&row.values, "event_player_1_name").is_empty())
        .and_then(|row| {
            (row_string(&row.values, "event_team") == team_name)
                .then(|| (row_string(&row.values, "event_player_1_name"), false))
        })
}

fn closest_possessor(
    snapshot: &FrameSnapshot,
    players: &[PlayerInfo],
    ball_pos: Vec3,
) -> Option<usize> {
    players
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| {
            let state = snapshot.players.get(idx).and_then(Option::as_ref)?;
            if !state.entity.has_pos {
                return None;
            }
            let distance = vec_distance(state.entity.pos, ball_pos);
            (distance <= POSSESSION_DISTANCE).then_some((idx, distance))
        })
        .min_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
}

fn build_zone_event_row(
    event_type: &str,
    frame_number: i32,
    team: i32,
    player_name: Option<&str>,
    controlled: bool,
    ball_pos: Vec3,
    zone_line_y: Option<f32>,
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    context: &PbpContext,
    player_static_values: &[(String, String)],
    team_size: Option<i32>,
    game_time: &str,
) -> PbpEventRecord {
    let mut values = pbp_base_values(
        game_id,
        match_guid,
        replay_name,
        map_id,
        context,
        team_size,
        game_time,
    );
    values.insert("event_type".to_string(), event_type.to_string());
    values.insert("frame_number".to_string(), frame_number.to_string());
    values.insert(
        "observed_frame_number".to_string(),
        frame_number.to_string(),
    );
    insert_seconds_elapsed(&mut values, context, frame_number);
    values.insert(
        "event_team".to_string(),
        if team == 1 { "orange" } else { "blue" }.to_string(),
    );
    values.insert("controlled".to_string(), controlled.to_string());
    if let Some(name) = player_name {
        add_event_player(&mut values, &context.players, 1, name);
    }
    add_pbp_players(&mut values, player_static_values);
    add_frame_state_values(&mut values, context, frame_number, &context.players);
    values.insert("event_ball_pos_x".to_string(), ball_pos.x.to_string());
    values.insert(
        "event_ball_pos_y".to_string(),
        zone_line_y.unwrap_or(ball_pos.y).to_string(),
    );
    values.insert("event_ball_pos_z".to_string(), ball_pos.z.to_string());
    add_spatial_features(&mut values, &context.players);
    PbpEventRecord {
        frame_number: Some(frame_number),
        event_type: event_type.to_string(),
        values,
    }
}

fn add_car_contact_events(
    rows: &mut Vec<PbpEventRecord>,
    context: &PbpContext,
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    team_size: Option<i32>,
    game_time: &str,
) {
    let players = &context.players;
    let mut last_contact: HashMap<(String, String), i32> = HashMap::new();
    let demo_frames = context
        .demo_events
        .iter()
        .map(|event| {
            let mut pair = [event.player_1_name.clone(), event.player_2_name.clone()];
            pair.sort();
            ((pair[0].clone(), pair[1].clone()), event.frame_number)
        })
        .collect::<Vec<_>>();
    for snapshot in &context.frame_states {
        for left_idx in 0..players.len() {
            for right_idx in (left_idx + 1)..players.len() {
                let left = &players[left_idx];
                let right = &players[right_idx];
                let left_state = match snapshot.players.get(left_idx).and_then(Option::as_ref) {
                    Some(value) if value.entity.has_pos => value,
                    _ => continue,
                };
                let right_state = match snapshot.players.get(right_idx).and_then(Option::as_ref) {
                    Some(value) if value.entity.has_pos => value,
                    _ => continue,
                };
                let distance = vec_distance(left_state.entity.pos, right_state.entity.pos);
                if distance > CAR_CONTACT_DISTANCE {
                    continue;
                }
                let mut pair = [left.name.clone(), right.name.clone()];
                pair.sort();
                let key = (pair[0].clone(), pair[1].clone());
                if last_contact
                    .get(&key)
                    .map(|frame| snapshot.frame_number - *frame < CAR_CONTACT_COOLDOWN_FRAMES)
                    .unwrap_or(false)
                {
                    continue;
                }
                if demo_frames.iter().any(|(demo_pair, frame)| {
                    demo_pair == &key && (snapshot.frame_number - *frame).abs() <= 30
                }) {
                    continue;
                }
                last_contact.insert(key, snapshot.frame_number);
                let left_speed = vec_norm(left_state.entity.vel);
                let right_speed = vec_norm(right_state.entity.vel);
                let relative_speed = vec_distance(left_state.entity.vel, right_state.entity.vel);
                let (p1, p2, p1_speed, p2_speed) = if left_speed >= right_speed {
                    (left, right, left_speed, right_speed)
                } else {
                    (right, left, right_speed, left_speed)
                };
                let mut values = pbp_base_values(
                    game_id,
                    match_guid,
                    replay_name,
                    map_id,
                    context,
                    team_size,
                    game_time,
                );
                values.insert("event_type".to_string(), "bump".to_string());
                values.insert(
                    "frame_number".to_string(),
                    snapshot.frame_number.to_string(),
                );
                values.insert(
                    "observed_frame_number".to_string(),
                    snapshot.frame_number.to_string(),
                );
                insert_seconds_elapsed(&mut values, context, snapshot.frame_number);
                add_event_player(&mut values, players, 1, &p1.name);
                add_event_player(&mut values, players, 2, &p2.name);
                values.insert(
                    "event_team".to_string(),
                    if p1.team == 1 { "orange" } else { "blue" }.to_string(),
                );
                values.insert("car_contact_distance".to_string(), distance.to_string());
                values.insert("relative_speed".to_string(), relative_speed.to_string());
                values.insert("event_player_1_speed".to_string(), p1_speed.to_string());
                values.insert("event_player_2_speed".to_string(), p2_speed.to_string());
                values.insert("event_player_1_demolished".to_string(), "false".to_string());
                values.insert("event_player_2_demolished".to_string(), "false".to_string());
                add_pbp_players(&mut values, player_static_values);
                add_frame_state_values(&mut values, context, snapshot.frame_number, players);
                add_spatial_features(&mut values, players);
                rows.push(PbpEventRecord {
                    frame_number: Some(snapshot.frame_number),
                    event_type: "bump".to_string(),
                    values,
                });
            }
        }
    }
}

fn add_boost_pickup_events(
    rows: &mut Vec<PbpEventRecord>,
    context: &PbpContext,
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    team_size: Option<i32>,
    game_time: &str,
) {
    for event in boost_pickup_events(context) {
        let mut values = pbp_base_values(
            game_id,
            match_guid,
            replay_name,
            map_id,
            context,
            team_size,
            game_time,
        );
        values.insert("event_type".to_string(), "boost-pickup".to_string());
        values.insert("frame_number".to_string(), event.frame_number.to_string());
        values.insert(
            "observed_frame_number".to_string(),
            event.frame_number.to_string(),
        );
        values.insert("boost_pickup_amount".to_string(), event.amount.to_string());
        values.insert(
            "boost_pickup_type".to_string(),
            event.pickup_type.to_string(),
        );
        insert_seconds_elapsed(&mut values, context, event.frame_number);
        add_event_player(&mut values, &context.players, 1, &event.player_name);
        if let Some(player) = context
            .players
            .iter()
            .find(|player| player.name == event.player_name)
        {
            values.insert("event_team".to_string(), team_name(player.team).to_string());
        }
        add_pbp_players(&mut values, player_static_values);
        add_frame_state_values(&mut values, context, event.frame_number, &context.players);
        add_spatial_features(&mut values, &context.players);
        rows.push(PbpEventRecord {
            frame_number: Some(event.frame_number),
            event_type: "boost-pickup".to_string(),
            values,
        });
    }
}

fn add_flip_reset_events(
    rows: &mut Vec<PbpEventRecord>,
    context: &PbpContext,
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
    team_size: Option<i32>,
    game_time: &str,
) {
    for event in flip_reset_events(context) {
        let Some(player) = context
            .players
            .iter()
            .find(|player| player.name == event.player_name)
        else {
            continue;
        };
        let mut values = pbp_base_values(
            game_id,
            match_guid,
            replay_name,
            map_id,
            context,
            team_size,
            game_time,
        );
        values.insert("event_type".to_string(), "flip-reset".to_string());
        values.insert("frame_number".to_string(), event.frame_number.to_string());
        values.insert(
            "observed_frame_number".to_string(),
            event.frame_number.to_string(),
        );
        values.insert("flip_reset".to_string(), "true".to_string());
        values.insert("reset_origin".to_string(), event.reset_origin.to_string());
        values.insert("event_team".to_string(), team_name(player.team).to_string());
        insert_seconds_elapsed(&mut values, context, event.frame_number);
        add_event_player(&mut values, &context.players, 1, &event.player_name);
        add_pbp_players(&mut values, player_static_values);
        add_frame_state_values(&mut values, context, event.frame_number, &context.players);
        if let Some(snapshot) = frame_snapshot(context, event.frame_number) {
            if let Some(ball) = snapshot.ball {
                values.insert("event_ball_pos_x".to_string(), ball.pos.x.to_string());
                values.insert("event_ball_pos_y".to_string(), ball.pos.y.to_string());
                values.insert("event_ball_pos_z".to_string(), ball.pos.z.to_string());
            }
            if let Some(player_state) = context
                .players
                .iter()
                .position(|candidate| candidate.name == event.player_name)
                .and_then(|idx| snapshot.players.get(idx))
                .and_then(Option::as_ref)
            {
                values.insert(
                    "aerialing".to_string(),
                    (player_state.entity.pos.z >= CROSSBAR_HEIGHT).to_string(),
                );
            }
        }
        add_spatial_features(&mut values, &context.players);
        rows.push(PbpEventRecord {
            frame_number: Some(event.frame_number),
            event_type: "flip-reset".to_string(),
            values,
        });
    }
}

fn boost_pickup_events(context: &PbpContext) -> Vec<BoostPickupEvent> {
    let mut events = Vec::new();
    let mut previous_boost = vec![None; context.players.len()];
    let mut previous_grant = vec![None; context.players.len()];
    let mut last_pickup_frame = vec![-10_000; context.players.len()];

    for snapshot in &context.frame_states {
        for (idx, player) in context.players.iter().enumerate() {
            let Some(state) = snapshot.players.get(idx).and_then(Option::as_ref) else {
                previous_boost[idx] = None;
                previous_grant[idx] = None;
                continue;
            };
            let Some(current_boost) = state.boost.map(i32::from) else {
                continue;
            };
            let prior_boost = previous_boost[idx];
            let grant = state.boost_collect.map(i32::from);
            let grant_changed =
                grant.is_some() && previous_grant[idx].is_some() && grant != previous_grant[idx];
            let boost_increased = prior_boost
                .map(|prior| boost_units(current_boost) > boost_units(prior))
                .unwrap_or(false);

            if (boost_increased || grant_changed)
                && snapshot.frame_number - last_pickup_frame[idx] > 2
            {
                let amount = boost_pickup_amount(prior_boost, current_boost, grant);
                if amount > 0 {
                    events.push(BoostPickupEvent {
                        frame_number: snapshot.frame_number,
                        player_name: player.name.clone(),
                        amount,
                        pickup_type: boost_pickup_type(prior_boost, current_boost, amount),
                    });
                    last_pickup_frame[idx] = snapshot.frame_number;
                }
            }

            previous_boost[idx] = Some(current_boost);
            previous_grant[idx] = grant;
        }
    }

    events
}

fn flip_reset_events(context: &PbpContext) -> Vec<FlipResetEvent> {
    let mut events = Vec::new();
    let mut previous_dodge_air_count = vec![None; context.players.len()];
    let mut previous_double_jump_air_count = vec![None; context.players.len()];
    let mut previous_refreshed_counter = vec![None; context.players.len()];
    let mut last_reset_frame = vec![-10_000; context.players.len()];

    for snapshot in &context.frame_states {
        for (idx, player) in context.players.iter().enumerate() {
            let Some(state) = snapshot.players.get(idx).and_then(Option::as_ref) else {
                previous_dodge_air_count[idx] = None;
                previous_double_jump_air_count[idx] = None;
                previous_refreshed_counter[idx] = None;
                continue;
            };
            let dodge_reset = state
                .dodge_air_activate_count
                .zip(previous_dodge_air_count[idx])
                .map(|(current, previous)| previous > 0 && current == 0)
                .unwrap_or(false);
            let double_jump_reset = state
                .double_jump_air_activate_count
                .zip(previous_double_jump_air_count[idx])
                .map(|(current, previous)| previous > 0 && current == 0)
                .unwrap_or(false);
            let refreshed_counter = state
                .dodges_refreshed_counter
                .zip(previous_refreshed_counter[idx])
                .map(|(current, previous)| current > previous)
                .unwrap_or(false);
            let refreshed = dodge_reset || double_jump_reset || refreshed_counter;
            if refreshed
                && state.entity.has_pos
                && state.entity.pos.z >= FLIP_RESET_MIN_CAR_Z
                && snapshot.frame_number - last_reset_frame[idx] > FLIP_RESET_FRAME_WINDOW
            {
                if let Some(reset_origin) = flip_refresh_contact(snapshot, idx, player, context) {
                    events.push(FlipResetEvent {
                        frame_number: snapshot.frame_number,
                        player_name: player.name.clone(),
                        reset_origin,
                    });
                    last_reset_frame[idx] = snapshot.frame_number;
                }
            }
            if let Some(value) = state.dodge_air_activate_count {
                previous_dodge_air_count[idx] = Some(value);
            }
            if let Some(value) = state.double_jump_air_activate_count {
                previous_double_jump_air_count[idx] = Some(value);
            }
            if let Some(value) = state.dodges_refreshed_counter {
                previous_refreshed_counter[idx] = Some(value);
            }
        }
    }

    events
}

fn flip_refresh_contact(
    snapshot: &FrameSnapshot,
    player_idx: usize,
    player: &PlayerInfo,
    context: &PbpContext,
) -> Option<&'static str> {
    let Some(state) = snapshot.players.get(player_idx).and_then(Option::as_ref) else {
        return None;
    };
    if !state.entity.has_pos {
        return None;
    }
    if let Some(ball) = snapshot.ball {
        if ball.has_pos
            && (ball_collision_distance(
                ball.pos,
                state.entity,
                player.car_id.parse().unwrap_or(23),
            ) <= FLIP_RESET_CONTACT_DISTANCE
                || underside_ball_contact(
                    ball.pos,
                    state.entity,
                    player.car_id.parse().unwrap_or(23),
                ))
        {
            return Some("ball");
        }
    }
    for (other_idx, other_player) in context.players.iter().enumerate() {
        if other_idx == player_idx {
            continue;
        }
        let Some(other_state) = snapshot.players.get(other_idx).and_then(Option::as_ref) else {
            continue;
        };
        if !other_state.entity.has_pos {
            continue;
        }
        if underside_car_contact(
            state.entity,
            player.car_id.parse().unwrap_or(23),
            other_state.entity,
            other_player.car_id.parse().unwrap_or(23),
        ) {
            return Some(if other_player.team == player.team {
                "teammate"
            } else {
                "opponent"
            });
        }
    }
    None
}

fn underside_ball_contact(ball_pos: Vec3, car_state: EntityState, car_id: i32) -> bool {
    let local = inverse_rotate(
        car_state.rot,
        Vec3 {
            x: ball_pos.x - car_state.pos.x,
            y: ball_pos.y - car_state.pos.y,
            z: ball_pos.z - car_state.pos.z,
        },
    );
    let (length, width, height, offset, elevation) = hitbox_dims(car_id);
    let lower_face = -height / 2.0 + elevation;
    let within_footprint = local.x >= -length / 2.0 + offset - BALL_RADIUS
        && local.x <= length / 2.0 + offset + BALL_RADIUS
        && local.y >= -width / 2.0 - BALL_RADIUS
        && local.y <= width / 2.0 + BALL_RADIUS;
    within_footprint
        && local.z <= lower_face + FLIP_RESET_UNDERSIDE_Z
        && ball_collision_distance(ball_pos, car_state, car_id) <= FLIP_RESET_CONTACT_DISTANCE
}

fn underside_car_contact(
    car_state: EntityState,
    car_id: i32,
    other_state: EntityState,
    other_car_id: i32,
) -> bool {
    let local_other = inverse_rotate(
        car_state.rot,
        Vec3 {
            x: other_state.pos.x - car_state.pos.x,
            y: other_state.pos.y - car_state.pos.y,
            z: other_state.pos.z - car_state.pos.z,
        },
    );
    let (length, width, height, offset, elevation) = hitbox_dims(car_id);
    let lower_face = -height / 2.0 + elevation;
    let within_footprint = local_other.x >= -length / 2.0 + offset - CAR_CONTACT_DISTANCE
        && local_other.x <= length / 2.0 + offset + CAR_CONTACT_DISTANCE
        && local_other.y >= -width / 2.0 - CAR_CONTACT_DISTANCE
        && local_other.y <= width / 2.0 + CAR_CONTACT_DISTANCE;
    within_footprint
        && local_other.z <= lower_face
        && vec_distance(car_state.pos, other_state.pos)
            <= CAR_CONTACT_DISTANCE + hitbox_dims(other_car_id).0 / 2.0
}

fn boost_pickup_amount(prior_boost: Option<i32>, current_boost: i32, grant: Option<i32>) -> i32 {
    let Some(prior_raw) = prior_boost else {
        return 0;
    };
    let prior = boost_units(prior_raw);
    let current = boost_units(current_boost);
    let delta = current - prior;
    if delta <= 0 && grant.is_none() {
        return 0;
    }
    if current == 33 && prior <= 5 {
        return 33;
    }
    if delta > 15 || (current >= 95 && delta > 12) {
        return 100;
    }
    if delta > 0 {
        return 12;
    }
    0
}

fn boost_pickup_type(prior_boost: Option<i32>, current_boost: i32, amount: i32) -> &'static str {
    if amount == 33
        || (boost_units(current_boost) == 33 && boost_units(prior_boost.unwrap_or(0)) <= 5)
    {
        "reset"
    } else if amount == 100 {
        "big"
    } else {
        "small"
    }
}

fn boost_units(raw_boost: i32) -> i32 {
    ((raw_boost as f32) * 100.0 / 255.0).round() as i32
}

fn raw_boost_units(scaled_boost: i32) -> u8 {
    ((scaled_boost as f32) * 255.0 / 100.0).round() as u8
}

fn post_process_pbp_rows(rows: &mut Vec<PbpEventRecord>, players: &[PlayerInfo]) {
    let touch_types = ["touch", "turnover", "pass", "shot", "goal", "kickoff"];
    let slot_by_id = players
        .iter()
        .map(|player| (player.id.clone(), player.slot.clone()))
        .collect::<HashMap<_, _>>();

    for row in rows.iter_mut() {
        if !row.values.contains_key("event_team") {
            let team = row_string(&row.values, "event_player_1_team");
            row.values.insert("event_team".to_string(), team);
        }
        for col in [
            "official_shot",
            "official_goal",
            "official_assist",
            "official_save",
            "official_demo",
            "previous_event_entry",
            "previous_event_exit",
            "controlled",
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
            "pass_in_play",
            "aerialing",
            "air_dribble",
            "ground_dribble",
            "flick_shot",
            "rebound",
            "double_tap",
            "flip-reset",
            "off_flip_reset",
            "off_double_tap",
            "off_wall",
            "off_ceiling",
        ] {
            if !row.values.contains_key(col) {
                row.values.insert(col.to_string(), "false".to_string());
            }
        }
        for col in [
            "official_shot_count",
            "official_goal_count",
            "official_assist_count",
            "official_save_count",
            "official_demo_count",
        ] {
            if !row.values.contains_key(col) {
                row.values.insert(col.to_string(), "0".to_string());
            }
        }
        add_event_location_flags(&mut row.values, &slot_by_id);
    }

    for idx in 0..rows.len() {
        if rows[idx].event_type == "bump" {
            let team_1 = row_string(&rows[idx].values, "event_player_1_team");
            let team_2 = row_string(&rows[idx].values, "event_player_2_team");
            if !team_1.is_empty() && !team_2.is_empty() && team_1 != team_2 {
                let p1_slot =
                    row_string(&rows[idx].values, "event_player_1_id").and_then_lookup(&slot_by_id);
                let p2_slot =
                    row_string(&rows[idx].values, "event_player_2_id").and_then_lookup(&slot_by_id);
                let p1_ball = p1_slot
                    .as_ref()
                    .and_then(|slot| {
                        parse_f32(rows[idx].values.get(&format!("{slot}_distance_to_ball")))
                    })
                    .unwrap_or(f32::MAX);
                let p2_ball = p2_slot
                    .as_ref()
                    .and_then(|slot| {
                        parse_f32(rows[idx].values.get(&format!("{slot}_distance_to_ball")))
                    })
                    .unwrap_or(f32::MAX);
                if p1_ball.min(p2_ball) <= CHALLENGE_TOUCH_BALL_DISTANCE {
                    let winning_team = rows[(idx + 1)..]
                        .iter()
                        .find(|row| touch_types.contains(&row.event_type.as_str()))
                        .map(|row| row_string(&row.values, "event_team"));
                    if winning_team.as_deref() == Some(team_2.as_str()) {
                        swap_event_players(&mut rows[idx].values);
                    }
                    rows[idx].event_type = "challenge".to_string();
                    rows[idx]
                        .values
                        .insert("event_type".to_string(), "challenge".to_string());
                    let team = row_string(&rows[idx].values, "event_player_1_team");
                    rows[idx].values.insert("event_team".to_string(), team);
                }
            }
        }
    }

    let mut last_touch_challenge_frame: HashMap<(String, String), i32> = HashMap::new();
    for idx in 0..rows.len() {
        if rows[idx].event_type != "touch" {
            continue;
        }
        let team_1 = row_string(&rows[idx].values, "event_player_1_team");
        if team_1.is_empty() {
            continue;
        }
        let p1_slot =
            row_string(&rows[idx].values, "event_player_1_id").and_then_lookup(&slot_by_id);
        let challenger = players
            .iter()
            .filter(|player| team_name(player.team) != team_1)
            .filter_map(|player| {
                let distance_to_ball = parse_f32(
                    rows[idx]
                        .values
                        .get(&format!("{}_distance_to_ball", player.slot)),
                )?;
                let distance_to_toucher = p1_slot
                    .as_ref()
                    .and_then(|slot| {
                        parse_f32(
                            rows[idx]
                                .values
                                .get(&format!("{slot}_distance_to_{}", player.slot)),
                        )
                    })
                    .unwrap_or(f32::MAX);
                (distance_to_ball <= CHALLENGE_TOUCH_BALL_DISTANCE
                    && distance_to_toucher <= CHALLENGE_TOUCH_PLAYER_DISTANCE)
                    .then(|| {
                        (
                            player.name.clone(),
                            team_name(player.team).to_string(),
                            distance_to_ball,
                        )
                    })
            })
            .min_by(|left, right| {
                left.2
                    .partial_cmp(&right.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let Some((challenger_name, challenger_team, _)) = challenger else {
            continue;
        };
        let toucher_name = row_string(&rows[idx].values, "event_player_1_name");
        let mut challenge_pair = [toucher_name.clone(), challenger_name.clone()];
        challenge_pair.sort();
        let challenge_key = (challenge_pair[0].clone(), challenge_pair[1].clone());
        let frame_number = rows[idx].frame_number.unwrap_or(i32::MAX);
        if last_touch_challenge_frame
            .get(&challenge_key)
            .map(|prior_frame| frame_number - *prior_frame <= CHALLENGE_EVENT_COOLDOWN_FRAMES)
            .unwrap_or(false)
        {
            continue;
        }
        let winning_team = rows[(idx + 1)..]
            .iter()
            .find(|row| touch_types.contains(&row.event_type.as_str()))
            .map(|row| row_string(&row.values, "event_team"));
        add_event_player(&mut rows[idx].values, players, 2, &challenger_name);
        if winning_team.as_deref() == Some(challenger_team.as_str()) {
            swap_event_players(&mut rows[idx].values);
        }
        rows[idx].event_type = "challenge".to_string();
        rows[idx]
            .values
            .insert("event_type".to_string(), "challenge".to_string());
        let team = row_string(&rows[idx].values, "event_player_1_team");
        rows[idx].values.insert("event_team".to_string(), team);
        last_touch_challenge_frame.insert(challenge_key, frame_number);
    }

    let mut last_challenge_frame: HashMap<(String, String), i32> = HashMap::new();
    for row in rows.iter_mut() {
        if row.event_type != "challenge" {
            continue;
        }
        let player_1 = row_string(&row.values, "event_player_1_name");
        let player_2 = row_string(&row.values, "event_player_2_name");
        if player_1.is_empty() || player_2.is_empty() {
            continue;
        }
        let mut pair = [player_1, player_2];
        pair.sort();
        let key = (pair[0].clone(), pair[1].clone());
        let frame_number = row.frame_number.unwrap_or(i32::MAX);
        let duplicate = last_challenge_frame
            .get(&key)
            .map(|prior_frame| frame_number - *prior_frame <= CHALLENGE_EVENT_COOLDOWN_FRAMES)
            .unwrap_or(false);
        if duplicate {
            let fallback_type = if row.values.contains_key("car_contact_distance") {
                "bump"
            } else {
                "touch"
            };
            row.event_type = fallback_type.to_string();
            row.values
                .insert("event_type".to_string(), fallback_type.to_string());
            let team = row_string(&row.values, "event_player_1_team");
            row.values.insert("event_team".to_string(), team);
        } else {
            last_challenge_frame.insert(key, frame_number);
        }
    }

    for idx in 0..rows.len() {
        if rows[idx].event_type != "touch" {
            continue;
        }
        let team = row_string(&rows[idx].values, "event_team");
        if team.is_empty() {
            continue;
        }
        let mut next_team = String::new();
        for next_row in &rows[(idx + 1)..] {
            if next_row.event_type == "goal"
                || next_row.event_type == "kickoff"
                || next_row.event_type == "challenge"
            {
                break;
            }
            if touch_types.contains(&next_row.event_type.as_str()) {
                next_team = row_string(&next_row.values, "event_team");
                break;
            }
        }
        if !next_team.is_empty() && next_team != team {
            rows[idx].event_type = "turnover".to_string();
            rows[idx]
                .values
                .insert("event_type".to_string(), "turnover".to_string());
        }
    }

    add_contact_microstat_flags(rows, &slot_by_id);
    add_microstat_events(rows);
    rows.sort_by_key(|row| (row.frame_number.unwrap_or(i32::MAX), row.event_type.clone()));

    add_zone_context_flags(rows);

    for idx in 0..rows.len() {
        if let Some(previous_idx) = previous_non_boost_event_idx(rows, idx) {
            let previous_seconds = row_f32(&rows[previous_idx].values, "seconds_elapsed");
            let seconds = row_f32(&rows[idx].values, "seconds_elapsed");
            let previous_event_type = rows[previous_idx].event_type.clone();
            let previous_event_entry = (previous_event_type == "entry").to_string();
            let previous_event_exit = (previous_event_type == "exit").to_string();
            set_float(
                &mut rows[idx].values,
                "seconds_from_last_event",
                seconds.zip(previous_seconds).map(|(now, prev)| now - prev),
            );
            rows[idx]
                .values
                .insert("previous_event_type".to_string(), previous_event_type);
            rows[idx]
                .values
                .insert("previous_event_entry".to_string(), previous_event_entry);
            rows[idx]
                .values
                .insert("previous_event_exit".to_string(), previous_event_exit);
            let ball_now = row_vec(&rows[idx].values, "ball", "pos");
            let ball_prev = row_vec(&rows[previous_idx].values, "ball", "pos");
            let ball_distance = distance_opt(ball_prev, ball_now);
            set_float(
                &mut rows[idx].values,
                "ball_distance_from_last_event",
                ball_distance,
            );
            let ball_angle = angle_opt(ball_prev, ball_now);
            set_float(
                &mut rows[idx].values,
                "ball_angle_from_last_event",
                ball_angle,
            );
            let previous_ball_angle =
                row_f32(&rows[previous_idx].values, "ball_angle_from_last_event");
            set_float(
                &mut rows[idx].values,
                "ball_angle_change_from_last_event",
                angle_delta_opt(previous_ball_angle, ball_angle),
            );
            if let (Some(distance), Some(seconds_delta)) = (
                ball_distance,
                row_f32(&rows[idx].values, "seconds_from_last_event"),
            ) {
                if seconds_delta > 0.0 {
                    let ball_speed = distance / seconds_delta;
                    set_float(
                        &mut rows[idx].values,
                        "ball_speed_from_last_event",
                        Some(ball_speed),
                    );
                    let previous_ball_speed =
                        row_f32(&rows[previous_idx].values, "ball_speed_from_last_event");
                    set_float(
                        &mut rows[idx].values,
                        "ball_speed_change_from_last_event",
                        previous_ball_speed.map(|previous| ball_speed - previous),
                    );
                }
            }
            let ball_vel_now = row_vec(&rows[idx].values, "ball", "vel");
            let ball_vel_prev = row_vec(&rows[previous_idx].values, "ball", "vel");
            if let (Some(now), Some(previous)) = (ball_vel_now, ball_vel_prev) {
                set_float(
                    &mut rows[idx].values,
                    "ball_vel_x_change_from_last_event",
                    Some(now.x - previous.x),
                );
                set_float(
                    &mut rows[idx].values,
                    "ball_vel_y_change_from_last_event",
                    Some(now.y - previous.y),
                );
                set_float(
                    &mut rows[idx].values,
                    "ball_vel_z_change_from_last_event",
                    Some(now.z - previous.z),
                );
            }
            for player in players {
                let now = row_vec(&rows[idx].values, &player.slot, "pos");
                let prev = row_vec(&rows[previous_idx].values, &player.slot, "pos");
                set_float(
                    &mut rows[idx].values,
                    &format!("{}_distance_from_last_event", player.slot),
                    distance_opt(prev, now),
                );
            }
        }
    }
    let mut blue_score = 0;
    let mut orange_score = 0;
    for row in rows.iter_mut() {
        row.values
            .insert("blue_score".to_string(), blue_score.to_string());
        row.values
            .insert("orange_score".to_string(), orange_score.to_string());
        if row.event_type == "goal" {
            if row_string(&row.values, "event_team") == "orange" {
                orange_score += 1;
            } else {
                blue_score += 1;
            }
        }
    }
    for idx in 0..rows.len().saturating_sub(1) {
        if let Some(seconds) = row_f32(&rows[idx + 1].values, "seconds_from_last_event") {
            set_float(&mut rows[idx].values, "event_length", Some(seconds));
        }
    }
    add_weighted_event_history(rows);

    for idx in 0..rows.len() {
        if rows[idx].event_type != "shot" && rows[idx].event_type != "goal" {
            continue;
        }
        let seconds = match row_f32(&rows[idx].values, "seconds_elapsed") {
            Some(value) => value,
            None => continue,
        };
        let team = row_string(&rows[idx].values, "event_team");
        let shooter_id = row_string(&rows[idx].values, "event_player_1_id");
        let mut flags: HashMap<&str, bool> = HashMap::new();
        for prior in rows[..idx].iter().rev() {
            let prior_seconds = match row_f32(&prior.values, "seconds_elapsed") {
                Some(value) => value,
                None => continue,
            };
            let delta = seconds - prior_seconds;
            if delta > DRIBBLE_WINDOW_SECONDS && delta > OFF_CHALLENGE_SECONDS {
                break;
            }
            if delta <= REBOUND_SECONDS
                && (prior.event_type == "shot" || prior.event_type == "goal")
            {
                flags.insert("rebound", true);
            }
            if delta <= OFF_DEMO_SECONDS
                && prior.event_type == "demo"
                && row_string(&prior.values, "event_team") == team
            {
                flags.insert("off_demo", true);
            }
            if delta <= OFF_KICKOFF_SECONDS && prior.event_type == "kickoff" {
                flags.insert("off_kickoff", true);
            }
            if delta <= OFF_CHALLENGE_SECONDS
                && prior.event_type == "challenge"
                && row_string(&prior.values, "event_team") == team
            {
                flags.insert("off_challenge_win", true);
            }
            if delta <= OFF_DEMO_SECONDS
                && prior.event_type == "bump"
                && row_string(&prior.values, "event_team") == team
            {
                flags.insert("off_bump", true);
            }
            if delta <= OFF_CHALLENGE_SECONDS
                && prior.event_type == "pass"
                && row_string(&prior.values, "event_team") == team
            {
                flags.insert("pass_in_play", true);
            }
            let same_shooter = row_string(&prior.values, "event_player_1_id") == shooter_id;
            if delta <= DRIBBLE_WINDOW_SECONDS && prior.event_type == "air-dribble" && same_shooter
            {
                flags.insert("off_air_dribble", true);
            }
            if delta <= DRIBBLE_WINDOW_SECONDS
                && prior.event_type == "ground-dribble"
                && same_shooter
            {
                flags.insert("off_ground_dribble", true);
            }
            if delta <= FLICK_WINDOW_SECONDS && prior.event_type == "flick" && same_shooter {
                flags.insert("off_flick", true);
            }
            if delta <= OFF_FLIP_RESET_SECONDS && prior.event_type == "flip-reset" && same_shooter {
                flags.insert("off_flip_reset", true);
            }
        }
        if shot_is_double_tap(rows, idx, &slot_by_id) {
            flags.insert("double_tap", true);
            flags.insert("off_double_tap", true);
        }
        for (key, value) in flags {
            rows[idx].values.insert(key.to_string(), value.to_string());
        }
        if !row_string(&rows[idx].values, "event_player_2_id").is_empty() {
            rows[idx]
                .values
                .insert("pass_in_play".to_string(), "true".to_string());
        }
    }
}

fn add_contact_microstat_flags(rows: &mut [PbpEventRecord], slot_by_id: &HashMap<String, String>) {
    for row in rows.iter_mut() {
        if !microstat_contact_event(&row.event_type) {
            continue;
        }
        let player_id = row_string(&row.values, "event_player_1_id");
        if let Some(slot) = slot_by_id.get(&player_id) {
            let z = parse_f32(row.values.get(&format!("{slot}_pos_z"))).unwrap_or(0.0);
            row.values
                .insert("aerialing".to_string(), (z >= CROSSBAR_HEIGHT).to_string());
        }
    }

    let mut air_dribble = vec![false; rows.len()];
    let mut ground_dribble = vec![false; rows.len()];
    let mut flick = vec![false; rows.len()];
    for idx in 0..rows.len() {
        if !microstat_contact_event(&rows[idx].event_type) {
            continue;
        }
        let player_id = row_string(&rows[idx].values, "event_player_1_id");
        if player_id.is_empty() {
            continue;
        }
        let seconds = match row_f32(&rows[idx].values, "seconds_elapsed") {
            Some(value) => value,
            None => continue,
        };
        let current_hood = hood_dribble_control(&rows[idx].values, slot_by_id, &player_id);
        let current_aerial = truthy(rows[idx].values.get("aerialing"));
        let current_flip = player_flipped(&rows[idx].values, slot_by_id, &player_id);
        for prior_idx in (0..idx).rev() {
            if !microstat_contact_event(&rows[prior_idx].event_type) {
                continue;
            }
            if row_string(&rows[prior_idx].values, "event_player_1_id") != player_id {
                continue;
            }
            let prior_seconds = match row_f32(&rows[prior_idx].values, "seconds_elapsed") {
                Some(value) => value,
                None => continue,
            };
            let delta = seconds - prior_seconds;
            if delta <= 0.0 {
                continue;
            }
            if delta > DRIBBLE_WINDOW_SECONDS {
                break;
            }
            let prior_aerial = truthy(rows[prior_idx].values.get("aerialing"));
            let prior_hood = hood_dribble_control(&rows[prior_idx].values, slot_by_id, &player_id);
            if current_aerial || prior_aerial {
                air_dribble[idx] = true;
            }
            if current_hood && prior_hood && !air_dribble[idx] {
                ground_dribble[idx] = true;
            }
            if ground_dribble[idx] && delta <= FLICK_WINDOW_SECONDS && current_flip {
                let ball_vel_z = row_f32(&rows[idx].values, "ball_vel_z").unwrap_or(0.0);
                flick[idx] = ball_vel_z > 250.0;
            }
            break;
        }
    }
    for idx in 0..rows.len() {
        if air_dribble[idx] {
            rows[idx]
                .values
                .insert("air_dribble".to_string(), "true".to_string());
        }
        if ground_dribble[idx] {
            rows[idx]
                .values
                .insert("ground_dribble".to_string(), "true".to_string());
        }
        if flick[idx] {
            rows[idx]
                .values
                .insert("flick_shot".to_string(), "true".to_string());
        }
    }
}

fn add_microstat_events(rows: &mut Vec<PbpEventRecord>) {
    let mut additions = Vec::new();
    let mut seen = HashSet::new();
    for row in rows.iter() {
        for (flag, event_type) in [
            ("air_dribble", "air-dribble"),
            ("ground_dribble", "ground-dribble"),
            ("flick_shot", "flick"),
        ] {
            if !truthy(row.values.get(flag)) {
                continue;
            }
            let player_id = row_string(&row.values, "event_player_1_id");
            let key = (
                event_type.to_string(),
                player_id,
                row.frame_number.unwrap_or(-1),
            );
            if !seen.insert(key) {
                continue;
            }
            let mut values = row.values.clone();
            values.insert("event_type".to_string(), event_type.to_string());
            clear_official_stat_values(&mut values);
            additions.push(PbpEventRecord {
                frame_number: row.frame_number,
                event_type: event_type.to_string(),
                values,
            });
        }
    }
    rows.extend(additions);
}

fn clear_official_stat_values(values: &mut RowValues) {
    for key in [
        "official_shot",
        "official_goal",
        "official_assist",
        "official_save",
        "official_demo",
    ] {
        values.insert(key.to_string(), "false".to_string());
    }
    for key in [
        "official_shot_count",
        "official_goal_count",
        "official_assist_count",
        "official_save_count",
        "official_demo_count",
    ] {
        values.insert(key.to_string(), "0".to_string());
    }
}

fn microstat_contact_event(event_type: &str) -> bool {
    matches!(
        event_type,
        "touch" | "turnover" | "pass" | "shot" | "goal" | "kickoff" | "challenge" | "bump"
    )
}

fn ball_contact_event(event_type: &str) -> bool {
    matches!(
        event_type,
        "touch" | "turnover" | "pass" | "shot" | "goal" | "kickoff" | "challenge"
    )
}

fn shot_is_double_tap(
    rows: &[PbpEventRecord],
    shot_idx: usize,
    slot_by_id: &HashMap<String, String>,
) -> bool {
    let shot = &rows[shot_idx];
    let shooter_id = row_string(&shot.values, "event_player_1_id");
    if shooter_id.is_empty() {
        return false;
    }
    let team = row_string(&shot.values, "event_team");
    if team.is_empty() {
        return false;
    }
    let shot_seconds = match row_f32(&shot.values, "seconds_elapsed") {
        Some(value) => value,
        None => return false,
    };
    let mut saw_back_wall_car_touch = false;
    for prior in rows[..shot_idx].iter().rev() {
        if prior.event_type == "kickoff" || prior.event_type == "goal" {
            break;
        }
        let prior_seconds = match row_f32(&prior.values, "seconds_elapsed") {
            Some(value) => value,
            None => continue,
        };
        let delta = shot_seconds - prior_seconds;
        if delta <= 0.0 {
            continue;
        }
        if delta > DOUBLE_TAP_SECONDS {
            break;
        }
        if !ball_contact_event(&prior.event_type) {
            continue;
        }
        let prior_player_id = row_string(&prior.values, "event_player_1_id");
        if prior_player_id == shooter_id {
            if double_tap_setup_contact(&prior.values, &team, slot_by_id) || saw_back_wall_car_touch
            {
                return true;
            }
            continue;
        }
        if contact_near_offensive_back_wall(&prior.values, &team, slot_by_id) {
            saw_back_wall_car_touch = true;
            continue;
        }
        break;
    }
    false
}

fn double_tap_setup_contact(
    values: &RowValues,
    team: &str,
    slot_by_id: &HashMap<String, String>,
) -> bool {
    let Some(ball_pos) =
        row_vec(values, "ball", "pos").or_else(|| row_vec(values, "event_ball", "pos"))
    else {
        return false;
    };
    let offensive_y = offensive_back_wall_y(team);
    if (ball_pos.y - offensive_y).abs() <= DOUBLE_TAP_BACK_WALL_DISTANCE {
        return true;
    }
    if let Some(ball_vel) =
        row_vec(values, "ball", "vel").or_else(|| row_vec(values, "event_ball", "vel"))
    {
        let y_delta = offensive_y - ball_pos.y;
        if y_delta * ball_vel.y > 0.0 && ball_vel.y.abs() > f32::EPSILON {
            let time_to_wall = y_delta / ball_vel.y;
            if time_to_wall > 0.0 && time_to_wall <= DOUBLE_TAP_BACK_WALL_PROJECTION_SECONDS {
                let projected_x = ball_pos.x + ball_vel.x * time_to_wall;
                let projected_z = ball_pos.z + ball_vel.z * time_to_wall
                    - 0.5 * GRAVITY * time_to_wall * time_to_wall;
                if projected_x.abs() <= SIDE_WALL_X + BALL_RADIUS
                    && (-BALL_RADIUS..=CEILING_Z + BALL_RADIUS).contains(&projected_z)
                {
                    return true;
                }
            }
        }
    }
    contact_near_offensive_back_wall(values, team, slot_by_id)
}

fn contact_near_offensive_back_wall(
    values: &RowValues,
    team: &str,
    slot_by_id: &HashMap<String, String>,
) -> bool {
    let offensive_y = offensive_back_wall_y(team);
    let ball_near_wall = row_vec(values, "ball", "pos")
        .or_else(|| row_vec(values, "event_ball", "pos"))
        .map(|pos| (pos.y - offensive_y).abs() <= DOUBLE_TAP_BACK_WALL_DISTANCE)
        .unwrap_or(false);
    for player_key in ["event_player_1_id", "event_player_2_id"] {
        let player_id = row_string(values, player_key);
        if player_id.is_empty() {
            continue;
        }
        if let Some(slot) = slot_by_id.get(&player_id) {
            if let Some(pos) = row_vec(values, slot, "pos") {
                if (pos.y - offensive_y).abs() <= DOUBLE_TAP_CAR_BACK_WALL_DISTANCE {
                    return true;
                }
            }
        }
    }
    ball_near_wall
}

fn offensive_back_wall_y(team: &str) -> f32 {
    if team == "orange" {
        -BACK_WALL_Y
    } else {
        BACK_WALL_Y
    }
}

fn add_zone_context_flags(rows: &mut [PbpEventRecord]) {
    for idx in 0..rows.len() {
        if rows[idx].event_type == "kickoff" {
            continue;
        }
        let seconds = match row_f32(&rows[idx].values, "seconds_elapsed") {
            Some(value) => value,
            None => continue,
        };
        let team = row_string(&rows[idx].values, "event_team");
        if team.is_empty() {
            continue;
        }
        let allow_entry_exit = !matches!(rows[idx].event_type.as_str(), "shot" | "goal")
            || event_in_offensive_third(&rows[idx].values, &team);
        let mut off_controlled_entry = false;
        let mut off_controlled_exit = false;
        let mut off_retrieval = false;
        let mut off_uncontrolled_entry = false;
        let mut off_uncontrolled_exit = false;
        for prior in rows[..idx].iter().rev() {
            if prior.event_type == "kickoff" || prior.event_type == "goal" {
                break;
            }
            let prior_seconds = match row_f32(&prior.values, "seconds_elapsed") {
                Some(value) => value,
                None => continue,
            };
            let delta = seconds - prior_seconds;
            if delta <= 0.0 {
                continue;
            }
            if delta > OFF_ZONE_EVENT_SECONDS {
                break;
            }
            if row_string(&prior.values, "event_team") != team {
                continue;
            }
            match prior.event_type.as_str() {
                "retrieval" => {
                    off_retrieval = true;
                }
                "entry" if allow_entry_exit => {
                    if truthy(prior.values.get("controlled")) {
                        off_controlled_entry = true;
                    } else {
                        off_uncontrolled_entry = true;
                    }
                }
                "exit" if allow_entry_exit => {
                    if truthy(prior.values.get("controlled")) {
                        off_controlled_exit = true;
                    } else {
                        off_uncontrolled_exit = true;
                    }
                }
                _ => {}
            }
        }
        if off_controlled_entry {
            rows[idx]
                .values
                .insert("off_controlled_entry".to_string(), "true".to_string());
        }
        if off_controlled_exit {
            rows[idx]
                .values
                .insert("off_controlled_exit".to_string(), "true".to_string());
        }
        if off_retrieval {
            rows[idx]
                .values
                .insert("off_retrieval".to_string(), "true".to_string());
        }
        if off_uncontrolled_entry {
            rows[idx]
                .values
                .insert("off_uncontrolled_entry".to_string(), "true".to_string());
        }
        if off_uncontrolled_exit {
            rows[idx]
                .values
                .insert("off_uncontrolled_exit".to_string(), "true".to_string());
        }
    }
}

fn event_in_offensive_third(values: &RowValues, team: &str) -> bool {
    let y = row_f32(values, "event_ball_pos_y")
        .or_else(|| row_f32(values, "ball_pos_y"))
        .unwrap_or(0.0);
    let third = BACK_WALL_Y / 3.0;
    if team == "orange" {
        y < -third
    } else {
        y > third
    }
}

fn add_weighted_event_history(rows: &mut [PbpEventRecord]) {
    let event_types = [
        "touch",
        "turnover",
        "pass",
        "shot",
        "goal",
        "save",
        "kickoff",
        "demo",
        "bump",
        "challenge",
        "entry",
        "exit",
        "retrieval",
    ];
    let mut weighted_counts = vec![0.0_f32; event_types.len()];
    let mut weighted_total = 0.0_f32;
    let mut last_seconds: Option<f32> = None;
    let mut kickoff_seconds: Option<f32> = None;

    for row in rows.iter_mut() {
        if row.event_type == "boost-pickup" {
            continue;
        }
        let seconds = match row_f32(&row.values, "seconds_elapsed") {
            Some(value) => value,
            None => continue,
        };
        if let Some(prior_seconds) = last_seconds {
            let decay =
                0.5_f32.powf(((seconds - prior_seconds) / HISTORY_HALF_LIFE_SECONDS).max(0.0));
            weighted_total *= decay;
            for value in &mut weighted_counts {
                *value *= decay;
            }
        }

        if row.event_type == "kickoff" {
            weighted_counts.fill(0.0);
            weighted_total = 0.0;
            kickoff_seconds = Some(seconds);
        }

        if row.event_type == "shot" || row.event_type == "goal" {
            if let Some(start_seconds) = kickoff_seconds {
                set_float(
                    &mut row.values,
                    "history_seconds_since_kickoff",
                    Some(seconds - start_seconds),
                );
            }
            set_float(
                &mut row.values,
                "history_weighted_event_count",
                Some(weighted_total),
            );
            for (idx, event_type) in event_types.iter().enumerate() {
                set_float(
                    &mut row.values,
                    &format!("history_weighted_{event_type}_count"),
                    Some(weighted_counts[idx]),
                );
            }
        }

        weighted_total += 1.0;
        if let Some(idx) = event_types
            .iter()
            .position(|event_type| *event_type == row.event_type)
        {
            weighted_counts[idx] += 1.0;
        }
        if row.event_type == "goal" {
            weighted_counts.fill(0.0);
            weighted_total = 0.0;
            kickoff_seconds = None;
        }
        last_seconds = Some(seconds);
    }
}

fn previous_non_boost_event_idx(rows: &[PbpEventRecord], idx: usize) -> Option<usize> {
    (0..idx)
        .rev()
        .find(|prior_idx| rows[*prior_idx].event_type != "boost-pickup")
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
        if value.is_finite() {
            values.insert(key.to_string(), value.to_string());
        }
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
        "pass_in_play",
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
        "title_id",
        "is_bot",
        "first_frame_in_game",
        "time_in_game",
        "party_leader_id",
        "mmr",
        "car_id",
        "car_name",
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
        "boost_raw",
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
        "jump_air_activate_count",
        "double_jump_air_activate_count",
        "dodge_air_activate_count",
        "dodges_refreshed_counter",
        "supersonic",
        "distance_to_ball",
        "angle_to_ball",
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
        "orange_player_1",
        "orange_player_2",
        "orange_player_3",
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
