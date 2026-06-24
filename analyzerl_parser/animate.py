"""Replay animation helpers backed by the bundled AnalyzeRL parser."""

import csv
import io
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Literal

from .parse import _boxcars_binary

np = None
animation = None
plt = None
Polygon = None
Rectangle = None
Button = None
Slider = None
Poly3DCollection = None

def boxcars_binary():
    return _boxcars_binary()


SIDE_WALL_X = 4096.0
BACK_NET_Y = 6000.0
BACK_WALL_Y = 5120.0
GOAL_CENTER_TO_POST = 892.755
CORNER_CATHETUS_LENGTH = 1152.0
CEILING_Z = 2044.0
DEMO_RESPAWN_SECONDS = 3.0
ANIMATION_FRAMES_PER_SECOND = 30.0
DEFAULT_3D_VIEW_ELEV = 28
DEFAULT_3D_VIEW_AZIM = -64
DEFAULT_HIDDEN_EVENT_TYPES = {'boost-pickup', 'game-join', 'kickoff', 'respawn'}
BIG_BOOST_PADS = [
    (-3072,-4096), (3072,-4096), 
    (-3584,0), (3584,0), 
    (-3072,4096), (3072,4096)
]
SMALL_BOOST_PADS = [
    (-1792, -4184),
    (0, -4240),
    (1792, -4184),
    (-940, -3308),
    (940, -3308),
    (0, -2816),
    (-3584, -2484),
    (-1788, -2300),
    (1788, -2300),
    (3584, -2484),
    (-2048, -1036),
    (0, -1024),
    (2048, -1036),
    (-1024, 0),
    (1024, 0),
    (-2048, 1036),
    (0, 1024),
    (2048, 1036),
    (-3584, 2484),
    (-1788, 2300),
    (1788, 2300),
    (3584, 2484),
    (0, 2816),
    (-940, 3308),
    (940, 3308),
    (-1792, 4184),
    (0, 4240),
    (1792, 4184),
]

def ensure_numpy():
    global np
    if np is None:
        import numpy as numpy_module
        np = numpy_module


def ensure_matplotlib():
    global animation, plt, Polygon, Rectangle, Button, Slider, Poly3DCollection
    if plt is not None:
        return
    import matplotlib
    if "agg" in matplotlib.get_backend().lower():
        for backend in ["TkAgg", "QtAgg", "Qt5Agg"]:
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as mpl_plt
    from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRectangle
    from matplotlib.widgets import Button as MplButton, Slider as MplSlider
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as MplPoly3DCollection

    animation = mpl_animation
    plt = mpl_plt
    Polygon = MplPolygon
    Rectangle = MplRectangle
    Button = MplButton
    Slider = MplSlider
    Poly3DCollection = MplPoly3DCollection


def parse_number(value, default=float('nan')):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def boost_amount(value):
    boost = parse_number(value)
    if not math.isfinite(boost):
        return None
    if boost > 100:
        boost = round(boost * 100.0 / 255.0)
    return int(min(max(boost, 0), 100))


def load_replay_animation(replay_path, parser_path=None):
    command = [
        parser_path or boxcars_binary(),
        'animate-json',
        '--replay',
        replay_path,
        '--frame-step',
        '1',
    ]
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
    )
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    payload = json.loads(result.stdout)
    payload['pbp'] = list(csv.DictReader(io.StringIO(payload.pop('pbp_csv', ''))))
    return payload


def apply_xg_to_animation_rows(rows, xg_model_path=None):
    if xg_model_path is None or not rows:
        return rows

    import polars as pl

    from .xg import apply_xg_to_pbp

    scored = apply_xg_to_pbp(pl.DataFrame(rows), xg_model_path)
    return scored.to_dicts()


def event_frame(row):
    frame = row.get('observed_frame_number') or row.get('frame_number') or row.get('recorded_frame_number')
    try:
        return int(float(frame))
    except (TypeError, ValueError):
        return None


def event_feed_frame(row):
    frame = row.get('recorded_frame_number') or row.get('frame_number') or row.get('observed_frame_number')
    try:
        return int(float(frame))
    except (TypeError, ValueError):
        return None


def event_label(row):
    event_type = row.get('event_type', '')
    player_1 = row.get('event_player_1_name', '')
    player_2 = row.get('event_player_2_name', '')
    team = row.get('event_team', '')
    xg_value = parse_number(row.get('xG'))
    xg_suffix = f' | xG {xg_value:.2f}' if event_type in ['shot', 'goal'] and math.isfinite(xg_value) else ''
    if event_type in ['demo', 'bump', 'challenge'] and player_2:
        return f'{event_type}: {player_1} -> {player_2}'
    if event_type in ['shot', 'goal'] and player_1 and player_2:
        return f'{event_type}: {player_1} | assist {player_2} ({team}){xg_suffix}'
    if event_type in ['shot', 'goal'] and player_1:
        return f'{event_type}: {player_1} ({team}){xg_suffix}'
    if player_1:
        return f'{event_type}: {player_1} ({team})'
    return f'{event_type}'


def build_event_index(rows, event_types=None):
    allowed = None
    if event_types:
        allowed = {value.strip() for value in event_types.split(',') if value.strip()}
    by_frame = {}
    for row in rows:
        if allowed is None and row.get('event_type') in DEFAULT_HIDDEN_EVENT_TYPES:
            continue
        if allowed is not None and row.get('event_type') not in allowed:
            continue
        frame = event_feed_frame(row)
        if frame is None:
            continue
        by_frame.setdefault(frame, []).append(row)
    return by_frame


def build_score_timeline(rows):
    timeline = []
    for row in rows:
        frame = event_frame(row)
        if frame is None:
            continue
        blue_score = int(parse_number(row.get('blue_score'), 0))
        orange_score = int(parse_number(row.get('orange_score'), 0))
        if row.get('event_type') == 'goal':
            if row.get('event_team') == 'orange':
                orange_score += 1
            else:
                blue_score += 1
        timeline.append((frame, blue_score, orange_score))
    timeline.sort(key=lambda value: value[0])
    return timeline


def score_at_frame(timeline, frame_number):
    blue_score = 0
    orange_score = 0
    for frame, blue_value, orange_value in timeline:
        if frame > frame_number:
            break
        blue_score = blue_value
        orange_score = orange_value
    return blue_score, orange_score


def replay_title(payload):
    return payload.get('game_id') or 'AnalyzeRL Replay'


def team_score_title(payload, blue_score, orange_score):
    blue_team = payload.get('blue_team_name') or 'Blue'
    orange_team = payload.get('orange_team_name') or 'Orange'
    return f'{blue_team} {blue_score} - {orange_score} {orange_team}'


def build_demo_windows(rows):
    windows = []
    for row in rows:
        if row.get('event_type') != 'demo':
            continue
        frame = event_frame(row)
        if frame is None:
            continue
        seconds = parse_number(row.get('seconds_elapsed'))
        victim_id = row.get('event_player_2_id', '')
        victim_name = row.get('event_player_2_name', '')
        if not victim_id and not victim_name:
            continue
        windows.append({
            'victim_id': victim_id,
            'victim_name': victim_name,
            'start_frame': frame,
            'end_frame': frame + int(DEMO_RESPAWN_SECONDS * ANIMATION_FRAMES_PER_SECOND),
            'start_seconds': seconds,
            'end_seconds': seconds + DEMO_RESPAWN_SECONDS if math.isfinite(seconds) else float('nan'),
        })
    return windows


def build_kickoff_reset_windows(frames):
    windows = []
    in_reset = False
    last_start = -10_000
    for frame in frames:
        frame_number = int(frame.get('frame_number', -1))
        ball = frame.get('ball') or {}
        pos = ball.get('pos') or []
        vel = ball.get('vel') or []
        centered = (
            len(pos) >= 3
            and len(vel) >= 3
            and abs(float(pos[0])) <= 25.0
            and abs(float(pos[1])) <= 25.0
            and 80.0 <= float(pos[2]) <= 110.0
            and math.sqrt(float(vel[0]) ** 2 + float(vel[1]) ** 2 + float(vel[2]) ** 2) <= 150.0
        )
        if centered and not in_reset and frame_number - last_start > 300:
            windows.append((frame_number, frame_number + int(ANIMATION_FRAMES_PER_SECOND * 3)))
            last_start = frame_number
        in_reset = centered
    return windows


def kickoff_boost_value(frame_number, boost, kickoff_windows):
    if any(start <= frame_number <= end for start, end in kickoff_windows):
        return 33
    return boost


def player_is_demoed(player, frame_number, seconds, demo_windows):
    player_id = player.get('id', '')
    player_name = player.get('name', '')
    for window in demo_windows:
        if window['victim_id']:
            if player_id != window['victim_id']:
                continue
        elif player_name != window['victim_name']:
            continue
        if window['start_frame'] <= frame_number < window['end_frame']:
            return True
        if math.isfinite(seconds) and math.isfinite(window['start_seconds']):
            if window['start_seconds'] <= seconds < window['end_seconds']:
                return True
    return False


def frame_position(frame, key):
    value = frame.get(key)
    if value is None:
        return None
    pos = value.get('pos')
    if not pos or len(pos) < 3:
        return None
    return float(pos[0]), float(pos[1]), float(pos[2])


def draw_field(ax):
    field_points = [
        (-SIDE_WALL_X + CORNER_CATHETUS_LENGTH, -BACK_WALL_Y),
        (SIDE_WALL_X - CORNER_CATHETUS_LENGTH, -BACK_WALL_Y),
        (SIDE_WALL_X, -BACK_WALL_Y + CORNER_CATHETUS_LENGTH),
        (SIDE_WALL_X, BACK_WALL_Y - CORNER_CATHETUS_LENGTH),
        (SIDE_WALL_X - CORNER_CATHETUS_LENGTH, BACK_WALL_Y),
        (-SIDE_WALL_X + CORNER_CATHETUS_LENGTH, BACK_WALL_Y),
        (-SIDE_WALL_X, BACK_WALL_Y - CORNER_CATHETUS_LENGTH),
        (-SIDE_WALL_X, -BACK_WALL_Y + CORNER_CATHETUS_LENGTH),
    ]
    ax.add_patch(
        Polygon(
            field_points,
            closed=True,
            fill=False,
            edgecolor='#dfe5ec',
            linewidth=1.7,
            joinstyle='miter',
            zorder=1,
        )
    )
    net_depth = BACK_NET_Y - BACK_WALL_Y
    ax.add_patch(
        Rectangle(
            (-GOAL_CENTER_TO_POST, -BACK_NET_Y),
            GOAL_CENTER_TO_POST * 2,
            net_depth,
            facecolor='#1f78c8',
            edgecolor='#4aa3ff',
            linewidth=1.5,
            alpha=0.18,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (-GOAL_CENTER_TO_POST, BACK_WALL_Y),
            GOAL_CENTER_TO_POST * 2,
            net_depth,
            facecolor='#d9632f',
            edgecolor='#f17d3f',
            linewidth=1.5,
            alpha=0.18,
            zorder=0,
        )
    )
    ax.plot(
        [-GOAL_CENTER_TO_POST, GOAL_CENTER_TO_POST],
        [-BACK_WALL_Y, -BACK_WALL_Y],
        color='#4aa3ff',
        linewidth=4,
        solid_capstyle='butt',
        zorder=2,
    )
    ax.plot(
        [-GOAL_CENTER_TO_POST, GOAL_CENTER_TO_POST],
        [BACK_WALL_Y, BACK_WALL_Y],
        color='#f17d3f',
        linewidth=4,
        solid_capstyle='butt',
        zorder=2,
    )
    for x_value in [-GOAL_CENTER_TO_POST, GOAL_CENTER_TO_POST]:
        ax.plot([x_value, x_value], [-BACK_NET_Y, -BACK_WALL_Y], color='#4aa3ff', linewidth=1.2, alpha=0.75, zorder=1)
        ax.plot([x_value, x_value], [BACK_WALL_Y, BACK_NET_Y], color='#f17d3f', linewidth=1.2, alpha=0.75, zorder=1)
    ax.axhline(0, color='#8792a2', linewidth=0.8, alpha=0.5, zorder=0)
    ax.axhline(BACK_WALL_Y / 3, color='#8792a2', linewidth=0.6, alpha=0.28, zorder=0)
    ax.axhline(-BACK_WALL_Y / 3, color='#8792a2', linewidth=0.6, alpha=0.28, zorder=0)


def draw_boost_pads_2d(ax):
    ensure_numpy()
    if SMALL_BOOST_PADS:
        small = np.asarray(SMALL_BOOST_PADS, dtype=np.float64)
        ax.scatter(small[:, 0], small[:, 1], s=18, c='#ffd24a', edgecolors='#111111',
                   linewidths=0.35, alpha=0.85, zorder=3)
    if BIG_BOOST_PADS:
        big = np.asarray(BIG_BOOST_PADS, dtype=np.float64)
        ax.scatter(big[:, 0], big[:, 1], s=58, c='#ff9b2f', edgecolors='#111111',
                   linewidths=0.6, alpha=0.95, zorder=3)


def draw_field_3d(ax):
    field_points = [
        (-SIDE_WALL_X + CORNER_CATHETUS_LENGTH, -BACK_WALL_Y, 0),
        (SIDE_WALL_X - CORNER_CATHETUS_LENGTH, -BACK_WALL_Y, 0),
        (SIDE_WALL_X, -BACK_WALL_Y + CORNER_CATHETUS_LENGTH, 0),
        (SIDE_WALL_X, BACK_WALL_Y - CORNER_CATHETUS_LENGTH, 0),
        (SIDE_WALL_X - CORNER_CATHETUS_LENGTH, BACK_WALL_Y, 0),
        (-SIDE_WALL_X + CORNER_CATHETUS_LENGTH, BACK_WALL_Y, 0),
        (-SIDE_WALL_X, BACK_WALL_Y - CORNER_CATHETUS_LENGTH, 0),
        (-SIDE_WALL_X, -BACK_WALL_Y + CORNER_CATHETUS_LENGTH, 0),
    ]
    xs, ys, zs = zip(*(field_points + [field_points[0]]))
    ax.plot(xs, ys, zs, color='#dfe5ec', linewidth=1.5)
    ax.plot([-SIDE_WALL_X, SIDE_WALL_X], [0, 0], [0, 0], color='#8792a2', linewidth=0.8, alpha=0.45)
    for y_value, color in [(-BACK_WALL_Y, '#4aa3ff'), (BACK_WALL_Y, '#f17d3f')]:
        ax.plot(
            [-GOAL_CENTER_TO_POST, GOAL_CENTER_TO_POST],
            [y_value, y_value],
            [0, 0],
            color=color,
            linewidth=4,
        )
        posts = [
            [(-GOAL_CENTER_TO_POST, y_value, 0), (-GOAL_CENTER_TO_POST, y_value, 642.775)],
            [(GOAL_CENTER_TO_POST, y_value, 0), (GOAL_CENTER_TO_POST, y_value, 642.775)],
            [(-GOAL_CENTER_TO_POST, y_value, 642.775), (GOAL_CENTER_TO_POST, y_value, 642.775)],
        ]
        for segment in posts:
            ax.plot(
                [point[0] for point in segment],
                [point[1] for point in segment],
                [point[2] for point in segment],
                color=color,
                linewidth=1.5,
            )
    floor = Poly3DCollection([field_points], facecolor='#20252b', edgecolor='none', alpha=0.28)
    ax.add_collection3d(floor)


def draw_boost_pads_3d(ax):
    ensure_numpy()
    if SMALL_BOOST_PADS:
        small = np.asarray(SMALL_BOOST_PADS, dtype=np.float64)
        ax.scatter(small[:, 0], small[:, 1], np.full(len(small), 10.0),
                   s=16, c='#ffd24a', edgecolors='#111111', linewidths=0.3,
                   alpha=0.9, depthshade=False, zorder=3)
    if BIG_BOOST_PADS:
        big = np.asarray(BIG_BOOST_PADS, dtype=np.float64)
        ax.scatter(big[:, 0], big[:, 1], np.full(len(big), 18.0),
                   s=56, c='#ff9b2f', edgecolors='#111111', linewidths=0.6,
                   alpha=0.98, depthshade=False, zorder=3)


def animate_replay(
    replay_path: str | os.PathLike[str],
    event_window_frames: int = 45,
    event_types: str | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    parser_path: str | os.PathLike[str] | None = None,
    render_mode: Literal['2d', '3d'] = '3d',
    export_path: str | os.PathLike[str] | None = None,
    view_elev: int = DEFAULT_3D_VIEW_ELEV,
    view_azim: int = DEFAULT_3D_VIEW_AZIM,
    xg_model_path: str | os.PathLike[str] | None = None,
) -> Any:
    """Animate a replay interactively or export it as video.

    Args:
        replay_path: Replay file to animate.
        event_window_frames: Number of prior frames shown in the event feed.
        event_types: Optional comma-separated event filter.
        start_frame: Optional first frame to render.
        end_frame: Optional last frame to render.
        parser_path: Optional explicit parser executable path.
        render_mode: ``2d`` or ``3d``.
        export_path: Optional GIF or MP4 output path.
        view_elev: Initial 3D camera elevation.
        view_azim: Initial 3D camera azimuth.
        xg_model_path: Optional saved xG model file or folder. When provided,
            shot and goal events show an ``xG`` value.

    Returns:
        An export path for export mode, or the GUI timer for interactive mode.
    """
    render_mode = str(render_mode).lower()
    payload = load_replay_animation(replay_path, parser_path=parser_path)
    payload['pbp'] = apply_xg_to_animation_rows(payload['pbp'], xg_model_path)
    frames = payload['frames']
    if start_frame is not None:
        frames = [frame for frame in frames if frame['frame_number'] >= start_frame]
    if end_frame is not None:
        frames = [frame for frame in frames if frame['frame_number'] <= end_frame]
    if not frames:
        raise ValueError('No frames available for the requested range')

    if export_path is None:
        try:
            import matplotlib
            if "agg" in matplotlib.get_backend().lower():
                matplotlib.use("TkAgg", force=True)
        except Exception:
            pass
    ensure_matplotlib()
    events_by_frame = build_event_index(payload['pbp'], event_types=event_types)
    ensure_numpy()
    event_frames = np.asarray(sorted(events_by_frame), dtype=np.int32)
    score_timeline = build_score_timeline(payload['pbp'])
    demo_windows = build_demo_windows(payload['pbp'])
    kickoff_windows = build_kickoff_reset_windows(frames)
    is_3d = render_mode != '2d'
    is_export = export_path is not None
    if not is_export and "agg" in plt.get_backend().lower():
        try:
            plt.switch_backend("TkAgg")
        except Exception:
            pass
    if is_3d:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(9, 12))
    fig.subplots_adjust(bottom=0.04 if is_export else 0.16)
    fig.canvas.manager.set_window_title(f"AnalyzeRL Replay: {payload.get('game_id', '')}")
    ax.set_xlim(-SIDE_WALL_X, SIDE_WALL_X)
    ax.set_ylim(-BACK_NET_Y, BACK_NET_Y)
    if is_3d:
        ax.set_zlim(0, CEILING_Z)
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.set_box_aspect((SIDE_WALL_X * 2, BACK_NET_Y * 2, CEILING_Z * 2.5))
        ax.xaxis.pane.set_facecolor('#20252b')
        ax.yaxis.pane.set_facecolor('#20252b')
        ax.zaxis.pane.set_facecolor('#20252b')
        ax.grid(False)
    else:
        ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#20252b')
    fig.patch.set_facecolor('#15181d')
    ax.set_xticks([])
    ax.set_yticks([])
    if is_3d:
        ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    if is_3d:
        ax.set_zlabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)

    if is_3d:
        draw_field_3d(ax)
        draw_boost_pads_3d(ax)
    else:
        draw_field(ax)
        draw_boost_pads_2d(ax)

    if is_3d:
        ball_artist = ax.scatter([], [], [], s=95, c='#f5f1df', edgecolors='#111111', zorder=5)
        blue_artist = ax.scatter([], [], [], s=90, c='#4aa3ff', edgecolors='#101820', zorder=4)
        orange_artist = ax.scatter([], [], [], s=90, c='#f17d3f', edgecolors='#101820', zorder=4)
    else:
        ball_artist = ax.scatter([], [], s=95, c='#f5f1df', edgecolors='#111111', zorder=5)
        blue_artist = ax.scatter([], [], s=90, c='#4aa3ff', edgecolors='#101820', zorder=4)
        orange_artist = ax.scatter([], [], s=90, c='#f17d3f', edgecolors='#101820', zorder=4)
    if is_3d:
        title_text = ax.text2D(0.01, 0.99, '', transform=ax.transAxes, va='top', ha='left',
                               color='#f4f7fb', fontsize=11)
        event_text = ax.text2D(0.5, 0.04, '', transform=ax.transAxes, va='bottom', ha='center',
                               color='#f4f7fb', fontsize=11,
                               bbox={'facecolor': '#15181d', 'edgecolor': '#8792a2', 'alpha': 0.85, 'pad': 8})
    else:
        title_text = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top', ha='left',
                             color='#f4f7fb', fontsize=11)
        event_text = ax.text(0.5, 0.04, '', transform=ax.transAxes, va='bottom', ha='center',
                             color='#f4f7fb', fontsize=11,
                             bbox={'facecolor': '#15181d', 'edgecolor': '#8792a2', 'alpha': 0.85, 'pad': 8})
    player_labels = []
    state = {
        'frame_idx': 0,
        'playing': not is_export,
        'base_interval': int(round(1000 / ANIMATION_FRAMES_PER_SECOND)),
        'play_start_time': None,
        'play_start_idx': 0,
        'updating_slider': False,
    }
    frame_slider = None
    prev_button = None
    play_button = None
    next_button = None
    if not is_export:
        frame_slider_ax = fig.add_axes([0.14, 0.065, 0.72, 0.025], facecolor='#20252b')
        prev_ax = fig.add_axes([0.43, 0.025, 0.08, 0.04])
        play_ax = fig.add_axes([0.53, 0.025, 0.10, 0.04])
        next_ax = fig.add_axes([0.65, 0.025, 0.08, 0.04])

        frame_slider = Slider(
            frame_slider_ax,
            'Frame',
            0,
            max(len(frames) - 1, 0),
            valinit=0,
            valstep=1,
            color='#4aa3ff',
        )
        prev_button = Button(prev_ax, '<')
        play_button = Button(play_ax, 'Pause')
        next_button = Button(next_ax, '>')

        for widget_ax in [frame_slider_ax, prev_ax, play_ax, next_ax]:
            widget_ax.tick_params(colors='#d9dee7')
            for spine in widget_ax.spines.values():
                spine.set_color('#8792a2')

    def set_offsets(artist, points):
        if is_3d:
            if points:
                array = np.asarray(points, dtype=np.float64)
                artist._offsets3d = (array[:, 0], array[:, 1], array[:, 2])
            else:
                artist._offsets3d = ([], [], [])
        elif points:
            artist.set_offsets(np.asarray(points, dtype=np.float64))
        else:
            artist.set_offsets(np.empty((0, 2)))

    def visible_events(frame_number):
        if not event_frames.size:
            return []
        lower_frame = frame_number - event_window_frames
        hit_frames = event_frames[(event_frames >= lower_frame) & (event_frames <= frame_number)]
        output = []
        for hit_frame in hit_frames:
            output.extend((int(hit_frame), row) for row in events_by_frame[int(hit_frame)])
        return output[-4:]

    def draw_frame(frame_idx):
        nonlocal player_labels
        frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))
        state['frame_idx'] = frame_idx
        frame = frames[frame_idx]
        frame_number = int(frame['frame_number'])
        seconds = parse_number(frame.get('seconds_elapsed'))
        while player_labels:
            player_labels.pop().remove()

        ball_pos = frame_position(frame, 'ball')
        if is_3d:
            set_offsets(ball_artist, [ball_pos] if ball_pos else [])
        else:
            set_offsets(ball_artist, [(ball_pos[0], ball_pos[1])] if ball_pos else [])

        blue_points = []
        orange_points = []
        for player in frame.get('players', []):
            if player_is_demoed(player, frame_number, seconds, demo_windows):
                continue
            pos = player.get('pos')
            if not pos or len(pos) < 3:
                continue
            point = (float(pos[0]), float(pos[1]), float(pos[2])) if is_3d else (float(pos[0]), float(pos[1]))
            if player.get('team') == 'orange':
                orange_points.append(point)
            else:
                blue_points.append(point)
            name = player.get('name', '')
            boost = kickoff_boost_value(frame_number, boost_amount(player.get('boost')), kickoff_windows)
            if is_3d:
                player_labels.append(
                    ax.text(point[0], point[1], point[2] + 130, name, color='#f4f7fb',
                            fontsize=7, ha='center', va='bottom', zorder=6)
                )
                if boost is not None:
                    player_labels.append(
                        ax.text(point[0], point[1] - 170, point[2] + 35, str(boost), color='#f4f7fb',
                                fontsize=7, ha='center', va='top', zorder=6)
                    )
            else:
                player_labels.append(
                    ax.text(point[0], point[1] + 130, name, color='#f4f7fb',
                            fontsize=7, ha='center', va='bottom', zorder=6)
                )
                if boost is not None:
                    player_labels.append(
                        ax.text(point[0], point[1] - 130, str(boost), color='#f4f7fb',
                                fontsize=7, ha='center', va='top', zorder=6)
                    )
        set_offsets(blue_artist, blue_points)
        set_offsets(orange_artist, orange_points)

        event_text.set_text('\n'.join(event_label(row) for _, row in visible_events(frame_number)))

        clock = '' if not math.isfinite(seconds) else f'{seconds:0.1f}s'
        blue_score, orange_score = score_at_frame(score_timeline, frame_number)
        title_text.set_text(
            f"{replay_title(payload)}\n{team_score_title(payload, blue_score, orange_score)} | frame {frame_number} | {clock}"
        )
        if frame_slider is not None and not state['updating_slider']:
            state['updating_slider'] = True
            frame_slider.set_val(frame_idx)
            state['updating_slider'] = False
        if not is_export:
            fig.canvas.draw_idle()
        return [ball_artist, blue_artist, orange_artist, title_text, event_text, *player_labels]

    def on_timer():
        if state['playing']:
            now = time.perf_counter()
            if state['play_start_time'] is None:
                state['play_start_time'] = now
                state['play_start_idx'] = state['frame_idx']
                return True
            elapsed = max(now - state['play_start_time'], 0.0)
            next_idx = state['play_start_idx'] + int(elapsed * ANIMATION_FRAMES_PER_SECOND)
            if next_idx <= state['frame_idx']:
                return True
            if next_idx >= len(frames):
                state['playing'] = False
                play_button.label.set_text('Play')
                next_idx = len(frames) - 1
            draw_frame(next_idx)
        return True

    def on_frame_slider(value):
        if state['updating_slider']:
            return
        state['playing'] = False
        state['play_start_time'] = None
        play_button.label.set_text('Play')
        draw_frame(int(value))

    def on_play(_event):
        state['playing'] = not state['playing']
        state['play_start_time'] = time.perf_counter()
        state['play_start_idx'] = state['frame_idx']
        play_button.label.set_text('Pause' if state['playing'] else 'Play')

    def on_prev(_event):
        state['playing'] = False
        state['play_start_time'] = None
        play_button.label.set_text('Play')
        draw_frame(state['frame_idx'] - 1)

    def on_next(_event):
        state['playing'] = False
        state['play_start_time'] = None
        play_button.label.set_text('Play')
        draw_frame(state['frame_idx'] + 1)

    def on_key(event):
        if event.key == ' ':
            on_play(event)
        elif event.key in ['left', 'a']:
            on_prev(event)
        elif event.key in ['right', 'd']:
            on_next(event)

    draw_frame(0)
    if not is_export:
        state['play_start_time'] = time.perf_counter()
        state['play_start_idx'] = 0
    if is_export:
        if is_3d:
            ax.view_init(elev=view_elev, azim=view_azim)
        output_folder = os.path.dirname(os.path.abspath(export_path))
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        extension = os.path.splitext(export_path)[1].lower()
        if extension == '.gif':
            writer = animation.PillowWriter(fps=ANIMATION_FRAMES_PER_SECOND)
        elif extension == '.mp4':
            writer = animation.FFMpegWriter(fps=ANIMATION_FRAMES_PER_SECOND, bitrate=2400)
        else:
            raise ValueError('Export path must end in .gif or .mp4')
        print(f'Exporting {len(frames):,} {render_mode} rendered frames to {export_path}', flush=True)
        progress_step = max(len(frames) // 10, 1)
        with writer.saving(fig, export_path, dpi=140):
            for frame_idx in range(len(frames)):
                if is_3d:
                    ax.view_init(elev=view_elev, azim=view_azim)
                draw_frame(frame_idx)
                writer.grab_frame()
                if frame_idx == len(frames) - 1 or (frame_idx + 1) % progress_step == 0:
                    print(f'Exported {frame_idx + 1:,}/{len(frames):,} frames', flush=True)
        plt.close(fig)
        return export_path

    frame_slider.on_changed(on_frame_slider)
    play_button.on_clicked(on_play)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    fig.canvas.mpl_connect('key_press_event', on_key)

    timer = fig.canvas.new_timer(interval=5)
    timer.add_callback(on_timer)
    timer.start()
    plt.show()
    return timer
