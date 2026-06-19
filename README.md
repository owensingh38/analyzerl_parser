# AnalyzeRL Parser

AnalyzeRL Parser is a Python package for parsing Rocket League replays into play-by-play and frame-level data built on the powerful Rust library [boxcars](https://github.com/nickbabcock/boxcars).

## Installation

```powershell
python -m pip install analyzerl_parser
```

The package requires Python 3.10 or newer. Install optional dataframe libraries for the return type you want to use:

```powershell
python -m pip install polars pandas
```

## Quick start

Parse a folder of replays to play-by-play Parquet files:

```python
import analyzerl_parser

pbp_files = analyzerl_parser.parse_replay(
    replay_path="data/replays",
    export="data/pbp",
    output="pbp",
    export_format="parquet",
    return_type="export",
    workers=4,
)
```

Parse a small sample and return one combined Polars dataframe:

```python
pbp = analyzerl_parser.parse_replay(
    replay_path="data/replays",
    export="data/pbp",
    output="pbp",
    export_format="parquet",
    return_type="polars",
    workers=1,
    limit=1,
)
```

Calculate player stats from replay files, PBP exports, frame exports, or already-loaded tabular data:

```python
stats = analyzerl_parser.calculate_stats(
    frames="data/pbp",
    return_type="polars",
)
```

## Python API

`parse_replay(replay_path="data/replays", export="data/frames", workers=4, return_type="export", output="frames", export_format=None, force=False, limit=None, xg_model_path=None)` parses one replay, a folder of replays, or a sequence of replay files and folders. `output` is `"frames"` or `"pbp"`. `export_format` is `"csv"` or `"parquet"`; when omitted, frames default to Parquet and PBP defaults to CSV. `return_type` is `"export"`, `"polars"`, or `"pandas"`. `limit` caps the number of replay inputs used from folders, and `force=True` overwrites existing exports. Set `xg_model_path` to an AnalyzeRL xG model file or folder when parsing PBP or frame output to add an `xG` column to shot and goal events.

`calculate_stats(frames, return_type="polars", export=None, group_by=None, rates=False, workers=4, parse_export="data/frames", force=False, limit=None, xg_model_path=None)` aggregates replay stats. The `frames` argument can be a replay file, folder of replays, one or more replay paths, a folder of PBP files, one or more PBP files, a folder of frame files, one or more frame files, or parsed tabular data. Replay inputs are parsed to PBP Parquet before stats are calculated. `group_by` defaults to `["replay_id", "player_id"]` and can be any columns in the stats output, such as `["player_id"]`, `["team"]`, or `["replay_id", "team"]`. Set `rates=True` to add per-five-minute and per-game rate columns using normalized time on field and games played. Set `xg_model_path` to an AnalyzeRL xG model file or folder to score source rows before expected-goal stats are aggregated. `return_type` is `"polars"`, `"pandas"`, or `"list"`. When `export` is provided, the output format is inferred from the path suffix, which must be `.csv` or `.parquet`.

`animate_replay(replay_path, event_window_frames=45, event_types=None, start_frame=None, end_frame=None, parser_path=None, render_mode="3d", export_path=None, view_elev=28, view_azim=-64, xg_model_path=None)` renders an interactive replay view or exports a GIF or MP4 at 30 fps and 1x replay speed. `render_mode` is `"2d"` or `"3d"`. `event_types` accepts a comma-separated filter such as `"shot,goal,demo"`, and `xg_model_path` can point at a saved AnalyzeRL xG model so shot and goal labels include xG.

## Exports

PBP exports contain one row per detected event and include `event_type`, replay identity, frame timing, player and team identity, ball and car state, snf event-specific fields for shot and goal rows. Frame exports contain analyzed all of this information in addition to extra rows for all frames in the match.

Export filenames are keyed by the input replay name:

```text
{replay_name}_pbp.csv
{replay_name}_pbp.parquet
{replay_name}_frames.csv
{replay_name}_frames.parquet
```

## Stats

By default, `calculate_stats` returns one row per player per replay. It includes core scoring, shooting, assists, shot assists, expected goals, expected assists, saves, touches, passes, turnovers, challenges, kickoffs, demos, bumps, entries, exits, retrievals, dribbles, flicks, flip resets, boost pickups, and boost totals when the source data contains those fields.

For and against columns are limited to goals, shots, expected goals, entries, exits, and demos. Other events are exposed as direct player counts or received/taken counts where that direction is part of the event itself.

## PBP event types

The parser tags frames with `event_type` as a labelling events which frequently occur in Rocket League matches.  Some labels are created directly from first-pass detection, while others are promoted during post-processing.

### `touch`

`touch` is a ball touch that is not upgraded to `kickoff`, `pass`, `shot`, `goal`, `turnover`, or `challenge`. In the parser, every detected ball-hit candidate starts as `touch` and remains `touch` unless later logic reclassifies it.

### `turnover`

`turnover` is a touch after which the next eligible touch event belongs to the other team, before any goal, kickoff, or challenge interrupts the sequence. In post-processing, the parser scans forward from each `touch`, and if the next subsequent touch-like event has a different `event_team`, that row is relabeled to `turnover`.

### `pass`

`pass` is a touch followed by the next touch from a different teammate in the same goal sequence, subject to duplicate suppression. In the parser, if adjacent ball events in the same `goal_number` belong to the same team and different players, the first touch is marked `pass`, `event_player_2_*` is set to the receiver, and repeat passer-receiver pairs are suppressed within 20 frames.

### `shot`

`shot` is a touch whose ball trajectory projects toward the opponent goal within the shot window, or a touch that is later credited as a save event target but not a goal. In the parser, `shot` is set when `is_shot(event, players)` is true or when the event is already marked `goal`, and official stat reconciliation can later preserve or demote shot credit.

### `goal`

`goal` is the final credited attacking touch for a scored goal. In the parser, for each header goal frame, it finds the latest touch by the scorer within 120 frames before the goal and marks that touch as `goal`.

### `save`

`save` is a defensive touch immediately following an opponent shot that did not become a goal. In the parser, if event `n - 1` is `shot`, not `goal`, and event `n` is touched by the opposing team, event `n` is marked with save credit, which usually lives on the defending touch row, though the parser can also synthesize a standalone `save` row when official save credit needs its own reconciled event frame.

### `kickoff`

`kickoff` is the first ball touch after a kickoff start. In the parser, kickoff windows are detected and the first qualifying touch after each kickoff start is relabeled to `kickoff`.

### `challenge`

`challenge` is a contested touch or car contact where an opposing player is close enough to the ball and toucher to qualify as a challenge, with duplicate suppression and winner assignment. In the parser, it can be created either by promoting a `bump` between opponents near the ball or by promoting a `touch` when an opposing player is within `CHALLENGE_TOUCH_BALL_DISTANCE` of the ball and within `CHALLENGE_TOUCH_PLAYER_DISTANCE` of the toucher, after which `event_player_2_*` is assigned to the challenger, player order may be swapped so `event_player_1_*` is the side that wins the next touch, and duplicate challenge pairs are suppressed inside the cooldown window.

### `shadow`

`shadow` is an off-ball defensive positioning event where a defender is between an opposing ball carrier and the defender's own net while the carrier is moving toward that net. In the parser, it is emitted from frame state when the carrier is the closest possessor, the defender is an opponent at a controlled following distance and lateral alignment, the carrier or ball is moving toward the defender's net, the situation is not close enough to qualify as a challenge, and the defender-carrier pair is outside the shadow cooldown; `event_player_1_*` is the shadowing defender and `event_player_2_*` is the ball carrier.

### `press`

`press` is an off-ball pressure event where an attacker hovers near an opposing ball carrier in the carrier's defensive end without entering challenge range. In the parser, it is emitted from frame state when the carrier is the closest possessor in their own defensive third, an opposing player is close to both the carrier and ball but not close enough to qualify as a challenge, and the defender-carrier pair is outside the press cooldown; `event_player_1_*` is the pressing player and `event_player_2_*` is the ball carrier.

### `bump`

`bump` is a non-demo car-to-car contact event that is not retained as a challenge after challenge promotion and deduplication. In the parser, when two cars are within `CAR_CONTACT_DISTANCE`, it emits `bump` with contact distance and relative speed unless the contact is near a demo frame or inside the bump cooldown, and some of these rows are later promoted to `challenge`.

### `demo`

`demo` is a demolition event with a feature-aligned contact frame. In the parser, official demo events are backtracked to the nearest qualifying car-contact frame within the demo window, and the resulting row is emitted as `demo`.

### `respawn`

`respawn` is a player returning to the field after being demolished. In the parser, each demo victim receives a `respawn` row at the first available frame at or after the standard demo respawn delay, with `event_player_1_*` set to the respawning player so downstream tools can model temporary demo off-field intervals.

### `game-join`

`game-join` is a player becoming active on the field for a team. In the parser, it is emitted when the player's team assignment becomes active in the network stream, with an inferred initial join added at the first frame where the player has car state if the explicit assignment was already active before the observed stream.

### `game-leave`

`game-leave` is a player leaving active field play for their team. In the parser, it is emitted when the player's team assignment becomes inactive in the network stream, and downstream stats use it to stop assigning time-on-field and team for/against context until the player joins again.

### `entry`

`entry` is an attacking team gaining the offensive third. In the parser, it can be emitted either from a ball-zone crossing where the ball moves from neutral or defensive space into that team's offensive third, or from a touch-derived entry where a player touch outside the offensive third is followed by the next ball state entering that team's offensive third, or the touched ball is already moving into it. Rows carry `controlled = true` when the event frame has a same-team closest possessor or `false` when the parser instead attributes the event from the latest same-team touch.

### `exit`

`exit` is a team clearing the ball out of its defensive third. In the parser, it can be emitted either from a ball-zone crossing where the ball leaves that team's defensive third or from a touch-derived exit where a player touch occurs in that team's defensive third and the next ball state is no longer in the defensive third, or the touched ball is already moving out. This is separate from the first-pass `clear` boolean used during touch classification even though both describe defensive relief.

### `retrieval`

`retrieval` is a team regaining close possession of a free ball after a loose interval. In the parser, while scanning frame states, if there was no prior possessor and a possessor appears after at least 90 frames since the last retrieval, it emits `retrieval` for that player and team.

### `boost-pickup`

`boost-pickup` is a boost gain event inferred from boost amount changes or boost collect state. In the parser, current and prior boost values are compared per player, and it emits `boost-pickup` with `boost_pickup_amount` and `boost_pickup_type` (`small`, `big`, or `reset`) when the inferred pickup threshold is met.

### `flip-reset`

`flip-reset` is a dodge refresh event detected while airborne after a player obtains a reset from contact with either the ball or another car. In the parser, it detects a reset in dodge or double-jump air counters, or an increase in `dodges_refreshed_counter`, then requires the car to be above `FLIP_RESET_MIN_CAR_Z`, enough time since the last reset event, and qualifying underside contact at that frame before emitting `flip-reset`; the row is labeled with `reset_origin`, which is `"ball"`, `"opponent"`, or `"teammate"` depending on the source of the reset contact.

## Documentation

Full package documentation can be viewed [here](https://owensingh38.github.io/analyzerl_parser).
