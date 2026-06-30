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

Export frame-by-frame state without event tagging:

```python
frames = analyzerl_parser.parse_replay(
    replay_path="data/replays",
    export="data/frames",
    output="frames-only",
    export_format="parquet",
    return_type="export",
)
```

Calculate player stats from replay files, PBP exports, frame exports, or already-loaded tabular data:

```python
stats = analyzerl_parser.calculate_stats(
    frames="data/pbp",
    return_type="polars",
)
```

## Exports

PBP exports contain one row per detected event and include `event_type`, replay identity, frame timing, player and team identity, ball and car state, and event-specific fields for shot and goal rows. Event rows include `event_length`, the seconds until the next event row, and `event_duration`, the intrinsic duration of an event when that event has its own span. Time-on-field and event-interval stats use `event_length`; `event_duration` can overlap other events and is not used as the between-event interval. Frame exports contain all of this information in addition to extra rows for all frames in the match. Frame-only exports contain frame-by-frame car, ball, and spatial state without event tagging or event overlays.

Row exports avoid per-row JSON payload columns. Metadata, player, actor, and attribute exports keep scalar fields only; full replay JSON is written only when the explicit JSON export mode is requested.

Each player slot includes a team-relative `rotation_role` column, such as `blue_player_1_rotation_role`. The role is `1` for the player closest to the ball on their own team, `2` for second closest, and so on through the team size. Empty player or ball state leaves the value null.

Export filenames are keyed by the input replay name:

```text
{replay_name}_pbp.csv
{replay_name}_pbp.parquet
{replay_name}_frames.csv
{replay_name}_frames.parquet
```

## PBP event types

The parser tags frames with `event_type` labels for events that frequently occur in Rocket League matches. The rules below describe the implementation order used by the Rust parser: ball-hit candidates are classified first, official replay stat credits are reconciled, frame-derived events are added, then post-processing can promote or relabel rows.

### Shared thresholds

The event rules use these constants:

```text
ball shot projection window: 3.0 seconds
goal credited touch window: 120 frames before the goal frame
pass duplicate window: 20 frames per passer/receiver pair
missed pass projection window: 2.5 seconds
missed pass target radius: 450 uu
missed pass maximum target miss: 1800 uu
missed pass minimum ball speed: 900 uu/s
missed pass minimum forward dot: 0.35
clear buffer: 400 uu around the defensive third line
field thirds: +/- 5120 / 3 uu on the y axis
car contact distance: 225 uu
car contact cooldown: 15 frames per car pair
demo respawn delay: 90 frames
demo contact search window: 90 frames before the demo event
demo duplicate cooldown: 150 frames
challenge ball distance: 425 uu
challenge player distance: 750 uu
challenge duplicate cooldown: 30 frames per player pair
press carrier distance: 900 uu
press ball distance: 900 uu
press duplicate cooldown: 60 frames per defender/carrier pair
shadow carrier distance: 500-1800 uu
shadow lateral distance: 1400 uu
shadow minimum speed toward net: 250 uu/s
shadow duplicate cooldown: 90 frames per defender/carrier pair
retrieval duplicate cooldown: 90 frames
zone event duplicate cooldown: 60 frames per event/team
whiff ball distance: 285 uu current, 520 uu previous
whiff pass-by corridor: 145 uu
whiff speed toward ball: 1150 uu/s, or 800 uu/s with boost/dodge/jump/flip input
whiff touch exclusion: any touch within 1 frame, or same-player touch within 10 frames
whiff direct-touch exclusion: same-player next touch within 90 frames
whiff next-touch requirement: another player touches within 120 frames
whiff duplicate cooldown: 150 frames per player
fake possession distance: 560 uu
fake defender ball distance: 780 uu
fake defender minimum speed: 575 uu/s
fake defender speed toward ball: 175 uu/s
fake ball speed drop: 180 uu/s
fake ball velocity change: 260 uu/s
fake ball direction dot threshold: 0.94
fake duplicate cooldown: 90 frames per carrier/defender pair
double commit ball distance: 1100 uu
double commit teammate distance: 1300 uu
double commit duplicate cooldown: 45 frames per teammate pair
rotation transition minimum run: 0.5 seconds
rotation stalled first-man run: 1.5 seconds
air/ground dribble window: 3.0 seconds
flick window: 1.0 second
hood-dribble horizontal distance: 180 uu
hood-dribble vertical separation: 70-260 uu
flip-reset contact distance: 230 uu
flip-reset minimum car z: 120 uu
flip-reset duplicate cooldown: 30 frames
double tap window: 5.0 seconds
double tap offensive back-wall distance: 900 uu for the ball, 700 uu for the car
double tap back-wall projection window: 3.5 seconds
shot context windows: rebound 3.0s, off demo 2.0s, off bump 2.0s, off kickoff 5.0s, off challenge/pass/fake/whiff 5.0s, off flip reset 2.0s
```

### `touch`

Every clustered ball-hit candidate starts as `touch`. It remains `touch` only if it is not later reclassified as `kickoff`, `goal`, `shot`, `missed-shot`, `missed-pass`, `exit`, `pass`, `challenge`, or `turnover`, and if official stat reconciliation does not promote or demote it.

### `turnover`

After challenge processing, each remaining `touch` scans forward until it reaches a `goal`, `kickoff`, or `challenge`. If the first later touch-like row (`touch`, `turnover`, `pass`, `shot`, `goal`, `kickoff`, or `challenge`) belongs to the other team, the original row is relabeled to `turnover`.

### `pass`

During ball-hit classification, a row is marked `pass` when the next ball event is in the same `goal_number`, is by a different player on the same team, and the same passer/receiver pair has not already been marked within 20 frames. The passer is `event_player_1_*`; the receiver is `event_player_2_*`.

### `shot`

During ball-hit classification, a row is marked `shot` when the ball is traveling toward the opponent goal, reaches the goal plane in 0.0-3.0 seconds, projects within the posts (`abs(x) <= 892.755`), and projects above ground but no higher than the crossbar (`0 < z <= 642.775`) after gravity. A row is also initially treated as shot-like when it is a credited save target. Official replay shot credits can later promote a nearby candidate to `shot` or synthesize a standalone official `shot` row when no candidate can be matched.

### `missed-shot`

During ball-hit classification, a row is marked `missed-shot` when it is not a `shot` or `goal`, but the ball is traveling toward the opponent goal, reaches the goal plane in 0.0-3.0 seconds, and misses the goal frame. Misses include projected positions outside the posts or above the crossbar, bounded to plausible attempts by `z <= 2044` and `abs(x) <= 3392.755` at the goal plane. `missed-shot` rows are included in `shot_attempts` and xG scoring, but not in official `shots`.

### `missed-pass`

During ball-hit classification, a row is marked `missed-pass` when it is not a completed `pass`, `shot`, `goal`, or `missed-shot`, the ball leaves the touch at least 900 uu/s, and a same-team target is plausibly in the outgoing path. The target must be a teammate other than the passer, the ball velocity must point toward that teammate with dot product at least 0.35, and the sampled ball path over 2.5 seconds must miss the target by more than 450 uu but no more than 1800 uu. Gravity is applied to the sampled ball path. The target teammate is written to `event_player_2_*`. `missed-pass` rows count in `missed_passes` but not in `passes`, `shot_attempts`, or xG.

### `goal`

For each replay-header goal, the parser finds the latest ball-hit candidate by the scorer at or before the goal frame and within 120 frames of that goal frame. That row is marked `goal`. If the scorer has an earlier qualifying `pass` in the same sequence, that passer is copied to `event_player_2_*` and the pass row receives assist credit.

### `save`

During ball-hit classification, when the previous ball event is a `shot`, is not a `goal`, and the current ball event belongs to the other team, the current event receives save credit. The parser stores the defender in `event_player_3_*`, points `event_player_1_*` back to the prior shooter, and initially labels the row `shot`; official save reconciliation can later mark the defending row with `official_save` or synthesize a standalone `save` row when no suitable candidate exists.

### `kickoff`

The first ball-hit candidate selected by the kickoff detector after each kickoff start is relabeled to `kickoff`. Zone events ignore kickoff frames, and post-goal frames are ignored until the next kickoff.

### `challenge`

`challenge` is created in two post-processing paths. First, an opponent `bump` is promoted when either car is within 425 uu of the ball. Second, a `touch` is promoted when the nearest opponent is within 425 uu of the ball and within 750 uu of the toucher. In both paths, the next touch-like row determines the winner: if the opponent's team wins the next touch, player order is swapped so `event_player_1_*` is the winner. Duplicate challenge pairs within 30 frames are reverted to `bump` or `touch`.

### `whiff`

For each consecutive frame pair and player, `whiff` is emitted only after an obvious ball attempt and a true near miss. The player must be within 285 uu of the ball in the current frame or have been within 520 uu in the previous frame, must be moving toward the ball at least 1150 uu/s, or at least 800 uu/s while boost, dodge, jump, double-jump, or flip state is active, and the player/ball paths must show that one went past or around the other. The path test passes when the relative player-ball vector crosses sides within 285 uu, or when the player path or ball path passes within 145 uu after the player and ball are moving apart from their closest approach. The row is skipped if any touch occurs within 1 frame, if that same player touches within 10 frames, if the same player is the next direct ball toucher within 90 frames, if no other player touches the ball within 120 frames, or if the same player already had a whiff within 150 frames.

### `fake`

For each `whiff`, the parser looks for an opposing player in possession of the ball as a ground dribble or air dribble. When possession is present, the whiff row is upgraded to `fake`. `event_player_1_*` remains the player who missed, while `event_player_2_*` is the possessor credited with the fake.

### `double-commit`

For each ball-contact row (`touch`, `turnover`, `pass`, `shot`, `missed-shot`, `missed-pass`, `goal`, `kickoff`, or `challenge`), the parser looks for the nearest teammate whose distance to the ball is at most 1100 uu and whose distance to the contact player is at most 1300 uu. Both players must pass the same intent validation used by `whiff`: moving toward the ball and either committed by speed or supported by active play inputs, within the double-commit ball-distance window. If found, it clones the row as `double-commit`, clears official stat flags, and suppresses the same unordered teammate pair for 45 frames. `event_player_1_*` is the responsible player whose path is less natural or higher-resistance based on distance, approach angle, rotation role, and speed toward the ball; `event_player_2_*` is the other committing player.

### `rotation-fill`, `rotation-cut`, and `rotation-stall`

Rotation events are emitted from per-frame `rotation_role` changes and are enabled by default. Pass `rotation_events=False` to `parse_replay(...)`, or `--no-rotation-events` to the Rust CLI, to disable them. A player must hold the previous role for at least 0.5 seconds before a role change can create `rotation-fill` or `rotation-cut`. `rotation-fill` marks normal advancement through the team rotation, including first man rotating to last man. `rotation-cut` marks skipping ahead in the rotation or first man rotating to a non-last-man role. `rotation-stall` fires once when a player remains first man for at least 1.5 seconds. Rotation rows set `event_player_1_*` to the rotating player, store the run length in `event_duration`, and carry `rotation_number` for that team.

### `shadow`

For each frame, the closest possessor is treated as the carrier. For each opponent, `shadow` is emitted when the opponent is not challenge-like, is 500-1800 uu from the carrier, is within 1400 uu laterally on x, is between the carrier and the opponent's own net on y, and the ball or carrier is moving toward that net at least 250 uu/s. The same defender/carrier pair is suppressed for 90 frames.

### `press`

For each frame, the closest possessor is treated as the carrier. For each opponent, `press` is emitted when the carrier is in their own defensive third, the opponent is within 900 uu of the carrier and 900 uu of the ball, and the pair is not challenge-like (`distance_to_carrier <= 750` and `distance_to_ball <= 425`). The same defender/carrier pair is suppressed for 60 frames.

### `bump`

For each frame and car pair, `bump` is emitted when both cars have position and are within 225 uu. The same unordered pair is suppressed for 15 frames. Contacts within 30 frames of a matching demo pair are skipped. `event_player_1_*` is the faster car, `event_player_2_*` is the other car, and later challenge logic can promote some opponent bumps to `challenge`.

### `demo`

`demo` comes from replay demolition data. The parser backtracks from the demo to the first prior frame within 90 frames where the two cars are within 225 uu, records that contact frame, and suppresses duplicate demo pairs for 150 frames. The row includes car-contact distance, relative speed, each player's speed, and demolished flags.

### `respawn`

For each demo victim, `respawn` is emitted at the first available frame at or after 90 frames after the demo, with `event_player_1_*` set to the respawning player.

### `game-join`

`game-join` is emitted when a player's team assignment becomes active in the network stream. If a player already has car state before an explicit join is observed, the parser adds an inferred initial `game-join` at the first frame where the player has car state.

### `game-leave`

`game-leave` is emitted when a player's team assignment becomes inactive in the network stream. Downstream stats use it to stop assigning time-on-field and team for/against context until the next `game-join`.

### `entry`

`entry` uses field thirds at `y = +/- 5120 / 3`. A frame-zone `entry` is emitted when the ball crosses into blue's offensive third (`zone == 1`) or orange's offensive third (`zone == -1`) from any other zone, after the first kickoff and not between a goal and the next kickoff. A touch-derived `entry` is emitted when a non-kickoff, non-exit touch is outside the offensive third and the next same-goal-number ball event enters the offensive third, or the touched ball is already moving toward the opponent half. Duplicate `entry` rows for the same team are suppressed for 60 frames. `controlled = true` when the closest possessor at the event frame is the credited player.

### `exit`

`exit` uses field thirds at `y = +/- 5120 / 3`. A frame-zone `exit` is emitted when the ball leaves blue's defensive third (`previous_zone == -1`) or orange's defensive third (`previous_zone == 1`), after the first kickoff and not between a goal and the next kickoff. A touch-derived `exit` is emitted when a non-kickoff, non-exit touch occurs in the team's defensive third and the next same-goal-number ball event is outside that defensive third, or the touched ball is already moving toward the opponent half. Duplicate `exit` rows for the same team are suppressed for 60 frames.

### `retrieval`

While scanning frame states after the first kickoff and outside post-goal dead time, `retrieval` is emitted when the previous frame had no closest possessor, the current frame has a closest possessor, and at least 90 frames have passed since the prior retrieval. The closest possessor becomes `event_player_1_*` and `controlled = true`.

### `boost-pickup`

For each player, `boost-pickup` is emitted when normalized boost increases or the `boost_collect` grant value changes, more than 2 frames have passed since the player's previous pickup, and the inferred pickup amount is positive. The row stores `boost_pickup_amount` and `boost_pickup_type` (`small`, `big`, or `reset`).

### `flip-reset`

For each player, `flip-reset` is emitted when the dodge air counter resets from positive to zero, the double-jump air counter resets from positive to zero, or `dodges_refreshed_counter` increases; the car has position; car z is at least 120 uu; and the player's last reset was more than 30 frames earlier. The frame must also have qualifying reset contact within 230 uu, and the row stores `reset_origin` as `ball`, `opponent`, or `teammate`.

### `air-dribble`

`air-dribble` is synthesized from contact rows. For the same player, if the current contact-like event and the previous contact-like event are within 3.0 seconds and either row has `aerialing = true` (`player z >= 642.775`), the current row receives `air_dribble = true` and a cloned `air-dribble` event row is emitted.

### `ground-dribble`

`ground-dribble` is synthesized from contact rows. For the same player, if the current contact-like event and the previous contact-like event are within 3.0 seconds, both qualify as hood-dribble control, and the current row was not marked as an air dribble, the current row receives `ground_dribble = true` and a cloned `ground-dribble` event row is emitted. Hood-dribble control requires the player to be horizontally within 180 uu of the ball and vertically separated from the ball by 70-260 uu.

### `flick`

`flick` is synthesized from contact rows. If a row is already marked as `ground_dribble`, is within 1.0 second of that player's previous contact-like row, the player has flipped, and `ball_vel_z > 250`, the row receives `flick_shot = true` and a cloned `flick` event row is emitted.

## Documentation

Full package documentation can be viewed [here](https://owensingh38.github.io/analyzerl_parser).
