use crate::*;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug)]
struct StintWindow {
    number: i32,
    start_frame: i32,
    end_frame: i32,
}

#[derive(Clone, Debug)]
pub(crate) struct EventModel {
    stints: Vec<StintWindow>,
}

impl EventModel {
    pub(crate) fn from_frames(context: &PbpContext) -> Self {
        let kickoff_starts = kickoff_start_frames_from_resets(&context.frame_states);
        let goal_frames = context
            .ball_events
            .iter()
            .filter(|event| event.goal)
            .map(|event| event.frame_number)
            .collect::<Vec<_>>();
        let mut stints = Vec::new();

        for (idx, start_frame) in kickoff_starts.iter().copied().enumerate() {
            let next_start = kickoff_starts.get(idx + 1).copied().unwrap_or(i32::MAX);
            let start_frame = unfreeze_frame(&context.frame_states, start_frame, next_start)
                .unwrap_or(start_frame);
            let end_frame = goal_frames
                .iter()
                .copied()
                .filter(|frame| *frame >= start_frame && *frame < next_start)
                .min()
                .unwrap_or(next_start.saturating_sub(1));

            if end_frame >= start_frame {
                stints.push(StintWindow {
                    number: stints.len() as i32 + 1,
                    start_frame,
                    end_frame,
                });
            }
        }

        Self { stints }
    }

    pub(crate) fn stint_number_for_frame(&self, frame_number: i32) -> Option<i32> {
        self.stints
            .iter()
            .find(|stint| frame_number >= stint.start_frame && frame_number <= stint.end_frame)
            .map(|stint| stint.number)
    }

    pub(crate) fn finalize_rows(&self, rows: &mut Vec<PbpEventRecord>) {
        let kickoff_frame_by_stint = self.kickoff_frame_by_stint(rows);
        rows.retain(|row| {
            if matches!(
                row.event_type.as_str(),
                "demo" | "game-join" | "game-leave" | "respawn"
            ) {
                return true;
            }
            let Some(frame) = row_frame_number(row) else {
                return false;
            };
            let Some(stint_number) = self.stint_number_for_frame(frame) else {
                return false;
            };
            let Some(kickoff_frame) = kickoff_frame_by_stint.get(&stint_number).copied() else {
                return false;
            };
            row.event_type == "kickoff" || frame >= kickoff_frame
        });

        for row in rows.iter_mut() {
            if let Some(frame) = row_frame_number(row) {
                if let Some(stint_number) = self.stint_number_for_frame(frame) {
                    row.values.insert_i32("stint_number", stint_number);
                }
            }
        }
    }

    fn kickoff_frame_by_stint(&self, rows: &[PbpEventRecord]) -> HashMap<i32, i32> {
        let mut output = HashMap::new();
        for row in rows.iter().filter(|row| row.event_type == "kickoff") {
            let Some(frame) = row_frame_number(row) else {
                continue;
            };
            let Some(stint_number) = self.stint_number_for_frame(frame) else {
                continue;
            };
            output
                .entry(stint_number)
                .and_modify(|existing| {
                    if frame < *existing {
                        *existing = frame;
                    }
                })
                .or_insert(frame);
        }
        output
    }
}

fn row_frame_number(row: &PbpEventRecord) -> Option<i32> {
    row_i32(&row.values, "observed_frame_number")
        .or_else(|| row.frame_number)
        .or_else(|| row_i32(&row.values, "frame_number"))
}

fn unfreeze_frame(
    frames: &[FrameSnapshot],
    start_frame: i32,
    next_start_frame: i32,
) -> Option<i32> {
    frames
        .iter()
        .filter(|snapshot| {
            snapshot.frame_number >= start_frame && snapshot.frame_number < next_start_frame
        })
        .find(|snapshot| {
            snapshot.players.iter().flatten().any(|player| {
                player.entity.has_pos
                    && (vec_norm(player.entity.vel) > 50.0
                        || player.boost_active
                        || player.throttle.unwrap_or(0) != 0
                        || player.steer.unwrap_or(0) != 0)
            })
        })
        .map(|snapshot| snapshot.frame_number)
}

pub(crate) fn demo_feature_contact(
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

pub(crate) fn add_zone_events(
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
    frame_snapshot(context, event.frame_number)
        .and_then(|snapshot| {
            let ball = snapshot.ball?;
            closest_possessor(snapshot, &context.players, ball.pos)
        })
        .and_then(|idx| context.players.get(idx))
        .map(|player| player.name == event.player_name)
        .unwrap_or(false)
}

pub(crate) fn add_pressure_events(
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

pub(crate) fn add_whiff_events(
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
    let mut last_whiff_frame: HashMap<String, i32> = HashMap::new();
    let (all_touch_frames, player_touch_frames) = touch_frame_indexes(context);
    let mut touch_events: Vec<(i32, &str)> = context
        .ball_events
        .iter()
        .map(|event| (event.frame_number, event.player_name.as_str()))
        .collect();
    touch_events.sort_unstable_by_key(|(frame_number, _)| *frame_number);

    for pair in context.frame_states.windows(2) {
        let previous_snapshot = &pair[0];
        let snapshot = &pair[1];
        let ball = match snapshot.ball {
            Some(value) if value.has_pos => value,
            _ => continue,
        };
        let previous_ball = match previous_snapshot.ball {
            Some(value) if value.has_pos => value,
            _ => continue,
        };

        for (player_idx, player) in context.players.iter().enumerate() {
            let Some(player_state) = snapshot.players.get(player_idx).and_then(Option::as_ref)
            else {
                continue;
            };
            let Some(previous_player_state) = previous_snapshot
                .players
                .get(player_idx)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            if !player_state.entity.has_pos || !previous_player_state.entity.has_pos {
                continue;
            }
            let recent = last_whiff_frame
                .get(&player.name)
                .map(|frame| snapshot.frame_number - *frame <= WHIFF_COOLDOWN_FRAMES)
                .unwrap_or(false);
            if recent
                || whiff_has_near_touch_from_indexes(
                    &all_touch_frames,
                    &player_touch_frames,
                    snapshot.frame_number,
                    &player.name,
                )
                || next_direct_touch_by_player(
                    &touch_events,
                    snapshot.frame_number,
                    player.name.as_str(),
                )
            {
                continue;
            }

            let player_pos = player_state.entity.pos;
            let distance_to_ball = vec_distance(player_pos, ball.pos);
            if distance_to_ball > WHIFF_BALL_DISTANCE {
                continue;
            }

            let Some(speed_toward_ball) =
                speed_toward_point(player_state.entity.vel, player_pos, ball.pos)
            else {
                continue;
            };
            let attempt_input_active = player_state.boost_active
                || player_state.dodge_active
                || player_state.jump_active
                || player_state.double_jump_active
                || player_state.flipped;
            let committed = speed_toward_ball >= WHIFF_COMMITTED_SPEED_TOWARD_BALL
                || (speed_toward_ball >= WHIFF_MIN_SPEED_TOWARD_BALL && attempt_input_active);
            if !committed {
                continue;
            }
            if !whiff_like_miss(
                previous_player_state.entity.pos,
                player_pos,
                previous_ball.pos,
                ball.pos,
            ) {
                continue;
            }

            let mut row = build_zone_event_row(
                "whiff",
                snapshot.frame_number,
                player.team,
                Some(player.name.as_str()),
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
            row.values
                .insert("distance".to_string(), distance_to_ball.to_string());
            rows.push(row);
            last_whiff_frame.insert(player.name.clone(), snapshot.frame_number);
        }
    }
}

fn touch_frame_indexes(context: &PbpContext) -> (Vec<i32>, HashMap<String, Vec<i32>>) {
    let mut all_touch_frames = Vec::with_capacity(context.ball_events.len());
    let mut player_touch_frames: HashMap<String, Vec<i32>> = HashMap::new();
    for event in &context.ball_events {
        all_touch_frames.push(event.frame_number);
        player_touch_frames
            .entry(event.player_name.clone())
            .or_default()
            .push(event.frame_number);
    }
    all_touch_frames.sort_unstable();
    for frames in player_touch_frames.values_mut() {
        frames.sort_unstable();
    }
    (all_touch_frames, player_touch_frames)
}

fn whiff_has_near_touch_from_indexes(
    all_touch_frames: &[i32],
    player_touch_frames: &HashMap<String, Vec<i32>>,
    frame_number: i32,
    player_name: &str,
) -> bool {
    frame_near(
        all_touch_frames,
        frame_number,
        WHIFF_ANY_TOUCH_EXCLUSION_FRAMES,
    ) || player_touch_frames
        .get(player_name)
        .map(|frames| frame_near(frames, frame_number, WHIFF_TOUCH_EXCLUSION_FRAMES))
        .unwrap_or(false)
}

fn next_direct_touch_by_player(
    touch_events: &[(i32, &str)],
    frame_number: i32,
    player_name: &str,
) -> bool {
    let idx = touch_events.partition_point(|(touch_frame, _)| *touch_frame <= frame_number);
    touch_events
        .get(idx)
        .map(|(touch_frame, touch_player)| {
            *touch_frame - frame_number <= WHIFF_DIRECT_TOUCH_WINDOW_FRAMES
                && *touch_player == player_name
        })
        .unwrap_or(false)
}

fn frame_near(frames: &[i32], frame_number: i32, window_frames: i32) -> bool {
    if frames.is_empty() {
        return false;
    }
    let lower = frame_number - window_frames;
    let upper = frame_number + window_frames;
    let idx = frames.partition_point(|frame| *frame < lower);
    while idx < frames.len() && frames[idx] <= upper {
        return true;
    }
    false
}

pub(crate) fn add_fake_events(
    rows: &mut Vec<PbpEventRecord>,
    context: &PbpContext,
    _player_static_values: &[(String, String)],
    _game_id: &str,
    _match_guid: &str,
    _replay_name: &str,
    _map_id: &str,
    _team_size: Option<i32>,
    _game_time: &str,
) {
    for row in rows.iter_mut().filter(|row| row.event_type == "whiff") {
        let Some(frame_number) = row.frame_number else {
            continue;
        };
        let Some(whiff_player) = context
            .players
            .iter()
            .find(|player| player.name == row_string(&row.values, "event_player_1_name"))
        else {
            continue;
        };
        let Some(snapshot) = frame_snapshot(context, frame_number) else {
            continue;
        };
        let Some(ball) = snapshot.ball.filter(|state| state.has_pos) else {
            continue;
        };
        let Some(possessor_idx) = closest_possessor_within(
            snapshot,
            &context.players,
            ball.pos,
            FAKE_POSSESSION_DISTANCE,
        ) else {
            continue;
        };
        let Some(possessor) = context.players.get(possessor_idx) else {
            continue;
        };
        if possessor.team == whiff_player.team {
            continue;
        }

        row.event_type = "fake".to_string();
        row.values
            .insert("event_type".to_string(), "fake".to_string());
        add_event_player(&mut row.values, &context.players, 2, &possessor.name);
        row.values.insert(
            "event_team".to_string(),
            team_name(possessor.team).to_string(),
        );
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
        .filter(|row| ball_contact_event(&row.event_type))
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
    closest_possessor_within(snapshot, players, ball_pos, POSSESSION_DISTANCE)
}

fn closest_possessor_within(
    snapshot: &FrameSnapshot,
    players: &[PlayerInfo],
    ball_pos: Vec3,
    max_distance: f32,
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
            (distance <= max_distance).then_some((idx, distance))
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
    PbpEventRecord {
        frame_number: Some(frame_number),
        event_type: event_type.to_string(),
        values,
    }
}

pub(crate) fn add_car_contact_events(
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
                rows.push(PbpEventRecord {
                    frame_number: Some(snapshot.frame_number),
                    event_type: "bump".to_string(),
                    values,
                });
            }
        }
    }
}

pub(crate) fn add_boost_pickup_events(
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
        rows.push(PbpEventRecord {
            frame_number: Some(event.frame_number),
            event_type: "boost-pickup".to_string(),
            values,
        });
    }
}

pub(crate) fn add_flip_reset_events(
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
        rows.push(PbpEventRecord {
            frame_number: Some(event.frame_number),
            event_type: "flip-reset".to_string(),
            values,
        });
    }
}

#[derive(Clone, Debug, Default)]
struct RotationRunState {
    role: Option<i32>,
    start_seconds: f32,
    stalled_emitted: bool,
}

pub(crate) fn add_rotation_events(
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
    let Some(team_size) = team_size else {
        return;
    };
    if team_size <= 1 {
        return;
    }

    let mut states = vec![RotationRunState::default(); context.players.len()];
    let mut roles = vec![None; context.players.len()];
    let mut rotation_numbers: HashMap<i32, i32> = HashMap::new();
    for snapshot in &context.frame_states {
        let Some(ball) = snapshot
            .ball
            .filter(|state| state.has_pos)
            .map(|state| state.pos)
        else {
            continue;
        };
        let seconds = snapshot
            .seconds_elapsed
            .unwrap_or(snapshot.frame_number as f32 / 30.0);
        fill_rotation_roles_for_snapshot(&mut roles, snapshot, &context.players, ball);

        for (player_idx, player) in context.players.iter().enumerate() {
            let Some(role) = roles.get(player_idx).copied().flatten() else {
                continue;
            };
            let state = &mut states[player_idx];
            let Some(previous_role) = state.role else {
                state.role = Some(role);
                state.start_seconds = seconds;
                state.stalled_emitted = false;
                continue;
            };

            if previous_role != role {
                let previous_duration = (seconds - state.start_seconds).max(0.0);
                if previous_duration >= 0.5 {
                    let event_type = if rotation_filled(previous_role, role, team_size) {
                        Some("rotation-fill")
                    } else if rotation_cut(previous_role, role, team_size) {
                        Some("rotation-cut")
                    } else {
                        None
                    };
                    if let Some(event_type) = event_type {
                        let rotation_number = current_rotation_number(
                            &mut rotation_numbers,
                            player.team,
                            previous_role,
                            role,
                        );
                        rows.push(build_rotation_event_row(
                            event_type,
                            snapshot.frame_number,
                            previous_duration,
                            rotation_number,
                            player,
                            context,
                            player_static_values,
                            game_id,
                            match_guid,
                            replay_name,
                            map_id,
                            Some(team_size),
                            game_time,
                        ));
                    }
                }
                state.role = Some(role);
                state.start_seconds = seconds;
                state.stalled_emitted = false;
                continue;
            }

            if role == 1 && !state.stalled_emitted && seconds - state.start_seconds >= 1.5 {
                let rotation_number = current_rotation_number(
                    &mut rotation_numbers,
                    player.team,
                    previous_role,
                    role,
                );
                rows.push(build_rotation_event_row(
                    "rotation-stall",
                    snapshot.frame_number,
                    seconds - state.start_seconds,
                    rotation_number,
                    player,
                    context,
                    player_static_values,
                    game_id,
                    match_guid,
                    replay_name,
                    map_id,
                    Some(team_size),
                    game_time,
                ));
                state.stalled_emitted = true;
            }
        }
    }
}

fn fill_rotation_roles_for_snapshot(
    roles: &mut [Option<i32>],
    snapshot: &FrameSnapshot,
    players: &[PlayerInfo],
    ball: Vec3,
) {
    for role in roles.iter_mut() {
        *role = None;
    }
    for (player_idx, player) in players.iter().enumerate() {
        let Some(pos) = snapshot
            .players
            .get(player_idx)
            .and_then(Option::as_ref)
            .filter(|state| state.entity.has_pos)
            .map(|state| state.entity.pos)
        else {
            continue;
        };
        let player_ball_distance = vec_distance(pos, ball);
        let closer_teammates = players
            .iter()
            .enumerate()
            .filter(|(teammate_idx, teammate)| {
                *teammate_idx != player_idx && teammate.team == player.team
            })
            .filter_map(|(teammate_idx, _)| {
                snapshot
                    .players
                    .get(teammate_idx)
                    .and_then(Option::as_ref)
                    .filter(|state| state.entity.has_pos)
                    .map(|state| state.entity.pos)
            })
            .filter(|teammate_pos| vec_distance(*teammate_pos, ball) < player_ball_distance)
            .count();
        if let Some(role) = roles.get_mut(player_idx) {
            *role = Some((closer_teammates + 1) as i32);
        }
    }
}

fn rotation_filled(previous_role: i32, role: i32, team_size: i32) -> bool {
    (previous_role > 1 && role == previous_role - 1) || (previous_role == 1 && role == team_size)
}

fn rotation_cut(previous_role: i32, role: i32, team_size: i32) -> bool {
    previous_role - role > 1 || (previous_role == 1 && role > 1 && role != team_size)
}

fn current_rotation_number(
    rotation_numbers: &mut HashMap<i32, i32>,
    team: i32,
    previous_role: i32,
    role: i32,
) -> i32 {
    let number = rotation_numbers.entry(team).or_insert(0);
    if *number == 0 || previous_role == 1 || role == 1 {
        *number += 1;
    }
    *number
}

fn build_rotation_event_row(
    event_type: &str,
    frame_number: i32,
    event_duration: f32,
    rotation_number: i32,
    player: &PlayerInfo,
    context: &PbpContext,
    player_static_values: &[(String, String)],
    game_id: &str,
    match_guid: &str,
    replay_name: &str,
    map_id: &str,
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
    values.insert("event_duration".to_string(), event_duration.to_string());
    values.insert_i32("rotation_number", rotation_number);
    values.insert("event_team".to_string(), team_name(player.team).to_string());
    insert_seconds_elapsed(&mut values, context, frame_number);
    add_event_player(&mut values, &context.players, 1, &player.name);
    add_pbp_players(&mut values, player_static_values);
    add_frame_state_values(&mut values, context, frame_number, &context.players);
    PbpEventRecord {
        frame_number: Some(frame_number),
        event_type: event_type.to_string(),
        values,
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

pub(crate) fn boost_units(raw_boost: i32) -> i32 {
    ((raw_boost as f32) * 100.0 / 255.0).round() as i32
}

pub(crate) fn raw_boost_units(scaled_boost: i32) -> u8 {
    ((scaled_boost as f32) * 255.0 / 100.0).round() as u8
}

pub(crate) fn post_process_pbp_rows(rows: &mut Vec<PbpEventRecord>, players: &[PlayerInfo]) {
    let touch_types = [
        "touch",
        "turnover",
        "pass",
        "shot",
        "missed-shot",
        "missed-pass",
        "goal",
        "kickoff",
    ];
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

    add_double_commit_events(rows, players, &slot_by_id);

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
    collapse_duplicate_missed_events(rows);

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
        if !matches!(
            rows[idx].event_type.as_str(),
            "shot" | "goal" | "missed-shot"
        ) {
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
                && matches!(prior.event_type.as_str(), "shot" | "goal" | "missed-shot")
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
                flags.insert("off_pass", true);
            }
            if delta <= OFF_CHALLENGE_SECONDS
                && prior.event_type == "fake"
                && row_string(&prior.values, "event_team") == team
            {
                flags.insert("off_fake", true);
            }
            if delta <= OFF_CHALLENGE_SECONDS
                && prior.event_type == "whiff"
                && row_string(&prior.values, "event_team") != team
            {
                flags.insert("off_whiff", true);
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
        if shot_has_rotation_cut_context(rows, idx, &slot_by_id) {
            flags.insert("off_rotation_cut", true);
        }
        for (key, value) in flags {
            rows[idx].values.insert(key.to_string(), value.to_string());
        }
        if !row_string(&rows[idx].values, "event_player_2_id").is_empty() {
            rows[idx]
                .values
                .insert("off_pass".to_string(), "true".to_string());
        }
    }
}

fn add_double_commit_events(
    rows: &mut Vec<PbpEventRecord>,
    players: &[PlayerInfo],
    slot_by_id: &HashMap<String, String>,
) {
    let mut additions = Vec::new();
    let mut last_by_pair: HashMap<(String, String), i32> = HashMap::new();
    for row in rows.iter() {
        if !ball_contact_event(&row.event_type) {
            continue;
        }
        let player_id = row_string(&row.values, "event_player_1_id");
        let team = row_string(&row.values, "event_player_1_team");
        if player_id.is_empty() || team.is_empty() {
            continue;
        }
        let frame_number = row.frame_number.unwrap_or(i32::MAX);
        let Some(player_slot) = slot_by_id.get(&player_id) else {
            continue;
        };
        if !whiff_intent_from_row(&row.values, player_slot, DOUBLE_COMMIT_BALL_DISTANCE) {
            continue;
        }
        let teammate = players
            .iter()
            .filter(|player| player.id != player_id && team_name(player.team) == team)
            .filter_map(|player| {
                if !whiff_intent_from_row(&row.values, &player.slot, DOUBLE_COMMIT_BALL_DISTANCE) {
                    return None;
                }
                let ball_distance =
                    row_f32(&row.values, &format!("{}_distance_to_ball", player.slot))?;
                let teammate_distance = row_f32(
                    &row.values,
                    &format!("{player_slot}_distance_to_{}", player.slot),
                )
                .unwrap_or(f32::MAX);
                (ball_distance <= DOUBLE_COMMIT_BALL_DISTANCE
                    && teammate_distance <= DOUBLE_COMMIT_TEAMMATE_DISTANCE)
                    .then_some((player, ball_distance))
            })
            .min_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let Some((teammate, _)) = teammate else {
            continue;
        };
        let mut pair = [player_id.clone(), teammate.id.clone()];
        pair.sort();
        let key = (pair[0].clone(), pair[1].clone());
        if last_by_pair
            .get(&key)
            .map(|prior_frame| frame_number - *prior_frame <= DOUBLE_COMMIT_COOLDOWN_FRAMES)
            .unwrap_or(false)
        {
            continue;
        }
        last_by_pair.insert(key, frame_number);

        let mut values = row.values.clone();
        values.insert("event_type".to_string(), "double-commit".to_string());
        let player_resistance = double_commit_resistance(&values, player_slot);
        let teammate_resistance = double_commit_resistance(&values, &teammate.slot);
        if teammate_resistance > player_resistance {
            add_event_player(&mut values, players, 1, &teammate.name);
            add_event_player_by_id(&mut values, players, 2, &player_id);
        } else {
            add_event_player_by_id(&mut values, players, 1, &player_id);
            add_event_player(&mut values, players, 2, &teammate.name);
        }
        clear_official_stat_values(&mut values);
        additions.push(PbpEventRecord {
            frame_number: row.frame_number,
            event_type: "double-commit".to_string(),
            values,
        });
    }
    rows.extend(additions);
}

fn whiff_intent_from_row(values: &RowValues, slot: &str, max_distance: f32) -> bool {
    let distance_to_ball =
        row_f32(values, &format!("{slot}_distance_to_ball")).or_else(|| {
            match (row_vec(values, slot, "pos"), row_vec(values, "ball", "pos")) {
                (Some(position), Some(ball)) => Some(vec_distance(position, ball)),
                _ => None,
            }
        });
    if !matches!(distance_to_ball, Some(distance) if distance <= max_distance) {
        return false;
    }

    let Some(position) = row_vec(values, slot, "pos") else {
        return false;
    };
    let Some(velocity) = row_vec(values, slot, "vel") else {
        return false;
    };
    let Some(ball) = row_vec(values, "ball", "pos") else {
        return false;
    };
    let Some(speed_toward_ball) = speed_toward_point(velocity, position, ball) else {
        return false;
    };
    let attempt_input_active = truthy(values.get(&format!("{slot}_boost_active")))
        || truthy(values.get(&format!("{slot}_dodge_active")))
        || truthy(values.get(&format!("{slot}_jump_active")))
        || truthy(values.get(&format!("{slot}_double_jump_active")))
        || truthy(values.get(&format!("{slot}_flipped")));

    speed_toward_ball >= WHIFF_COMMITTED_SPEED_TOWARD_BALL
        || (speed_toward_ball >= WHIFF_MIN_SPEED_TOWARD_BALL && attempt_input_active)
}

fn double_commit_resistance(values: &RowValues, slot: &str) -> f32 {
    let distance = row_f32(values, &format!("{slot}_distance_to_ball"))
        .unwrap_or(DOUBLE_COMMIT_BALL_DISTANCE * 2.0)
        / DOUBLE_COMMIT_BALL_DISTANCE;
    let angle = row_f32(values, &format!("{slot}_angle_to_ball"))
        .map(|value| value.abs())
        .unwrap_or(std::f32::consts::PI)
        / std::f32::consts::PI;
    let role = row_f32(values, &format!("{slot}_rotation_role")).unwrap_or(4.0) * 0.25;
    let speed = match (
        row_vec(values, slot, "vel"),
        row_vec(values, slot, "pos"),
        row_vec(values, "ball", "pos"),
    ) {
        (Some(velocity), Some(position), Some(ball)) => {
            -speed_toward_point(velocity, position, ball).unwrap_or(0.0) / 1000.0
        }
        _ => 0.0,
    };

    distance + angle + role + speed
}

fn add_event_player_by_id(
    values: &mut RowValues,
    players: &[PlayerInfo],
    player_number: usize,
    player_id: &str,
) {
    if let Some(player) = players.iter().find(|player| player.id == player_id) {
        add_event_player(values, players, player_number, &player.name);
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
    let mut dribble_duration = vec![0.0_f32; rows.len()];
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
                dribble_duration[idx] = dribble_duration[idx].max(delta);
            }
            if current_hood && prior_hood && !air_dribble[idx] {
                ground_dribble[idx] = true;
                dribble_duration[idx] = dribble_duration[idx].max(delta);
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
            set_float(
                &mut rows[idx].values,
                "event_duration",
                Some(dribble_duration[idx].max(1.0 / 30.0)),
            );
        }
        if ground_dribble[idx] {
            rows[idx]
                .values
                .insert("ground_dribble".to_string(), "true".to_string());
            set_float(
                &mut rows[idx].values,
                "event_duration",
                Some(dribble_duration[idx].max(1.0 / 30.0)),
            );
        }
        if flick[idx] {
            rows[idx]
                .values
                .insert("flick_shot".to_string(), "true".to_string());
        }
    }
}

fn collapse_duplicate_missed_events(rows: &mut Vec<PbpEventRecord>) {
    let mut seen: HashSet<(String, String, i32)> = HashSet::new();
    rows.retain(|row| {
        if !matches!(row.event_type.as_str(), "missed-shot" | "missed-pass") {
            return true;
        }
        let player_id = row_string(&row.values, "event_player_1_id");
        let frame = row.frame_number.unwrap_or(-1);
        let key = (row.event_type.clone(), player_id, frame / 3);
        seen.insert(key)
    });
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
        "touch"
            | "turnover"
            | "pass"
            | "shot"
            | "missed-shot"
            | "missed-pass"
            | "goal"
            | "kickoff"
            | "challenge"
            | "bump"
    )
}

fn ball_contact_event(event_type: &str) -> bool {
    matches!(
        event_type,
        "touch"
            | "turnover"
            | "pass"
            | "shot"
            | "missed-shot"
            | "missed-pass"
            | "goal"
            | "kickoff"
            | "challenge"
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

fn shot_has_rotation_cut_context(
    rows: &[PbpEventRecord],
    shot_idx: usize,
    slot_by_id: &HashMap<String, String>,
) -> bool {
    let shot = &rows[shot_idx];
    let shooter_id = row_string(&shot.values, "event_player_1_id");
    let Some(slot) = slot_by_id.get(&shooter_id) else {
        return false;
    };
    let Some(current_role) = row_f32(&shot.values, &format!("{slot}_rotation_role")) else {
        return false;
    };
    let team_size = row_f32(&shot.values, "team_size").unwrap_or(0.0);
    if team_size <= 1.0 {
        return false;
    }

    for prior in rows[..shot_idx].iter().rev() {
        if prior.event_type == "kickoff" || prior.event_type == "goal" {
            break;
        }
        if row_string(&prior.values, "event_player_1_id") != shooter_id {
            continue;
        }
        let Some(previous_role) = row_f32(&prior.values, &format!("{slot}_rotation_role")) else {
            continue;
        };
        if (previous_role - current_role).abs() < f32::EPSILON {
            return false;
        }
        if previous_role <= 1.0 {
            return (current_role - team_size).abs() >= f32::EPSILON;
        }
        return current_role < previous_role - 1.0;
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
        let allow_entry_exit = !matches!(
            rows[idx].event_type.as_str(),
            "shot" | "goal" | "missed-shot"
        ) || event_in_offensive_third(&rows[idx].values, &team);
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

        if matches!(row.event_type.as_str(), "shot" | "goal" | "missed-shot") {
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
