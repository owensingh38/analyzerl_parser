use crate::engineering::*;
use crate::*;

pub(crate) fn build_pbp_rows(
    game_id: &str,
    replay: &Replay,
    options: PbpBuildOptions,
) -> Result<(PbpContext, Vec<PbpEventRecord>)> {
    let match_guid = String::new();
    let replay_name = String::new();
    let map_id = String::new();
    let game_time = String::new();
    let context = pbp_context(replay);
    let event_model = EventModel::from_frames(&context);
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
    add_whiff_events(
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
    add_fake_events(
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

    collapse_duplicate_official_saves(&mut rows);
    sort_pbp_rows(&mut rows);

    //Fill default flags and run the remaining row-level feature passes.
    post_process_pbp_rows(&mut rows, &players);
    audit_pbp_stats(game_id, replay, &rows, &context)?;
    if options.rotation_events {
        add_rotation_events(
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
    }
    sort_pbp_rows(&mut rows);
    event_model.finalize_rows(&mut rows);
    sort_pbp_rows(&mut rows);
    for (idx, row) in rows.iter_mut().enumerate() {
        if !row.values.contains_key("observed_frame_number") {
            if let Some(frame) = row.frame_number {
                row.values.insert_i32("observed_frame_number", frame);
            }
        }
        row.values.insert_i32("event_number", (idx + 1) as i32);
    }
    Ok((context, rows))
}

fn sort_pbp_rows(rows: &mut [PbpEventRecord]) {
    rows.sort_by(|left, right| {
        sort_frame_number(left)
            .unwrap_or(i32::MAX)
            .cmp(&sort_frame_number(right).unwrap_or(i32::MAX))
            .then_with(|| {
                event_sort_priority(&left.event_type).cmp(&event_sort_priority(&right.event_type))
            })
            .then_with(|| left.event_type.cmp(&right.event_type))
    });
}

fn sort_frame_number(row: &PbpEventRecord) -> Option<i32> {
    row_i32(&row.values, "observed_frame_number")
        .or(row.frame_number)
        .or_else(|| row_i32(&row.values, "frame_number"))
}

fn event_sort_priority(event_type: &str) -> i32 {
    if event_type == "kickoff" {
        0
    } else {
        1
    }
}
